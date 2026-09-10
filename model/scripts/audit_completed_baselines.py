"""Verify persisted baseline artifacts without changing training results.

This audit reloads both checkpoints, recomputes classification metrics from the
saved validation probabilities, checks manifest/config provenance, and inspects
TensorBoard event tags.  It writes only versioned audit outputs under
``model/results/audits``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import _multiclass_bootstrap
from multiclass.config import ExperimentConfig
from multiclass.metrics import calculate_metrics


BASELINE_RUN_IDS = (
    "baseline_densenet121_s42",
    "baseline_densenet169_s42",
    "baseline_densenet201_s42",
    "baseline_resnet50_s42",
    "baseline_resnet101_s42",
    "baseline_resnet152_s42",
    "baseline_yolov5_cls_s42",
    "baseline_yolov8_cls_s42",
)
CLASS_NAMES = ("Normal", "Viral", "Bacterial")
EXPECTED_FILES = (
    "best_checkpoint.pt",
    "last_checkpoint.pt",
    "config.json",
    "status.json",
    "model_metadata.json",
    "history.csv",
    "metrics.json",
    "predictions.csv",
    "sample_errors.csv",
    "classification_report.csv",
    "confusion_matrix.csv",
    "confusion_matrix.png",
    "roc_curves.png",
    "pr_curves.png",
    "calibration_curve.png",
    "training_curves.png",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def close(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return left == right


def compare_metrics(expected: dict[str, Any], observed: dict[str, Any], errors: list[str]) -> None:
    scalar_keys = (
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "matthews_correlation_coefficient",
        "cohen_kappa",
        "log_loss",
        "brier_score",
        "expected_calibration_error",
        "roc_auc_ovr_macro",
        "pr_auc_macro",
    )
    for key in scalar_keys:
        if not close(expected.get(key), observed.get(key)):
            errors.append(f"recomputed metric mismatch: {key}")
    if expected.get("confusion_matrix") != observed.get("confusion_matrix"):
        errors.append("recomputed metric mismatch: confusion_matrix")
    for name in CLASS_NAMES:
        for key in ("precision", "recall", "specificity", "f1", "support", "roc_auc", "pr_auc"):
            if not close(expected["per_class"][name].get(key), observed["per_class"][name].get(key)):
                errors.append(f"recomputed metric mismatch: {name}.{key}")


def tensorboard_tags(path: Path) -> dict[str, Any]:
    event_files = sorted(path.glob("events.out.tfevents.*")) if path.exists() else []
    result: dict[str, Any] = {
        "directory_exists": path.exists(),
        "event_file_count": len(event_files),
        "event_bytes": sum(item.stat().st_size for item in event_files),
        "tags": {},
    }
    if not event_files:
        return result
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        accumulator = EventAccumulator(str(path))
        accumulator.Reload()
        result["tags"] = accumulator.Tags()
    except Exception as error:  # pragma: no cover - diagnostic path
        result["inspection_error"] = f"{type(error).__name__}: {error}"
    return result


def audit_run(run_dir: Path, manifest_hash: str) -> dict[str, Any]:
    import torch

    run_id = run_dir.name
    errors: list[str] = []
    warnings: list[str] = []
    missing = [name for name in EXPECTED_FILES if not (run_dir / name).exists()]
    if missing:
        errors.append(f"missing expected files: {', '.join(missing)}")
    config_payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    config = ExperimentConfig(**config_payload)
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "model_metadata.json").read_text(encoding="utf-8"))
    history = read_csv(run_dir / "history.csv")
    predictions = read_csv(run_dir / "predictions.csv")

    if status.get("status") != "completed":
        errors.append(f"status is {status.get('status')!r}, not completed")
    for source, value in (
        ("status", status.get("config_hash")),
        ("metrics", metrics.get("configuration_hash")),
    ):
        if value != config.config_hash:
            errors.append(f"{source} config hash mismatch")
    if status.get("manifest_hash") != manifest_hash or metrics.get("manifest_version") is None:
        errors.append("manifest provenance mismatch or missing")
    if metrics.get("run_id") != run_id or metrics.get("split") != "validation":
        errors.append("metrics run_id/split mismatch")
    if tuple(metadata.get("output_shape_smoke", ())) != (2, 3):
        errors.append("model output smoke shape is not [2, 3]")
    if int(metadata.get("parameter_count", -1)) != int(metrics.get("parameter_count", -2)):
        errors.append("parameter count mismatch between metadata and metrics")
    if int(metadata.get("trainable_parameter_count", -1)) != int(metrics.get("trainable_parameter_count", -2)):
        errors.append("trainable parameter count mismatch between metadata and metrics")

    if len(predictions) != 874:
        errors.append(f"prediction row count is {len(predictions)}, expected 874")
    if predictions:
        labels = np.asarray([int(row["true_index"]) for row in predictions], dtype=np.int64)
        probabilities = np.asarray(
            [
                [
                    float(row["probability_normal"]),
                    float(row["probability_viral"]),
                    float(row["probability_bacterial"]),
                ]
                for row in predictions
            ],
            dtype=np.float64,
        )
        recomputed = calculate_metrics(labels, probabilities, CLASS_NAMES)
        compare_metrics(recomputed, metrics, errors)

    matrix = np.loadtxt(run_dir / "confusion_matrix.csv", delimiter=",", dtype=int).tolist()
    if matrix != metrics.get("confusion_matrix"):
        errors.append("confusion_matrix.csv does not match metrics.json")
    if not history:
        errors.append("history.csv is empty")
    else:
        best_epoch = int(status["best_epoch"])
        history_by_epoch = {int(row["epoch"]): row for row in history}
        if best_epoch not in history_by_epoch:
            errors.append("best epoch missing from history")
        elif not close(history_by_epoch[best_epoch]["val_macro_f1"], metrics.get("macro_f1")):
            errors.append("best history macro F1 does not match metrics")
        if int(status["current_epoch"]) != int(history[-1]["epoch"]):
            errors.append("last history epoch does not match status")

    checkpoint_facts = {}
    for kind, expected_epoch in (
        ("best", int(status["best_epoch"])),
        ("last", int(status["current_epoch"])),
    ):
        path = run_dir / f"{kind}_checkpoint.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_facts[kind] = {
            "epoch": int(checkpoint["epoch"]),
            "config_hash": checkpoint.get("config_hash"),
            "state_tensor_count": len(checkpoint.get("model_state_dict", {})),
            "file_size_bytes": path.stat().st_size,
        }
        if int(checkpoint["epoch"]) != expected_epoch:
            errors.append(f"{kind} checkpoint epoch mismatch")
        if checkpoint.get("config_hash") != config.config_hash:
            errors.append(f"{kind} checkpoint config hash mismatch")
        embedded = ExperimentConfig(**checkpoint["config"])
        if embedded.config_hash != config.config_hash:
            errors.append(f"{kind} checkpoint embedded config mismatch")
        if not checkpoint.get("model_state_dict"):
            errors.append(f"{kind} checkpoint has no model state")
        del checkpoint

    if int(metrics.get("model_file_size_bytes", -1)) != (run_dir / "best_checkpoint.pt").stat().st_size:
        errors.append("recorded model file size does not match best checkpoint")
    tb = tensorboard_tags(run_dir / "tensorboard")
    nonempty_tags = any(bool(value) for value in tb.get("tags", {}).values())
    if config.tensorboard and not nonempty_tags:
        warnings.append("TensorBoard was enabled but the event file contains no logged tags")

    return {
        "run_id": run_id,
        "architecture": config.architecture,
        "preprocessing": config.preprocessing,
        "config_hash": config.config_hash,
        "manifest_hash": status.get("manifest_hash"),
        "status": status.get("status"),
        "current_epoch": status.get("current_epoch"),
        "best_epoch": status.get("best_epoch"),
        "validation_macro_f1": metrics.get("macro_f1"),
        "prediction_rows": len(predictions),
        "parameter_count": metrics.get("parameter_count"),
        "checkpoint_facts": checkpoint_facts,
        "tensorboard": tb,
        "warnings": warnings,
        "errors": errors,
        "audit_status": "passed" if not errors else "failed",
    }


def main() -> int:
    repo = _multiclass_bootstrap.REPO_ROOT
    results = repo / "model/results"
    manifest = json.loads((repo / "model/data_manifests/split_summary.json").read_text(encoding="utf-8"))
    records = [audit_run(results / "runs" / run_id, manifest["manifest_version_hash"]) for run_id in BASELINE_RUN_IDS]
    output_dir = results / "audits"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "scope": "eight completed controlled validation baselines",
        "manifest_version": manifest["manifest_version"],
        "manifest_hash": manifest["manifest_version_hash"],
        "passed_runs": sum(item["audit_status"] == "passed" for item in records),
        "failed_runs": sum(item["audit_status"] == "failed" for item in records),
        "records": records,
    }
    json_path = output_dir / "completed_baseline_artifact_audit.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv_path = output_dir / "completed_baseline_artifact_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        columns = (
            "run_id",
            "architecture",
            "preprocessing",
            "status",
            "current_epoch",
            "best_epoch",
            "validation_macro_f1",
            "prediction_rows",
            "parameter_count",
            "audit_status",
            "warnings",
            "errors",
        )
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in records:
            row = {key: item.get(key, "") for key in columns}
            row["warnings"] = " | ".join(item["warnings"])
            row["errors"] = " | ".join(item["errors"])
            writer.writerow(row)
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))
    print(json_path)
    print(csv_path)
    return 1 if payload["failed_runs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
