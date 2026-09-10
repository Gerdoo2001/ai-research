"""Consolidate every matrix row and verify per-run metric provenance."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import _multiclass_bootstrap


SUMMARY_COLUMNS = [
    "run_id", "stage", "status", "architecture", "preprocessing", "seed", "reason",
    "validation_accuracy", "validation_balanced_accuracy", "validation_macro_f1",
    "validation_weighted_f1", "recall_normal", "recall_viral", "recall_bacterial",
    "validation_log_loss", "validation_ece", "validation_brier", "best_epoch",
    "training_duration_seconds", "parameter_count", "model_file_size_bytes",
    "peak_gpu_memory_bytes", "latency_ms_per_image", "throughput_images_per_second",
    "run_dir", "config_hash", "manifest_hash",
]

TUNING_COLUMNS = [
    "study_id", "trial_id", "trial_number", "parent_run_id", "run_id", "status",
    "parameters_json", "validation_macro_f1", "validation_balanced_accuracy",
    "minimum_per_class_recall", "started_at", "completed_at", "duration_seconds",
    "run_dir", "reason",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{column: row.get(column, "") for column in columns} for row in rows])


def metric_value(metrics: dict[str, Any], *keys: str) -> Any:
    value: Any = metrics
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return ""
        value = value[key]
    return value


def main() -> int:
    repo = _multiclass_bootstrap.REPO_ROOT
    results = repo / "model/results"
    matrix = read_csv(results / "run_matrix.csv")
    ledger = {row["run_id"]: row for row in read_csv(results / "experiment_ledger.csv")}
    summary_rows = []
    consistency_errors = []
    for planned in matrix:
        run_id = planned["run_id"]
        tracked = ledger.get(run_id, {})
        status = tracked.get("status") or planned["status"]
        run_dir = Path(tracked["run_dir"]) if tracked.get("run_dir") else None
        metrics = {}
        if run_dir and (run_dir / "metrics.json").exists():
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            if metrics.get("run_id") != run_id:
                consistency_errors.append(f"{run_id}: metrics run_id mismatch")
            if tracked.get("config_hash") and metrics.get("configuration_hash") != tracked["config_hash"]:
                consistency_errors.append(f"{run_id}: configuration hash mismatch")
        if status == "completed" and planned["stage"] in {"baseline", "preprocessing_ablation", "focused_comparison", "robustness"} and not metrics:
            consistency_errors.append(f"{run_id}: completed selection run lacks metrics.json")
        row = {
            "run_id": run_id,
            "stage": planned["stage"],
            "status": status,
            "architecture": tracked.get("architecture") or planned["architecture"],
            "preprocessing": tracked.get("preprocessing") or planned["preprocessing"],
            "seed": tracked.get("seed") or planned["seed"],
            "reason": tracked.get("failure_reason") or planned["reason"],
            "validation_accuracy": metrics.get("accuracy", ""),
            "validation_balanced_accuracy": metrics.get("balanced_accuracy", ""),
            "validation_macro_f1": metrics.get("macro_f1", ""),
            "validation_weighted_f1": metrics.get("weighted_f1", ""),
            "recall_normal": metric_value(metrics, "per_class", "Normal", "recall"),
            "recall_viral": metric_value(metrics, "per_class", "Viral", "recall"),
            "recall_bacterial": metric_value(metrics, "per_class", "Bacterial", "recall"),
            "validation_log_loss": metrics.get("log_loss", ""),
            "validation_ece": metrics.get("expected_calibration_error", ""),
            "validation_brier": metrics.get("brier_score", ""),
            "best_epoch": metrics.get("best_epoch", tracked.get("best_epoch", "")),
            "training_duration_seconds": metrics.get("training_duration_seconds", ""),
            "parameter_count": metrics.get("parameter_count", ""),
            "model_file_size_bytes": metrics.get("model_file_size_bytes", ""),
            "peak_gpu_memory_bytes": metrics.get("peak_gpu_memory_bytes", ""),
            "latency_ms_per_image": metric_value(metrics, "inference", "latency_ms_per_image"),
            "throughput_images_per_second": metric_value(metrics, "inference", "throughput_images_per_second"),
            "run_dir": str(run_dir) if run_dir else "",
            "config_hash": tracked.get("config_hash", ""),
            "manifest_hash": tracked.get("manifest_hash", ""),
        }
        summary_rows.append(row)

    write_csv(results / "summary.csv", SUMMARY_COLUMNS, summary_rows)
    (results / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "rows": summary_rows,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    ranking = [row for row in summary_rows if row["stage"] == "baseline" and row["status"] == "completed"]
    ranking.sort(key=lambda row: float(row["validation_macro_f1"]), reverse=True)
    for index, row in enumerate(ranking, 1):
        row["rank"] = index
    write_csv(results / "model_ranking.csv", ["rank", *SUMMARY_COLUMNS], ranking)

    ablation_control_ids = {
        "baseline_densenet169_s42",
        "baseline_densenet201_s42",
        "baseline_resnet50_s42",
    }
    ablation = [
        row for row in summary_rows
        if (row["run_id"] in ablation_control_ids or row["stage"] == "preprocessing_ablation")
    ]
    write_csv(results / "preprocessing_ablation.csv", SUMMARY_COLUMNS, ablation)
    tuning_rows = []
    tuning_root = results / "tuning"
    if tuning_root.exists():
        for trial_csv in sorted(tuning_root.glob("*/trials.csv")):
            tuning_rows.extend(read_csv(trial_csv))
    for trial in tuning_rows:
        if trial.get("status") != "completed":
            continue
        run_dir_text = trial.get("run_dir", "")
        metrics_path = Path(run_dir_text) / "metrics.json" if run_dir_text else None
        if not metrics_path or not metrics_path.exists():
            consistency_errors.append(f"{trial.get('trial_id')}: completed tuning trial lacks metrics.json")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("run_id") != trial.get("run_id"):
            consistency_errors.append(f"{trial.get('trial_id')}: tuning metrics run_id mismatch")
        try:
            exported = float(trial["validation_macro_f1"])
            original = float(metrics["macro_f1"])
            if abs(exported - original) > 1e-12:
                consistency_errors.append(f"{trial.get('trial_id')}: tuning macro F1 mismatch")
        except (KeyError, TypeError, ValueError):
            consistency_errors.append(f"{trial.get('trial_id')}: invalid tuning macro F1")
    write_csv(results / "tuning_trials.csv", TUNING_COLUMNS, tuning_rows)
    robustness = [row for row in summary_rows if row["stage"] == "robustness"]
    write_csv(results / "seed_robustness.csv", SUMMARY_COLUMNS, robustness)
    test_rows = [row for row in summary_rows if row["stage"] == "locked_test"]
    write_csv(results / "test_results.csv", SUMMARY_COLUMNS, test_rows)
    audit = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "matrix_rows": len(matrix),
        "summary_rows": len(summary_rows),
        "completed_baselines": len(ranking),
        "tuning_trials": len(tuning_rows),
        "consistency_error_count": len(consistency_errors),
        "consistency_errors": consistency_errors,
    }
    (results / "consolidation_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 1 if consistency_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
