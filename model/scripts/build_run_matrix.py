"""Build the controlled staged experiment matrix from measured resources."""

from __future__ import annotations

import csv
import json
from datetime import timedelta
from pathlib import Path

import _multiclass_bootstrap


COLUMNS = [
    "run_id", "stage", "architecture", "preprocessing", "tuning", "seed",
    "input_size", "batch_size", "status", "dependencies", "estimated_duration",
    "estimated_storage_bytes", "reason",
]


def duration(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def main() -> int:
    repo = _multiclass_bootstrap.REPO_ROOT
    results = repo / "model/results"
    resource = json.loads((results / "training_resource_estimate.json").read_text(encoding="utf-8"))
    resource_by_arch = {item["architecture"]: item for item in resource["results"]}
    pilot_history = list(csv.DictReader(
        (results / "runs/pilot_densenet121_full_epoch/history.csv").open("r", encoding="utf-8-sig", newline="")
    ))
    observed_epoch = float(pilot_history[-1]["elapsed_seconds"])
    densenet_compute_epoch = (
        resource_by_arch["densenet121"]["estimated_train_epoch_compute_seconds"]
        + resource_by_arch["densenet121"]["estimated_validation_epoch_compute_seconds"]
    )
    observed_non_compute_overhead = max(0.0, observed_epoch - densenet_compute_epoch)

    rows = []
    ledger = list(csv.DictReader(
        (results / "experiment_ledger.csv").open("r", encoding="utf-8-sig", newline="")
    ))
    ledger_by_id = {item["run_id"]: item for item in ledger}
    for item in ledger:
        run_id = item["run_id"]
        if not run_id.startswith(("pilot_", "smoke_")):
            continue
        stage = "pilot" if run_id.startswith("pilot_") else "smoke"
        run_dir = Path(item["run_dir"])
        storage = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file()) if run_dir.exists() else 0
        rows.append(
            {
                "run_id": run_id,
                "stage": stage,
                "architecture": item["architecture"],
                "preprocessing": item["preprocessing"],
                "tuning": "none",
                "seed": item["seed"],
                "input_size": 224,
                "batch_size": 3 if stage == "smoke" else 16,
                "status": item["status"],
                "dependencies": "isolated PyTorch CUDA environment",
                "estimated_duration": "actual artifact",
                "estimated_storage_bytes": storage,
                "reason": "pipeline diagnostic only; not model-selection evidence",
            }
        )

    for architecture in (
        "densenet121", "densenet169", "densenet201", "resnet50", "resnet101", "resnet152",
        "yolov5_cls", "yolov8_cls",
    ):
        measured = resource_by_arch[architecture]
        compute_epoch = (
            measured["estimated_train_epoch_compute_seconds"]
            + measured["estimated_validation_epoch_compute_seconds"]
        )
        end_to_end_estimate = (compute_epoch + observed_non_compute_overhead) * 12 + 20.0
        run_id = f"baseline_{architecture}_s42"
        rows.append(
            {
                "run_id": run_id,
                "stage": "baseline",
                "architecture": architecture,
                "preprocessing": "baseline",
                "tuning": "none",
                "seed": 42,
                "input_size": 224,
                "batch_size": measured["batch_size"],
                "status": ledger_by_id.get(run_id, {}).get("status", "planned"),
                "dependencies": "ImageNet/official classification weights; locked manifests; CUDA",
                "estimated_duration": duration(end_to_end_estimate),
                "estimated_storage_bytes": measured["estimated_run_storage_bytes"],
                "reason": "controlled architecture benchmark; validation selection only",
            }
        )
    rows.append(
        {
            "run_id": "baseline_yolov10_cls_s42", "stage": "baseline", "architecture": "yolov10_cls",
            "preprocessing": "baseline", "tuning": "none", "seed": 42, "input_size": 224,
            "batch_size": "", "status": "unsupported", "dependencies": "genuine classification head absent",
            "estimated_duration": "0:00:00", "estimated_storage_bytes": 0,
            "reason": "Ultralytics 8.3.168 provides v10Detect YAMLs but no YOLOv10 classification YAML/checkpoint",
        }
    )

    for architecture in ("densenet169", "densenet201", "resnet50"):
        measured = resource_by_arch[architecture]
        estimated = duration(
            (
                measured["estimated_train_epoch_compute_seconds"]
                + measured["estimated_validation_epoch_compute_seconds"]
                + observed_non_compute_overhead
            )
            * 12
            + 20.0
        )
        for preprocessing in ("clahe", "histogram_equalization", "pseudo_color"):
            run_id = f"ablation_{architecture}_{preprocessing}_s42"
            rows.append(
                {
                    "run_id": run_id,
                    "stage": "preprocessing_ablation",
                    "architecture": architecture,
                    "preprocessing": preprocessing,
                    "tuning": "none",
                    "seed": 42,
                    "input_size": 224,
                    "batch_size": measured["batch_size"],
                    "status": ledger_by_id.get(run_id, {}).get("status", "planned"),
                    "dependencies": "locked manifests; completed baseline control; CUDA; OpenCV/Pillow",
                    "estimated_duration": estimated,
                    "estimated_storage_bytes": measured["estimated_run_storage_bytes"],
                    "reason": f"independent transform comparison against baseline_{architecture}_s42",
                }
            )
    rows.extend(
        [
            {"run_id": "ablation_pca_unsupported", "stage": "preprocessing_ablation", "architecture": "not_applicable",
             "preprocessing": "pca", "tuning": "none", "seed": 42, "input_size": 224, "batch_size": "",
             "status": "unsupported", "dependencies": "justified train-only feature-space design absent",
             "estimated_duration": "0:00:00", "estimated_storage_bytes": 0,
             "reason": "raw-image PCA is not scientifically justified; feature-space experiment deferred"},
            {"run_id": "ablation_film_unsupported", "stage": "preprocessing_ablation", "architecture": "not_applicable",
             "preprocessing": "film", "tuning": "none", "seed": 42, "input_size": 224, "batch_size": "",
             "status": "unsupported", "dependencies": "verified definition/conditioning signal absent",
             "estimated_duration": "0:00:00", "estimated_storage_bytes": 0,
             "reason": "FILM remains unresolved in repository evidence"},
        ]
    )
    for rank in (1, 2, 3):
        rows.append(
            {"run_id": f"focused_top{rank}_bestprep_s42", "stage": "focused_comparison", "architecture": "TBD_AFTER_BASELINE",
             "preprocessing": "TBD_AFTER_ABLATION", "tuning": "none", "seed": 42, "input_size": 224,
             "batch_size": "TBD", "status": "planned", "dependencies": "Stage B/C validation results",
             "estimated_duration": "TBD", "estimated_storage_bytes": "TBD",
             "reason": "resolved only from validation macro F1 and declared tie-breakers"}
        )
        rows.append(
            {"run_id": f"tuning_top{rank}", "stage": "tuning", "architecture": "TBD_AFTER_FOCUSED",
             "preprocessing": "TBD_AFTER_FOCUSED", "tuning": "Optuna TPE, validation only", "seed": 42,
             "input_size": "TBD", "batch_size": "TBD", "status": "planned", "dependencies": "Optuna; Stage D results",
             "estimated_duration": "TBD", "estimated_storage_bytes": "TBD",
             "reason": "not SAS Tuning; every completed/pruned/failed trial will be retained"}
        )
    for seed in (42, 123, 2026):
        rows.append(
            {"run_id": f"final_selected_s{seed}", "stage": "robustness", "architecture": "TBD_AFTER_TUNING",
             "preprocessing": "TBD_AFTER_TUNING", "tuning": "selected", "seed": seed, "input_size": "TBD",
             "batch_size": "TBD", "status": "planned", "dependencies": "final validation-selected configuration",
             "estimated_duration": "TBD", "estimated_storage_bytes": "TBD",
             "reason": "three-seed robustness; independently retrained after tuning selection"}
        )
    rows.append(
        {"run_id": "final_locked_test", "stage": "locked_test", "architecture": "TBD_AFTER_ROBUSTNESS",
         "preprocessing": "TBD_AFTER_ROBUSTNESS", "tuning": "final", "seed": "selected ensemble/reporting policy",
         "input_size": "TBD", "batch_size": "TBD", "status": "planned",
         "dependencies": "all selection frozen; explicit --allow-locked-test",
         "estimated_duration": "TBD", "estimated_storage_bytes": "TBD",
         "reason": "single final locked-test evaluation; never used for selection"}
    )

    output = results / "run_matrix.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "row_count": len(rows),
        "observed_densenet121_epoch_seconds": observed_epoch,
        "observed_non_compute_overhead_seconds_per_epoch": observed_non_compute_overhead,
        "planned_baseline_count": 8,
        "unsupported_baseline_count": 1,
        "planned_preprocessing_ablation_count": 9,
        "preprocessing_representatives": ["densenet169", "densenet201", "resnet50"],
        "estimated_baseline_duration_seconds": sum(
            (
                resource_by_arch[a]["estimated_train_epoch_compute_seconds"]
                + resource_by_arch[a]["estimated_validation_epoch_compute_seconds"]
                + observed_non_compute_overhead
            ) * 12 + 20.0
            for a in resource_by_arch
        ),
        "estimated_baseline_storage_bytes": resource["estimated_baseline_matrix_storage_bytes"],
    }
    (results / "run_matrix_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
