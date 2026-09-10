"""Resolve staged matrix placeholders using validation-only persisted evidence.

Run this only after consolidating results. Each substage is gated on the
required predecessor artifacts and records the exact evidence and commands used
to resolve placeholder rows. The locked test manifest is never loaded here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import _multiclass_bootstrap


FOCUSED_ARCHITECTURES = ("densenet169", "densenet201", "resnet50")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["focused", "tuning", "robustness", "test"], required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    columns = list(rows[0])
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def number(row: dict[str, str], key: str, *, default: float) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def ranking_key(row: dict[str, str]) -> tuple[Any, ...]:
    recalls = [
        number(row, "recall_normal", default=-math.inf),
        number(row, "recall_viral", default=-math.inf),
        number(row, "recall_bacterial", default=-math.inf),
    ]
    return (
        -number(row, "validation_macro_f1", default=-math.inf),
        -number(row, "validation_balanced_accuracy", default=-math.inf),
        -min(recalls),
        number(row, "validation_log_loss", default=math.inf),
        number(row, "model_file_size_bytes", default=math.inf),
        number(row, "latency_ms_per_image", default=math.inf),
        row["run_id"],
    )


def require_completed(rows: list[dict[str, str]], expected: int, label: str) -> list[dict[str, str]]:
    completed = [row for row in rows if row.get("status") == "completed"]
    if len(completed) != expected:
        raise RuntimeError(f"{label} requires {expected} completed rows; found {len(completed)}")
    return completed


def update_rows(matrix: list[dict[str, str]], updates: dict[str, dict[str, Any]]) -> None:
    seen = set()
    for row in matrix:
        if row["run_id"] in updates:
            if row["status"] in {"running", "completed"}:
                for key in ("architecture", "preprocessing", "input_size", "batch_size", "seed"):
                    if key in updates[row["run_id"]] and row.get(key) != str(updates[row["run_id"]][key]):
                        raise RuntimeError(
                            f"resolved value changed for active/completed row {row['run_id']}: "
                            f"{key}={row.get(key)!r} versus {updates[row['run_id']][key]!r}"
                        )
                seen.add(row["run_id"])
                continue
            row.update({key: str(value) for key, value in updates[row["run_id"]].items()})
            seen.add(row["run_id"])
    missing = set(updates) - seen
    if missing:
        raise KeyError(f"matrix rows not found: {sorted(missing)}")


def save_resolution(path: Path, stage: str, payload: dict[str, Any]) -> None:
    existing: dict[str, Any] = {"schema_version": 1, "stages": {}}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.setdefault("stages", {})[stage] = {
        "resolved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **payload,
    }
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


def resolve_focused(matrix: list[dict[str, str]], summary: list[dict[str, str]], results: Path) -> dict[str, Any]:
    baselines = require_completed(
        [row for row in summary if row["stage"] == "baseline" and row["architecture"] != "yolov10_cls"],
        8,
        "focused resolution baseline gate",
    )
    ablations = require_completed(
        [row for row in summary if row["stage"] == "preprocessing_ablation" and row["status"] != "unsupported"],
        9,
        "focused resolution preprocessing gate",
    )
    representatives = FOCUSED_ARCHITECTURES
    best_by_representative = []
    preprocessing_rankings = {}
    for architecture in representatives:
        control = next(row for row in baselines if row["architecture"] == architecture)
        candidates = [control, *[row for row in ablations if row["architecture"] == architecture]]
        ordered = sorted(candidates, key=ranking_key)
        best_by_representative.append(ordered[0])
        preprocessing_rankings[architecture] = [row["run_id"] for row in ordered]
    selected = sorted(best_by_representative, key=ranking_key)
    updates: dict[str, dict[str, Any]] = {}
    for index, evidence in enumerate(selected, 1):
        run_id = f"focused_top{index}_bestprep_s42"
        source_matrix = next(row for row in matrix if row["run_id"] == evidence["run_id"])
        updates[run_id] = {
            "architecture": evidence["architecture"],
            "preprocessing": evidence["preprocessing"],
            "batch_size": source_matrix["batch_size"],
            "status": "skipped",
            "dependencies": "Stage B/C validation results resolved",
            "estimated_duration": "0:00:00",
            "estimated_storage_bytes": "0",
            "reason": f"reuse completed validation evidence {evidence['run_id']}; redundant retraining prohibited",
        }
    update_rows(matrix, updates)
    payload = {
        "selection_data": "validation only",
        "primary_metric": "macro F1",
        "tie_breakers": ["balanced accuracy", "minimum per-class recall", "log loss", "size", "latency"],
        "representative_models": list(representatives),
        "preprocessing_rankings": preprocessing_rankings,
        "selected_completed_evidence": [row["run_id"] for row in selected],
        "commands": [],
        "reason": "All four preprocessing conditions were already evaluated on every representative; focused rows are aliases only.",
    }
    return payload


def resolve_tuning(matrix: list[dict[str, str]], summary: list[dict[str, str]], results: Path) -> dict[str, Any]:
    candidates = [
        row for row in summary
        if row["status"] == "completed"
        and row["stage"] in {"baseline", "preprocessing_ablation", "focused_comparison"}
        and row["architecture"] in FOCUSED_ARCHITECTURES
    ]
    architectures = {row["architecture"] for row in candidates}
    if architectures != set(FOCUSED_ARCHITECTURES):
        raise RuntimeError(
            "tuning resolution requires completed evidence for the three preregistered "
            f"focused architectures; found {sorted(architectures)}"
        )
    best_by_architecture = []
    for architecture in sorted(architectures):
        best_by_architecture.append(sorted([row for row in candidates if row["architecture"] == architecture], key=ranking_key)[0])
    selected = sorted(best_by_architecture, key=ranking_key)
    updates: dict[str, dict[str, Any]] = {}
    plan_rows = []
    for index, parent in enumerate(selected, 1):
        run_id = f"tuning_top{index}"
        source_matrix = next(row for row in matrix if row["run_id"] == parent["run_id"])
        command = (
            f".venv\\Scripts\\python.exe model\\scripts\\run_tuning.py --matrix-run-id {run_id} "
            f"--parent-run-id {parent['run_id']} --trials 4 --epochs 6"
        )
        updates[run_id] = {
            "architecture": parent["architecture"],
            "preprocessing": parent["preprocessing"],
            "input_size": source_matrix["input_size"],
            "batch_size": source_matrix["batch_size"],
            "status": "planned",
            "dependencies": f"Optuna 4.9.0; parent {parent['run_id']}; locked validation only",
            "estimated_duration": "4 trials x 6 epochs; architecture-dependent",
            "estimated_storage_bytes": "4 trial artifact sets",
            "reason": f"parent_run_id={parent['run_id']}; validation rank {index} after best-per-architecture filtering",
        }
        plan_rows.append({"matrix_run_id": run_id, "parent_run_id": parent["run_id"], "command": command})
    update_rows(matrix, updates)
    (results / "tuning_plan.json").write_text(
        json.dumps({"schema_version": 1, "selection_data": "validation only", "studies": plan_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "selection_data": "validation only",
        "eligible_architectures": list(FOCUSED_ARCHITECTURES),
        "eligibility_reason": (
            "DenseNet169 is the baseline macro-F1 leader, DenseNet201 is the baseline Viral-recall "
            "leader, and ResNet50 is the preregistered performance-efficiency representative."
        ),
        "best_per_architecture": [row["run_id"] for row in sorted(best_by_architecture, key=ranking_key)],
        "selected_parents": [row["run_id"] for row in selected],
        "studies": plan_rows,
    }


def resolve_robustness(matrix: list[dict[str, str]], summary: list[dict[str, str]], results: Path, repo: Path) -> dict[str, Any]:
    studies = []
    for index in (1, 2, 3):
        path = results / "tuning" / f"tuning_top{index}" / "study_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"tuning study is incomplete: {path}")
        studies.append(json.loads(path.read_text(encoding="utf-8")))

    def study_key(study: dict[str, Any]) -> tuple[Any, ...]:
        metrics = json.loads((Path(study["best_run_dir"]) / "metrics.json").read_text(encoding="utf-8"))
        recalls = [metrics["per_class"][name]["recall"] for name in ("Normal", "Viral", "Bacterial")]
        return (
            -float(metrics["macro_f1"]),
            -float(metrics["balanced_accuracy"]),
            -float(min(recalls)),
            float(metrics["log_loss"]),
            float(metrics["model_file_size_bytes"]),
            float(metrics["inference"]["latency_ms_per_image"]),
            study["study_id"],
        )

    selected = sorted(studies, key=study_key)[0]
    best_config_path = results / "tuning" / selected["study_id"] / "best_config.json"
    config = json.loads(best_config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "run_id": None,
            "epochs": 12,
            "seed": 42,
            "early_stopping_patience": 4,
            "controlled_stop_after_epoch": None,
            "smoke_test": False,
            "max_train_samples": None,
            "max_val_samples": None,
        }
    )
    final_config_path = repo / "model/configs/final_selected.json"
    final_config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    resource = json.loads((results / "training_resource_estimate.json").read_text(encoding="utf-8"))
    resource_by_arch = {row["architecture"]: row for row in resource["results"]}
    measured = resource_by_arch[config["architecture"]]
    updates = {}
    for seed in (42, 123, 2026):
        updates[f"final_selected_s{seed}"] = {
            "architecture": config["architecture"],
            "preprocessing": config["preprocessing"],
            "input_size": config["input_size"],
            "batch_size": config["batch_size"],
            "status": "planned",
            "dependencies": f"frozen validation-selected config from {selected['best_run_id']}; CUDA",
            "estimated_duration": "12 epochs maximum with early stopping",
            "estimated_storage_bytes": measured["estimated_run_storage_bytes"],
            "reason": "independent robustness retrain; no test access",
        }
    update_rows(matrix, updates)
    return {
        "selection_data": "validation only",
        "selected_study": selected["study_id"],
        "selected_tuning_run": selected["best_run_id"],
        "selected_validation_macro_f1": selected["best_validation_macro_f1"],
        "final_config": str(final_config_path),
        "robustness_seeds": [42, 123, 2026],
        "command": (
            ".venv\\Scripts\\python.exe model\\scripts\\run_benchmark.py --stage robustness "
            "--config model\\configs\\final_selected.json --resume"
        ),
    }


def resolve_test(matrix: list[dict[str, str]], summary: list[dict[str, str]], results: Path) -> dict[str, Any]:
    existing_lock = results / "final_selection.json"
    if existing_lock.exists():
        existing = json.loads(existing_lock.read_text(encoding="utf-8"))
        if existing.get("status") == "locked_test_evaluated":
            raise RuntimeError("locked test has already been evaluated; selection cannot be reopened")
    robustness = require_completed(
        [row for row in summary if row["stage"] == "robustness"],
        3,
        "locked-test freeze gate",
    )
    by_seed = {int(row["seed"]): row for row in robustness}
    if set(by_seed) != {42, 123, 2026}:
        raise RuntimeError(f"robustness seeds must be 42, 123, 2026; found {sorted(by_seed)}")
    scores = [float(by_seed[seed]["validation_macro_f1"]) for seed in (42, 123, 2026)]
    representative = by_seed[42]
    run_dir = Path(representative["run_dir"])
    checkpoint = run_dir / "best_checkpoint.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    selection_lock = {
        "schema_version": 1,
        "status": "selection_frozen_before_test",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_data": "validation only",
        "architecture": representative["architecture"],
        "preprocessing": representative["preprocessing"],
        "representative_seed_policy": "predeclared seed 42 checkpoint; not chosen by test performance",
        "selected_run_id": representative["run_id"],
        "selected_run_dir": str(run_dir),
        "selected_checkpoint": str(checkpoint),
        "configuration_hash": representative["config_hash"],
        "manifest_hash": representative["manifest_hash"],
        "validation_macro_f1_by_seed": {str(seed): float(by_seed[seed]["validation_macro_f1"]) for seed in (42, 123, 2026)},
        "validation_macro_f1_mean": mean(scores),
        "validation_macro_f1_standard_deviation": stdev(scores),
        "test_metrics": None,
    }
    lock_path = results / "final_selection.json"
    lock_path.write_text(json.dumps(selection_lock, indent=2) + "\n", encoding="utf-8")
    update_rows(
        matrix,
        {
            "final_locked_test": {
                "architecture": representative["architecture"],
                "preprocessing": representative["preprocessing"],
                "seed": 42,
                "input_size": next(row for row in matrix if row["run_id"] == representative["run_id"])["input_size"],
                "batch_size": next(row for row in matrix if row["run_id"] == representative["run_id"])["batch_size"],
                "status": "planned",
                "dependencies": f"frozen selection {lock_path}; explicit --allow-locked-test",
                "estimated_duration": "single inference pass",
                "estimated_storage_bytes": "evaluation artifacts only",
                "reason": "selection frozen from validation evidence; representative seed policy fixed before test",
            }
        },
    )
    return {
        "selection_lock": str(lock_path),
        "representative_run": representative["run_id"],
        "validation_macro_f1_mean": mean(scores),
        "validation_macro_f1_standard_deviation": stdev(scores),
        "command": (
            f".venv\\Scripts\\python.exe model\\scripts\\evaluate_multiclass.py --run-dir \"{run_dir}\" "
            "--split test --checkpoint best --allow-locked-test --final-selection model\\results\\final_selection.json"
        ),
    }


def main() -> int:
    args = parse_args()
    repo = _multiclass_bootstrap.REPO_ROOT
    results = repo / "model/results"
    matrix_path = results / "run_matrix.csv"
    summary_path = results / "summary.csv"
    matrix = read_csv(matrix_path)
    summary = read_csv(summary_path)
    if args.stage == "focused":
        payload = resolve_focused(matrix, summary, results)
    elif args.stage == "tuning":
        payload = resolve_tuning(matrix, summary, results)
    elif args.stage == "robustness":
        payload = resolve_robustness(matrix, summary, results, repo)
    else:
        payload = resolve_test(matrix, summary, results)
    write_csv_atomic(matrix_path, matrix)
    save_resolution(results / "selection_resolution.json", args.stage, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
