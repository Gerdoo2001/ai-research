"""Freeze FT-004 and run paired validation bootstrap comparisons.

This script does not train a model and never reads the locked-test split.  It
uses the already-saved validation probabilities, verifies row/label alignment,
freezes the seed-42 three-member deployment definition, and quantifies paired
uncertainty for the validation comparisons used in the final decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "model" / "results" / "runs"
ENSEMBLE_ROOT = REPO_ROOT / "model" / "results" / "ensemble_confirmation"
OUTPUT_ROOT = REPO_ROOT / "model" / "results" / "final_ft004_ensemble"
CONFIG_PATH = REPO_ROOT / "model" / "configs" / "final_ft004_ensemble.json"

CLASS_NAMES = ("Normal", "Viral", "Bacterial")
PROBABILITY_COLUMNS = (
    "probability_normal",
    "probability_viral",
    "probability_bacterial",
)
SEEDS = (42, 123, 2026)

FT004_MEMBERS = {
    42: (
        "staged_densenet201_pseudo_color_then_histogram_equalization_then_clahe_s42",
        "staged_densenet201_pseudo_color_then_histogram_equalization_s42",
        "ablation_densenet201_histogram_equalization_s42",
    ),
    123: (
        "robustness_densenet201_pseudo_color_then_histogram_equalization_then_clahe_s123",
        "confirmation_densenet201_viridis_ge_s123",
        "confirmation_densenet201_ge_s123",
    ),
    2026: (
        "robustness_densenet201_pseudo_color_then_histogram_equalization_then_clahe_s2026",
        "confirmation_densenet201_viridis_ge_s2026",
        "confirmation_densenet201_ge_s2026",
    ),
}

FT010_RUNS = {
    42: "fine_tuning_densenet201_viridis_ge_clahe_label_smoothing005_s42",
    123: "confirmation_densenet201_viridis_ge_clahe_label_smoothing005_s123",
    2026: "confirmation_densenet201_viridis_ge_clahe_label_smoothing005_s2026",
}

ORIGINAL_RUNS = {seed: FT004_MEMBERS[seed][0] for seed in SEEDS}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            time.sleep(0.2 * (attempt + 1))
    path.write_text(serialized, encoding="utf-8")
    temporary.unlink(missing_ok=True)


def read_predictions(path: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: row["sample_id"])
    labels = np.asarray([int(row["true_index"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [[float(row[column]) for column in PROBABILITY_COLUMNS] for row in rows],
        dtype=np.float64,
    )
    return rows, labels, probabilities


def validate_alignment(
    reference_rows: list[dict[str, str]],
    reference_labels: np.ndarray,
    rows: list[dict[str, str]],
    labels: np.ndarray,
    label: str,
) -> None:
    reference_identity = [
        (row["sample_id"], row["relative_path"], row["true_class"])
        for row in reference_rows
    ]
    identity = [
        (row["sample_id"], row["relative_path"], row["true_class"])
        for row in rows
    ]
    if identity != reference_identity or not np.array_equal(labels, reference_labels):
        raise RuntimeError(f"validation sample/label alignment failure: {label}")


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = np.argmax(probabilities, axis=1)
    f1_values = []
    recalls = []
    for class_index in range(3):
        true_positive = int(np.sum((labels == class_index) & (predictions == class_index)))
        false_positive = int(np.sum((labels != class_index) & (predictions == class_index)))
        false_negative = int(np.sum((labels == class_index) & (predictions != class_index)))
        denominator = (2 * true_positive) + false_positive + false_negative
        f1_values.append((2 * true_positive / denominator) if denominator else 0.0)
        recall_denominator = true_positive + false_negative
        recalls.append(true_positive / recall_denominator if recall_denominator else 0.0)
    return {
        "macro_f1": float(np.mean(f1_values)),
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": float(np.mean(recalls)),
        "viral_recall": float(recalls[1]),
    }


def metric_value(labels: np.ndarray, probabilities: np.ndarray, metric_name: str) -> float:
    return metrics(labels, probabilities)[metric_name]


def load_systems() -> tuple[
    list[dict[str, str]],
    np.ndarray,
    dict[str, dict[int, np.ndarray]],
    dict[str, dict[int, str]],
]:
    systems: dict[str, dict[int, np.ndarray]] = {
        "FT-004 ensemble": {},
        "FT-010 label smoothing": {},
        "Original VIRIDIS -> GE -> CLAHE": {},
    }
    paths: dict[str, dict[int, str]] = {name: {} for name in systems}
    reference_rows: list[dict[str, str]] | None = None
    reference_labels: np.ndarray | None = None

    for seed in SEEDS:
        entries = {
            "FT-004 ensemble": ENSEMBLE_ROOT / f"ensemble_predictions_s{seed}.csv",
            "FT-010 label smoothing": RUNS_ROOT / FT010_RUNS[seed] / "predictions.csv",
            "Original VIRIDIS -> GE -> CLAHE": RUNS_ROOT / ORIGINAL_RUNS[seed] / "predictions.csv",
        }
        for system_name, path in entries.items():
            rows, labels, probabilities = read_predictions(path)
            if reference_rows is None:
                reference_rows, reference_labels = rows, labels
            else:
                assert reference_labels is not None
                validate_alignment(reference_rows, reference_labels, rows, labels, f"{system_name} seed {seed}")
            systems[system_name][seed] = probabilities
            paths[system_name][seed] = str(path)

    assert reference_rows is not None and reference_labels is not None
    return reference_rows, reference_labels, systems, paths


def stratified_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for class_index in range(3):
        class_indices = np.flatnonzero(labels == class_index)
        parts.append(rng.choice(class_indices, size=len(class_indices), replace=True))
    return np.concatenate(parts)


def summarize_samples(samples: np.ndarray, point_difference: float) -> dict[str, Any]:
    lower, upper = np.quantile(samples, [0.025, 0.975])
    probability_positive = float(np.mean(samples > 0.0))
    return {
        "point_difference": float(point_difference),
        "point_difference_percentage_points": float(point_difference * 100.0),
        "bootstrap_mean_difference": float(np.mean(samples)),
        "bootstrap_standard_error": float(np.std(samples, ddof=1)),
        "ci_95_percentile": [float(lower), float(upper)],
        "ci_95_percentage_points": [float(lower * 100.0), float(upper * 100.0)],
        "bootstrap_probability_difference_above_zero": probability_positive,
        "descriptive_interpretation": (
            "95% paired bootstrap interval excludes zero"
            if lower > 0.0 or upper < 0.0
            else "95% paired bootstrap interval includes zero"
        ),
    }


def paired_multiseed_bootstrap(
    labels: np.ndarray,
    systems: dict[str, dict[int, np.ndarray]],
    candidate: str,
    comparator: str,
    iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    metric_names = ("macro_f1", "accuracy", "balanced_accuracy", "viral_recall")
    point = {}
    for metric_name in metric_names:
        candidate_mean = statistics.mean(
            metric_value(labels, systems[candidate][seed], metric_name) for seed in SEEDS
        )
        comparator_mean = statistics.mean(
            metric_value(labels, systems[comparator][seed], metric_name) for seed in SEEDS
        )
        point[metric_name] = candidate_mean - comparator_mean

    rng = np.random.default_rng(bootstrap_seed)
    samples = {name: np.empty(iterations, dtype=np.float64) for name in metric_names}
    for iteration in range(iterations):
        indices = stratified_indices(labels, rng)
        sampled_labels = labels[indices]
        for metric_name in metric_names:
            candidate_value = statistics.mean(
                metric_value(sampled_labels, systems[candidate][seed][indices], metric_name)
                for seed in SEEDS
            )
            comparator_value = statistics.mean(
                metric_value(sampled_labels, systems[comparator][seed][indices], metric_name)
                for seed in SEEDS
            )
            samples[metric_name][iteration] = candidate_value - comparator_value

    return {
        "candidate": candidate,
        "comparator": comparator,
        "paired_unit": "validation image; one class-stratified resample shared across all three seeds",
        "seed_handling": "metric calculated per fixed seed and averaged across seeds 42, 123, and 2026",
        "metrics": {
            metric_name: summarize_samples(samples[metric_name], point[metric_name])
            for metric_name in metric_names
        },
    }


def paired_single_seed_bootstrap(
    labels: np.ndarray,
    candidate_probabilities: np.ndarray,
    comparator_probabilities: np.ndarray,
    candidate: str,
    comparator: str,
    iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    metric_names = ("macro_f1", "accuracy", "balanced_accuracy", "viral_recall")
    point = {
        metric_name: metric_value(labels, candidate_probabilities, metric_name)
        - metric_value(labels, comparator_probabilities, metric_name)
        for metric_name in metric_names
    }
    rng = np.random.default_rng(bootstrap_seed)
    samples = {name: np.empty(iterations, dtype=np.float64) for name in metric_names}
    for iteration in range(iterations):
        indices = stratified_indices(labels, rng)
        sampled_labels = labels[indices]
        for metric_name in metric_names:
            samples[metric_name][iteration] = metric_value(
                sampled_labels, candidate_probabilities[indices], metric_name
            ) - metric_value(sampled_labels, comparator_probabilities[indices], metric_name)
    return {
        "candidate": candidate,
        "comparator": comparator,
        "paired_unit": "validation image, class-stratified paired bootstrap",
        "seed_handling": "seed 42 only; exploratory because FT-009B lacks three-seed confirmation",
        "metrics": {
            metric_name: summarize_samples(samples[metric_name], point[metric_name])
            for metric_name in metric_names
        },
    }


def build_frozen_selection(paths: dict[str, dict[int, str]]) -> dict[str, Any]:
    member_preprocessing = (
        "VIRIDIS -> Global Histogram Equalization (GE) -> CLAHE",
        "VIRIDIS -> Global Histogram Equalization (GE)",
        "Global Histogram Equalization (GE)",
    )
    members = []
    for order, (run_id, preprocessing) in enumerate(zip(FT004_MEMBERS[42], member_preprocessing), 1):
        run_dir = RUNS_ROOT / run_id
        config_path = run_dir / "config.json"
        status_path = run_dir / "status.json"
        metrics_path = run_dir / "metrics.json"
        predictions_path = run_dir / "predictions.csv"
        checkpoint_path = run_dir / "best_checkpoint.pt"
        status = read_json(status_path)
        run_metrics = read_json(metrics_path)
        if status.get("status") != "completed":
            raise RuntimeError(f"cannot freeze incomplete ensemble member: {run_id}")
        members.append(
            {
                "member_order": order,
                "run_id": run_id,
                "architecture": "densenet201",
                "preprocessing": preprocessing,
                "seed": 42,
                "best_epoch": status["best_epoch"],
                "validation_macro_f1": run_metrics["macro_f1"],
                "configuration_path": str(config_path),
                "configuration_sha256": sha256(config_path),
                "configuration_hash_recorded": status["config_hash"],
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256(checkpoint_path),
                "checkpoint_size_bytes": checkpoint_path.stat().st_size,
                "validation_predictions_path": str(predictions_path),
                "validation_predictions_sha256": sha256(predictions_path),
            }
        )

    ensemble_summary = read_json(ENSEMBLE_ROOT / "summary.json")
    return {
        "schema_version": 1,
        "selection_id": "FINAL-VALIDATION-SELECTED-FT004-ENSEMBLE-S42",
        "status": "frozen_validation_selected_research_model",
        "frozen_at": now_iso(),
        "selection_basis": "highest mean validation macro F1 among fully three-seed-confirmed finalists",
        "primary_metric": "macro_f1",
        "selection_split": "validation",
        "validation_rows": 874,
        "class_order": list(CLASS_NAMES),
        "deployable_member_count": 3,
        "deployable_seed": 42,
        "deployable_seed_reason": (
            "Seed 42 is the original predeclared reference seed used to define FT-004; it was not "
            "chosen retrospectively from the three confirmation scores."
        ),
        "inference_rule": {
            "method": "equal arithmetic mean of three class-probability vectors",
            "formula": "p_final = (p_member_1 + p_member_2 + p_member_3) / 3",
            "prediction": "argmax(p_final) in class order [Normal, Viral, Bacterial]",
            "learned_ensemble_weights": False,
            "threshold_tuning": False,
        },
        "fixed_shared_configuration": {
            "input_size": 224,
            "weights": "ImageNet initialization",
            "optimizer": "AdamW",
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "batch_size": 8,
            "loss": "balanced class-weighted cross-entropy",
            "maximum_epochs": 12,
            "early_stopping_patience": 4,
            "training_augmentation": "horizontal flip p=0.5 and rotation +/-5 degrees",
        },
        "members": members,
        "three_seed_confirmation": {
            "seeds": list(SEEDS),
            "summary_path": str(ENSEMBLE_ROOT / "summary.json"),
            "summary_sha256": sha256(ENSEMBLE_ROOT / "summary.json"),
            "mean_macro_f1": ensemble_summary["aggregate"]["macro_f1"]["mean"],
            "sample_std_macro_f1": ensemble_summary["aggregate"]["macro_f1"]["sample_std"],
            "mean_accuracy": ensemble_summary["aggregate"]["accuracy"]["mean"],
            "mean_balanced_accuracy": ensemble_summary["aggregate"]["balanced_accuracy"]["mean"],
            "mean_viral_recall": ensemble_summary["aggregate"]["viral_recall"]["mean"],
            "prediction_paths": paths["FT-004 ensemble"],
        },
        "locked_test_accessed_for_this_selection": False,
        "historical_test_boundary": (
            "The previously opened locked-test metrics belong only to the earlier DenseNet-201 + GE "
            "configuration and are not transferred to FT-004. FT-004 remains validation-selected and "
            "requires genuinely independent external evaluation for an unbiased final performance claim."
        ),
        "intended_use": "research comparison and prototype demonstration; not clinical diagnosis",
    }


def write_comparison_csv(payload: dict[str, Any], path: Path) -> None:
    rows = []
    for comparison in payload["comparisons"]:
        for metric_name, result in comparison["metrics"].items():
            rows.append(
                {
                    "candidate": comparison["candidate"],
                    "comparator": comparison["comparator"],
                    "metric": metric_name,
                    "difference_percentage_points": result["point_difference_percentage_points"],
                    "ci95_lower_percentage_points": result["ci_95_percentage_points"][0],
                    "ci95_upper_percentage_points": result["ci_95_percentage_points"][1],
                    "bootstrap_probability_above_zero": result[
                        "bootstrap_probability_difference_above_zero"
                    ],
                    "interpretation": result["descriptive_interpretation"],
                }
            )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    args = parser.parse_args()
    if args.iterations < 1000:
        raise ValueError("at least 1,000 bootstrap iterations are required")

    rows, labels, systems, paths = load_systems()
    hierarchical_path = REPO_ROOT / "model" / "results" / "ft009b_hierarchical_s42" / "predictions.csv"
    hierarchical_rows, hierarchical_labels, hierarchical_probabilities = read_predictions(hierarchical_path)
    validate_alignment(rows, labels, hierarchical_rows, hierarchical_labels, "FT-009B seed 42")

    selection = build_frozen_selection(paths)
    atomic_json(CONFIG_PATH, selection)
    atomic_json(OUTPUT_ROOT / "frozen_selection.json", selection)

    comparisons = [
        paired_multiseed_bootstrap(
            labels,
            systems,
            "FT-004 ensemble",
            "Original VIRIDIS -> GE -> CLAHE",
            args.iterations,
            args.bootstrap_seed,
        ),
        paired_multiseed_bootstrap(
            labels,
            systems,
            "FT-004 ensemble",
            "FT-010 label smoothing",
            args.iterations,
            args.bootstrap_seed + 1,
        ),
        paired_single_seed_bootstrap(
            labels,
            systems["FT-004 ensemble"][42],
            hierarchical_probabilities,
            "FT-004 ensemble seed 42",
            "FT-009B hierarchical seed 42",
            args.iterations,
            args.bootstrap_seed + 2,
        ),
    ]
    bootstrap_payload = {
        "schema_version": 1,
        "created_at": now_iso(),
        "analysis": "paired class-stratified nonparametric bootstrap",
        "iterations": args.iterations,
        "bootstrap_random_seed": args.bootstrap_seed,
        "validation_rows": len(rows),
        "class_counts": {
            CLASS_NAMES[index]: int(np.sum(labels == index)) for index in range(3)
        },
        "locked_test_accessed": False,
        "comparisons": comparisons,
        "interpretation_boundary": (
            "These intervals quantify validation-sample uncertainty under the fixed observed seeds. "
            "They do not create an independent test result and three seeds are too few for a strong "
            "population-level random-seed inference."
        ),
    }
    atomic_json(OUTPUT_ROOT / "paired_bootstrap.json", bootstrap_payload)
    write_comparison_csv(bootstrap_payload, OUTPUT_ROOT / "paired_bootstrap.csv")

    decision = {
        "schema_version": 1,
        "created_at": now_iso(),
        "chosen_system": "FT-004 three-member probability ensemble",
        "frozen_selection_path": str(OUTPUT_ROOT / "frozen_selection.json"),
        "decision_rule": "highest fully three-seed-confirmed mean validation macro F1",
        "ft004_mean_macro_f1": selection["three_seed_confirmation"]["mean_macro_f1"],
        "ft010_mean_macro_f1": statistics.mean(
            metric_value(labels, systems["FT-010 label smoothing"][seed], "macro_f1")
            for seed in SEEDS
        ),
        "original_mean_macro_f1": statistics.mean(
            metric_value(labels, systems["Original VIRIDIS -> GE -> CLAHE"][seed], "macro_f1")
            for seed in SEEDS
        ),
        "selection_statement": (
            "FT-004 is chosen because it has the highest mean validation macro F1 among the finalists "
            "that completed the same three-seed confirmation. Statistical uncertainty is reported "
            "separately and the selection is not represented as independent clinical validation."
        ),
        "locked_test_accessed": False,
    }
    atomic_json(OUTPUT_ROOT / "decision.json", decision)
    print(json.dumps({"selection": selection, "bootstrap": bootstrap_payload, "decision": decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
