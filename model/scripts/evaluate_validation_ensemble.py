"""Evaluate the frozen three-member validation ensemble without test access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model" / "src"))

from multiclass.metrics import calculate_metrics  # noqa: E402


CLASS_NAMES = ("Normal", "Viral", "Bacterial")
PROBABILITY_COLUMNS = ("probability_normal", "probability_viral", "probability_bacterial")
MEMBERS_BY_SEED = {
    42: [
        "staged_densenet201_pseudo_color_then_histogram_equalization_then_clahe_s42",
        "staged_densenet201_pseudo_color_then_histogram_equalization_s42",
        "ablation_densenet201_histogram_equalization_s42",
    ],
    123: [
        "robustness_densenet201_pseudo_color_then_histogram_equalization_then_clahe_s123",
        "confirmation_densenet201_viridis_ge_s123",
        "confirmation_densenet201_ge_s123",
    ],
    2026: [
        "robustness_densenet201_pseudo_color_then_histogram_equalization_then_clahe_s2026",
        "confirmation_densenet201_viridis_ge_s2026",
        "confirmation_densenet201_ge_s2026",
    ],
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2 * (attempt + 1))


def read_predictions(run_id: str) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, Path]:
    path = REPO_ROOT / "model" / "results" / "runs" / run_id / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing validation predictions: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: row["sample_id"])
    labels = np.asarray([int(row["true_index"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [[float(row[column]) for column in PROBABILITY_COLUMNS] for row in rows], dtype=np.float64
    )
    return rows, labels, probabilities, path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="model/results/ensemble_confirmation",
        help="Repository-relative output directory",
    )
    args = parser.parse_args()
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed: list[dict[str, object]] = []

    for seed, members in MEMBERS_BY_SEED.items():
        loaded = [read_predictions(run_id) for run_id in members]
        base_rows, base_labels = loaded[0][0], loaded[0][1]
        base_ids = [row["sample_id"] for row in base_rows]
        for rows, labels, _, _ in loaded[1:]:
            if [row["sample_id"] for row in rows] != base_ids:
                raise RuntimeError(f"sample alignment failure for seed {seed}")
            if not np.array_equal(labels, base_labels):
                raise RuntimeError(f"label alignment failure for seed {seed}")
        ensemble_probabilities = np.mean([entry[2] for entry in loaded], axis=0)
        metrics = calculate_metrics(base_labels, ensemble_probabilities, CLASS_NAMES)
        metrics.update(
            seed=seed,
            split="validation",
            ensemble_method="equal arithmetic mean of three class-probability vectors",
            locked_test_accessed=False,
        )
        member_records = []
        for run_id, (_, _, probabilities, path) in zip(members, loaded):
            member_metrics = calculate_metrics(base_labels, probabilities, CLASS_NAMES)
            member_records.append(
                {
                    "run_id": run_id,
                    "predictions_sha256": sha256(path),
                    "macro_f1": member_metrics["macro_f1"],
                    "accuracy": member_metrics["accuracy"],
                    "balanced_accuracy": member_metrics["balanced_accuracy"],
                    "viral_recall": member_metrics["per_class"]["Viral"]["recall"],
                }
            )
        prediction_path = output_dir / f"ensemble_predictions_s{seed}.csv"
        with prediction_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "sample_id", "relative_path", "true_index", "true_class", "predicted_index",
                "predicted_class", *PROBABILITY_COLUMNS,
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row, probabilities in zip(base_rows, ensemble_probabilities):
                predicted = int(np.argmax(probabilities))
                writer.writerow(
                    {
                        "sample_id": row["sample_id"],
                        "relative_path": row["relative_path"],
                        "true_index": row["true_index"],
                        "true_class": row["true_class"],
                        "predicted_index": predicted,
                        "predicted_class": CLASS_NAMES[predicted],
                        **{column: float(value) for column, value in zip(PROBABILITY_COLUMNS, probabilities)},
                    }
                )
        atomic_json(output_dir / f"ensemble_metrics_s{seed}.json", metrics)
        per_seed.append(
            {
                "seed": seed,
                "members": member_records,
                "metrics": metrics,
                "predictions_path": str(prediction_path),
            }
        )

    metric_paths = {
        "macro_f1": lambda record: record["metrics"]["macro_f1"],
        "accuracy": lambda record: record["metrics"]["accuracy"],
        "balanced_accuracy": lambda record: record["metrics"]["balanced_accuracy"],
        "viral_recall": lambda record: record["metrics"]["per_class"]["Viral"]["recall"],
    }
    aggregate = {}
    for name, getter in metric_paths.items():
        values = [float(getter(record)) for record in per_seed]
        aggregate[name] = {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values),
            "minimum": min(values),
            "maximum": max(values),
            "values_by_seed": {str(record["seed"]): value for record, value in zip(per_seed, values)},
        }
    summary = {
        "schema_version": 1,
        "created_at": now(),
        "experiment_id": "FT-004-ENS-CONFIRMATION",
        "task": "three-class validation probability ensemble",
        "class_order": list(CLASS_NAMES),
        "method": "equal arithmetic mean; frozen three-member definition",
        "member_preprocessing": ["VIRIDIS -> GE -> CLAHE", "VIRIDIS -> GE", "GE"],
        "seeds": [42, 123, 2026],
        "locked_test_accessed": False,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "interpretation": (
            "Validation confirmation only. The locked test remains prohibited until the ensemble "
            "membership and selection rule are formally frozen."
        ),
    }
    atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
