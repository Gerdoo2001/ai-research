"""Analyze the completed representative preprocessing matrix on validation only."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import _multiclass_bootstrap
import numpy as np


ARCHITECTURES = ("densenet169", "densenet201", "resnet50")
PREPROCESSING = ("baseline", "clahe", "histogram_equalization", "pseudo_color")
CLASS_NAMES = ("Normal", "Viral", "Bacterial")
BOOTSTRAP_REPLICATES = 2000


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_predictions(path: Path):
    rows = read_csv(path)
    sample_ids = [row["sample_id"] for row in rows]
    labels = np.asarray([int(row["true_index"]) for row in rows], dtype=np.int64)
    predictions = np.asarray([int(row["predicted_index"]) for row in rows], dtype=np.int64)
    return sample_ids, labels, predictions


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    for class_index in range(3):
        true_positive = int(((labels == class_index) & (predictions == class_index)).sum())
        false_positive = int(((labels != class_index) & (predictions == class_index)).sum())
        false_negative = int(((labels == class_index) & (predictions != class_index)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append((2 * true_positive / denominator) if denominator else 0.0)
    return float(np.mean(scores))


def stratified_delta_ci(labels, control_predictions, candidate_predictions, *, seed: int):
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(labels == index) for index in range(3)]
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in by_class]
        )
        deltas[replicate] = macro_f1(labels[sampled], candidate_predictions[sampled]) - macro_f1(
            labels[sampled], control_predictions[sampled]
        )
    return {
        "lower": float(np.quantile(deltas, 0.025)),
        "median": float(np.quantile(deltas, 0.5)),
        "upper": float(np.quantile(deltas, 0.975)),
        "replicates": BOOTSTRAP_REPLICATES,
        "method": "paired class-stratified nonparametric bootstrap on the fixed validation set",
    }


def ranking_key(record):
    return (
        -record["validation_macro_f1"],
        -record["validation_balanced_accuracy"],
        -record["minimum_per_class_recall"],
        record["validation_log_loss"],
        record["model_file_size_bytes"],
        record["latency_ms_per_image"],
        record["run_id"],
    )


def main() -> int:
    repo = _multiclass_bootstrap.REPO_ROOT
    results = repo / "model" / "results"
    runs = results / "runs"
    summary = read_csv(results / "summary.csv")
    by_key = {
        (row["architecture"], row["preprocessing"]): row
        for row in summary
        if row["status"] == "completed" and row["architecture"] in ARCHITECTURES
    }
    missing = [key for key in ((a, p) for a in ARCHITECTURES for p in PREPROCESSING) if key not in by_key]
    if missing:
        raise RuntimeError(f"preprocessing matrix is incomplete: {missing}")

    records = []
    for arch_index, architecture in enumerate(ARCHITECTURES):
        control = by_key[(architecture, "baseline")]
        control_ids, control_labels, control_predictions = load_predictions(
            Path(control["run_dir"]) / "predictions.csv"
        )
        for prep_index, preprocessing in enumerate(PREPROCESSING):
            row = by_key[(architecture, preprocessing)]
            metrics = json.loads((Path(row["run_dir"]) / "metrics.json").read_text(encoding="utf-8"))
            sample_ids, labels, predictions = load_predictions(Path(row["run_dir"]) / "predictions.csv")
            if sample_ids != control_ids or not np.array_equal(labels, control_labels):
                raise RuntimeError(f"validation prediction alignment mismatch: {row['run_id']}")
            observed_delta = float(metrics["macro_f1"]) - float(control["validation_macro_f1"])
            ci = (
                {"lower": 0.0, "median": 0.0, "upper": 0.0, "replicates": 0, "method": "control"}
                if preprocessing == "baseline"
                else stratified_delta_ci(
                    labels,
                    control_predictions,
                    predictions,
                    seed=4200 + arch_index * 10 + prep_index,
                )
            )
            interpretation = "control"
            if preprocessing != "baseline":
                if ci["lower"] > 0:
                    interpretation = "bootstrap interval favors candidate"
                elif ci["upper"] < 0:
                    interpretation = "bootstrap interval favors baseline"
                else:
                    interpretation = "bootstrap interval includes no difference"
            records.append(
                {
                    "run_id": row["run_id"],
                    "architecture": architecture,
                    "preprocessing": preprocessing,
                    "validation_macro_f1": float(metrics["macro_f1"]),
                    "delta_macro_f1_vs_same_model_baseline": observed_delta,
                    "delta_bootstrap_ci_lower": ci["lower"],
                    "delta_bootstrap_ci_median": ci["median"],
                    "delta_bootstrap_ci_upper": ci["upper"],
                    "validation_balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "minimum_per_class_recall": min(
                        float(metrics["per_class"][name]["recall"]) for name in CLASS_NAMES
                    ),
                    "recall_normal": float(metrics["per_class"]["Normal"]["recall"]),
                    "recall_viral": float(metrics["per_class"]["Viral"]["recall"]),
                    "recall_bacterial": float(metrics["per_class"]["Bacterial"]["recall"]),
                    "validation_log_loss": float(metrics["log_loss"]),
                    "validation_ece": float(metrics["expected_calibration_error"]),
                    "model_file_size_bytes": int(metrics["model_file_size_bytes"]),
                    "latency_ms_per_image": float(metrics["inference"]["latency_ms_per_image"]),
                    "best_epoch": int(metrics["best_epoch"]),
                    "training_duration_seconds": float(metrics["training_duration_seconds"]),
                    "bootstrap_interpretation": interpretation,
                }
            )

    selected = {}
    within_rankings = {}
    for architecture in ARCHITECTURES:
        ordered = sorted(
            [record for record in records if record["architecture"] == architecture],
            key=ranking_key,
        )
        within_rankings[architecture] = [record["run_id"] for record in ordered]
        selected[architecture] = ordered[0]["run_id"]

    columns = list(records[0])
    csv_path = results / "preprocessing_ablation_analysis.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_data": "locked validation only; test not accessed",
        "primary_metric": "validation macro F1",
        "tie_breakers": [
            "balanced accuracy",
            "minimum per-class recall",
            "log loss",
            "model file size",
            "compute-only latency",
            "run ID",
        ],
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "method": "paired class-stratified nonparametric bootstrap",
            "interpretation": "descriptive uncertainty on the selection set, not an independent confirmatory test",
        },
        "within_model_rankings": within_rankings,
        "selected_completed_run_by_representative": selected,
        "records": records,
    }
    (results / "preprocessing_ablation_analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    display = {"baseline": "No enhancement", "clahe": "CLAHE", "histogram_equalization": "Global HE", "pseudo_color": "VIRIDIS"}
    table_rows = "\n".join(
        f"| {record['architecture']} | {display[record['preprocessing']]} | {record['validation_macro_f1']:.6f} | {record['delta_macro_f1_vs_same_model_baseline']:+.6f} | [{record['delta_bootstrap_ci_lower']:+.6f}, {record['delta_bootstrap_ci_upper']:+.6f}] | {record['recall_viral']:.3f} | {record['best_epoch']} | {record['bootstrap_interpretation']} |"
        for record in records
    )
    winners = "\n".join(f"- **{architecture}:** `{run_id}`" for architecture, run_id in selected.items())
    md = f"""# Preprocessing Ablation Results

Generated: {payload['generated_at']}

This report is generated only when all twelve validation artifacts exist: three completed no-enhancement controls and nine new enhanced runs. The locked test set is not loaded.

## Results

| Architecture | Condition | Validation macro F1 | Delta vs same-model control | Paired 95% bootstrap interval | Viral recall | Best epoch | Descriptive interpretation |
|---|---|---:|---:|---|---:|---:|---|
{table_rows}

## Representative winners

{winners}

The primary selection metric is validation macro F1. Ties are resolved by balanced accuracy, minimum per-class recall, log loss, model size, compute-only latency, then run ID. The paired class-stratified bootstrap uses {BOOTSTRAP_REPLICATES:,} resamples and describes uncertainty on this same validation set; it is not an independent test and does not unlock test data.

VIRIDIS remains labeled exploratory even if it scores well. A score increase would show only that this fixed colormap helped the present validation pipeline; it would not prove that artificial color adds clinical information.
"""
    (repo / "model" / "docs" / "09_preprocessing_ablation_results_continuation_v2.md").write_text(
        md, encoding="utf-8"
    )
    print(json.dumps({"selected": selected, "csv": str(csv_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
