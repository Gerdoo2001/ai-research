"""Fit temperature on validation predictions and report post-freeze test calibration."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import _multiclass_bootstrap
import numpy as np
from multiclass.metrics import calculate_metrics


CLASS_NAMES = ("Normal", "Viral", "Bacterial")


def read_predictions(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.asarray([int(row["true_index"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [
            [float(row["probability_normal"]), float(row["probability_viral"]), float(row["probability_bacterial"])]
            for row in rows
        ],
        dtype=np.float64,
    )
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return labels, probabilities


def with_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def bin_table(labels, probabilities, bins=15):
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:]), 1):
        mask = (confidence > lower) & (confidence <= upper)
        rows.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "count": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()) if mask.any() else "",
                "accuracy": float((predictions[mask] == labels[mask]).mean()) if mask.any() else "",
            }
        )
    return rows


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize_scalar

    repo = _multiclass_bootstrap.REPO_ROOT
    results = repo / "model/results"
    selection = json.loads((results / "final_selection.json").read_text(encoding="utf-8"))
    if selection.get("status") != "locked_test_evaluated":
        raise RuntimeError("temperature scaling is post-freeze and requires the completed locked-test evaluation")
    validation_dir = Path(selection["selected_run_dir"])
    test_dir = results / "runs/final_locked_test"
    val_labels, val_probabilities = read_predictions(validation_dir / "predictions.csv")
    test_labels, test_probabilities = read_predictions(test_dir / "predictions.csv")

    def objective(temperature):
        scaled = with_temperature(val_probabilities, float(temperature))
        return -float(np.log(scaled[np.arange(len(val_labels)), val_labels]).mean())

    optimized = minimize_scalar(objective, bounds=(0.05, 10.0), method="bounded", options={"xatol": 1e-7})
    if not optimized.success:
        raise RuntimeError(f"temperature optimization failed: {optimized.message}")
    temperature = float(optimized.x)
    val_before = calculate_metrics(val_labels, val_probabilities, CLASS_NAMES)
    val_after = calculate_metrics(val_labels, with_temperature(val_probabilities, temperature), CLASS_NAMES)
    test_before = calculate_metrics(test_labels, test_probabilities, CLASS_NAMES)
    calibrated_test = with_temperature(test_probabilities, temperature)
    test_after = calculate_metrics(test_labels, calibrated_test, CLASS_NAMES)
    output_dir = results / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    before_bins = bin_table(test_labels, test_probabilities)
    after_bins = bin_table(test_labels, calibrated_test)
    rows = []
    for before, after in zip(before_bins, after_bins):
        rows.append(
            {
                "bin": before["bin"],
                "lower": before["lower"],
                "upper": before["upper"],
                "count_before": before["count"],
                "mean_confidence_before": before["mean_confidence"],
                "accuracy_before": before["accuracy"],
                "count_after": after["count"],
                "mean_confidence_after": after["mean_confidence"],
                "accuracy_after": after["accuracy"],
            }
        )
    with (output_dir / "test_calibration_bins.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharex=True, sharey=True)
    for axis, table, title in ((axes[0], before_bins, "Before temperature scaling"), (axes[1], after_bins, "After temperature scaling")):
        points = [(row["mean_confidence"], row["accuracy"]) for row in table if row["count"]]
        axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        axis.plot([point[0] for point in points], [point[1] for point in points], color="black", marker="o", linewidth=1.5)
        axis.set_title(title, color="black")
        axis.set_xlabel("Mean confidence")
        axis.grid(True, color="0.85", linewidth=0.6)
    axes[0].set_ylabel("Observed accuracy")
    figure.tight_layout()
    figure.savefig(output_dir / "test_calibration_temperature_scaling.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    def compact(metrics):
        return {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "log_loss": metrics["log_loss"],
            "expected_calibration_error": metrics["expected_calibration_error"],
            "brier_score": metrics["brier_score"],
        }

    payload = {
        "schema_version": 1,
        "selection_status": selection["status"],
        "temperature_fit_split": "validation",
        "temperature_evaluation_split": "single frozen locked test",
        "temperature": temperature,
        "optimizer_success": bool(optimized.success),
        "validation_before": compact(val_before),
        "validation_after": compact(val_after),
        "test_before": compact(test_before),
        "test_after": compact(test_after),
        "interpretation_boundary": "Temperature was fit on validation only after model configuration freeze; locked-test labels did not influence temperature or retraining.",
    }
    (output_dir / "final_temperature_scaling.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
