"""Validation/test evaluation and per-run artifact generation."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import calculate_metrics
from .tracking import atomic_write_json


def collect_predictions(model, loader, device, *, amp: bool) -> tuple[float, np.ndarray, np.ndarray, list[str]]:
    import torch
    import torch.nn.functional as functional

    model.eval()
    losses: list[float] = []
    labels: list[int] = []
    probabilities: list[np.ndarray] = []
    paths: list[str] = []
    with torch.inference_mode():
        for inputs, targets, relative_paths in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits = model(inputs)
                loss = functional.cross_entropy(logits, targets)
            losses.append(float(loss.detach().cpu()) * inputs.shape[0])
            labels.extend(targets.detach().cpu().tolist())
            probabilities.extend(torch.softmax(logits, dim=1).detach().cpu().numpy())
            paths.extend(relative_paths)
    return (
        sum(losses) / max(1, len(labels)),
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
        paths,
    )


def evaluate_model(model, loader, device, class_names, *, amp: bool) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[str]]:
    loss, labels, probabilities, paths = collect_predictions(model, loader, device, amp=amp)
    metrics = calculate_metrics(labels, probabilities, class_names)
    metrics["loss"] = loss
    return metrics, labels, probabilities, paths


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_evaluation_artifacts(
    output_dir: Path,
    metrics: dict[str, Any],
    labels: np.ndarray,
    probabilities: np.ndarray,
    paths: list[str],
    class_names: tuple[str, ...] | list[str],
    *,
    split: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
    from sklearn.preprocessing import label_binarize

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "metrics.json", {"split": split, **metrics})
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    report_rows = []
    for name in class_names:
        report_rows.append({"class_name": name, **metrics["per_class"][name]})
    _write_csv(
        output_dir / "classification_report.csv",
        ["class_name", "precision", "recall", "specificity", "f1", "support", "roc_auc", "pr_auc"],
        report_rows,
    )

    predictions = probabilities.argmax(axis=1)
    prediction_rows = []
    error_rows = []
    for index, path in enumerate(paths):
        row = {
            "sample_id": f"{split}_{index:05d}",
            "relative_path": path,
            "true_index": int(labels[index]),
            "true_class": class_names[int(labels[index])],
            "predicted_index": int(predictions[index]),
            "predicted_class": class_names[int(predictions[index])],
            "confidence": float(probabilities[index].max()),
        }
        for class_index, name in enumerate(class_names):
            row[f"probability_{name.lower()}"] = float(probabilities[index, class_index])
        prediction_rows.append(row)
        if labels[index] != predictions[index]:
            error_rows.append(row)
    columns = list(prediction_rows[0]) if prediction_rows else []
    _write_csv(output_dir / "predictions.csv", columns, prediction_rows)
    _write_csv(output_dir / "sample_errors.csv", columns, sorted(error_rows, key=lambda row: -row["confidence"]))

    figure, axis = plt.subplots(figsize=(6.0, 5.2))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_names)), class_names, rotation=25, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(f"{split.title()} confusion matrix")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)

    one_hot = label_binarize(labels, classes=np.arange(len(class_names)))
    if len(class_names) == 2 and one_hot.shape[1] == 1:
        one_hot = np.column_stack([1 - one_hot[:, 0], one_hot[:, 0]])
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    for index, name in enumerate(class_names):
        if len(np.unique(one_hot[:, index])) > 1:
            RocCurveDisplay.from_predictions(one_hot[:, index], probabilities[:, index], name=name, ax=axis)
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set_title(f"{split.title()} one-vs-rest ROC curves")
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curves.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    for index, name in enumerate(class_names):
        if len(np.unique(one_hot[:, index])) > 1:
            PrecisionRecallDisplay.from_predictions(one_hot[:, index], probabilities[:, index], name=name, ax=axis)
    axis.set_title(f"{split.title()} one-vs-rest precision-recall curves")
    figure.tight_layout()
    figure.savefig(output_dir / "pr_curves.png", dpi=180)
    plt.close(figure)

    confidence = probabilities.max(axis=1)
    correctness = (predictions == labels).astype(int)
    fraction, mean = calibration_curve(correctness, confidence, n_bins=10, strategy="uniform")
    figure, axis = plt.subplots(figsize=(6.0, 5.0))
    axis.plot(mean, fraction, marker="o", label="Observed")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Accuracy")
    axis.set_title(f"{split.title()} confidence calibration")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "calibration_curve.png", dpi=180)
    plt.close(figure)


def measure_inference(model, device, input_size: int, *, batch_size: int = 1, warmup: int = 10, runs: int = 50) -> dict[str, Any]:
    import torch

    inputs = torch.zeros(batch_size, 3, input_size, input_size, device=device)
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    images = runs * batch_size
    return {
        "hardware": str(device),
        "batch_size": batch_size,
        "input_size": input_size,
        "warmup_runs": warmup,
        "measured_runs": runs,
        "preprocessing_included": False,
        "host_to_device_transfer_included": False,
        "latency_ms_per_image": elapsed * 1000.0 / images,
        "throughput_images_per_second": images / elapsed,
    }


def plot_training_history(history: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set_title("Cross-entropy loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], label="Macro F1")
    axes[1].plot(epochs, [row["val_balanced_accuracy"] for row in history], label="Balanced accuracy")
    axes[1].set_title("Validation selection metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
