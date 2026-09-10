"""Probability-based multiclass metrics and calibration summaries."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return ece


def calculate_metrics(y_true, probabilities, class_names: tuple[str, ...] | list[str]) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        log_loss,
        matthews_corrcoef,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize

    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(class_names):
        raise ValueError("probability matrix does not match class_names")
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    labels = np.arange(len(class_names))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=labels, zero_division=0
    )
    macro = precision_recall_fscore_support(y_true, predictions, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, predictions, average="weighted", zero_division=0)
    matrix = confusion_matrix(y_true, predictions, labels=labels)
    specificity = []
    for index in labels:
        tp = matrix[index, index]
        fn = matrix[index, :].sum() - tp
        fp = matrix[:, index].sum() - tp
        tn = matrix.sum() - tp - fn - fp
        specificity.append(float(tn / (tn + fp)) if tn + fp else math.nan)
    one_hot = label_binarize(y_true, classes=labels)
    if len(class_names) == 2 and one_hot.shape[1] == 1:
        one_hot = np.column_stack([1 - one_hot[:, 0], one_hot[:, 0]])
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
        "matthews_correlation_coefficient": float(matthews_corrcoef(y_true, predictions)),
        "cohen_kappa": float(cohen_kappa_score(y_true, predictions)),
        "log_loss": float(log_loss(y_true, probabilities, labels=labels)),
        "brier_score": brier,
        "expected_calibration_error": expected_calibration_error(y_true, probabilities),
        "confusion_matrix": matrix.tolist(),
        "per_class": {},
    }
    try:
        result["roc_auc_ovr_macro"] = float(roc_auc_score(one_hot, probabilities, average="macro"))
    except ValueError:
        result["roc_auc_ovr_macro"] = None
    try:
        result["pr_auc_macro"] = float(average_precision_score(one_hot, probabilities, average="macro"))
    except ValueError:
        result["pr_auc_macro"] = None
    for index, name in enumerate(class_names):
        try:
            roc_auc = float(roc_auc_score(one_hot[:, index], probabilities[:, index]))
        except ValueError:
            roc_auc = None
        try:
            pr_auc = float(average_precision_score(one_hot[:, index], probabilities[:, index]))
        except ValueError:
            pr_auc = None
        result["per_class"][name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "specificity": specificity[index],
            "f1": float(f1[index]),
            "support": int(support[index]),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }
    return result
