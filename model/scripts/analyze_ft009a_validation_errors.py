"""FT-009A matched validation-error analysis for the selected full-image systems."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "model" / "results" / "ft009a_validation_error_analysis"
DATASET_ROOT = REPO_ROOT / "model" / "datasets" / "chest_xray"
CLASS_NAMES = ("Normal", "Viral", "Bacterial")
PROBABILITY_COLUMNS = ("probability_normal", "probability_viral", "probability_bacterial")

# Project-defined, pre-run evidence gate. These numbers are not copied from a paper.
ETIOLOGY_ERROR_DOMINANCE_THRESHOLD = 0.70
BINARY_BALANCED_ACCURACY_GAP_THRESHOLD = 0.10

SOURCES = {
    "ft_ref": {
        "label": "FT-REF DenseNet-201",
        "predictions": REPO_ROOT
        / "model/results/runs/staged_densenet201_pseudo_color_then_histogram_equalization_then_clahe_s42/predictions.csv",
        "metrics": REPO_ROOT
        / "model/results/runs/staged_densenet201_pseudo_color_then_histogram_equalization_then_clahe_s42/metrics.json",
        "expected_sha256": "10c2cd4466049389ee51f495e8c0c7176616ae7ac03a38b4174edc3307491007",
    },
    "ft007_label_smoothing": {
        "label": "FT-007 label smoothing 0.05",
        "predictions": REPO_ROOT
        / "model/results/runs/fine_tuning_densenet201_viridis_ge_clahe_label_smoothing005_s42/predictions.csv",
        "metrics": REPO_ROOT
        / "model/results/runs/fine_tuning_densenet201_viridis_ge_clahe_label_smoothing005_s42/metrics.json",
        "expected_sha256": "b20b0e0097ef3aac0d2626a39ee6fe1dfda8a525d6c298d5a026fab7508cb150",
    },
    "ft004_ensemble_s42": {
        "label": "FT-004 frozen ensemble seed 42",
        "predictions": REPO_ROOT / "model/results/ensemble_confirmation/ensemble_predictions_s42.csv",
        "metrics": REPO_ROOT / "model/results/ensemble_confirmation/ensemble_metrics_s42.json",
        "expected_sha256": "ba1dfb653c0093396677576e15c9e77d4dc22945f37123a0c765a8bd1d1351de",
    },
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
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_predictions(path: Path) -> pd.DataFrame:
    required = {
        "sample_id",
        "relative_path",
        "true_index",
        "true_class",
        "predicted_index",
        "predicted_class",
        *PROBABILITY_COLUMNS,
    }
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate sample IDs")
    for column in ("true_index", "predicted_index"):
        frame[column] = frame[column].astype(int)
    probabilities = frame[list(PROBABILITY_COLUMNS)].astype(float).to_numpy()
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1.001).any():
        raise ValueError(f"{path} contains invalid probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=0.02):
        raise ValueError(f"{path} probabilities do not sum to one")
    if not np.array_equal(probabilities.argmax(axis=1), frame["predicted_index"].to_numpy()):
        raise ValueError(f"{path} predicted indices disagree with argmax probabilities")
    expected_true = frame["true_index"].map(dict(enumerate(CLASS_NAMES)))
    expected_pred = frame["predicted_index"].map(dict(enumerate(CLASS_NAMES)))
    if not expected_true.equals(frame["true_class"]) or not expected_pred.equals(frame["predicted_class"]):
        raise ValueError(f"{path} class names disagree with class indices")
    if "confidence" not in frame.columns:
        frame["confidence"] = probabilities.max(axis=1)
    frame["confidence"] = frame["confidence"].astype(float)
    return frame.sort_values("sample_id").reset_index(drop=True)


def align_predictions(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("no prediction frames supplied")
    first_key = next(iter(frames))
    identity_base = frames[first_key][["sample_id", "relative_path", "true_index", "true_class"]].copy()
    aligned = identity_base.copy()
    for key, frame in frames.items():
        identity = frame[["sample_id", "relative_path", "true_index", "true_class"]].reset_index(drop=True)
        if not identity_base.equals(identity):
            raise ValueError(f"{key} does not use the identical ordered validation cohort")
        for column in ("predicted_index", "predicted_class", "confidence", *PROBABILITY_COLUMNS):
            aligned[f"{key}_{column}"] = frame[column].to_numpy()
        aligned[f"{key}_correct"] = frame["predicted_index"].to_numpy() == frame["true_index"].to_numpy()
    return aligned


def error_type(true_index: int, predicted_index: int) -> str:
    if true_index == predicted_index:
        return "correct"
    if {int(true_index), int(predicted_index)} == {1, 2}:
        return "viral_bacterial_etiology"
    return "normal_pneumonia_boundary"


def multiclass_metrics(frame: pd.DataFrame, key: str) -> dict:
    truth = frame["true_index"].to_numpy()
    predicted = frame[f"{key}_predicted_index"].to_numpy()
    matrix = confusion_matrix(truth, predicted, labels=(0, 1, 2))
    recalls = recall_score(truth, predicted, labels=(0, 1, 2), average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "per_class_recall": {CLASS_NAMES[index]: float(value) for index, value in enumerate(recalls)},
        "confusion_matrix": matrix.tolist(),
    }


def binary_task_metrics(frame: pd.DataFrame, key: str) -> dict:
    truth = frame["true_index"].to_numpy()
    normal_probability = frame[f"{key}_probability_normal"].to_numpy(float)
    pneumonia_probability = (
        frame[f"{key}_probability_viral"].to_numpy(float)
        + frame[f"{key}_probability_bacterial"].to_numpy(float)
    )
    normal_vs_pneumonia_truth = (truth != 0).astype(int)
    normal_vs_pneumonia_predicted = (pneumonia_probability > normal_probability).astype(int)

    pneumonia = frame[frame["true_index"].isin((1, 2))]
    etiology_truth = (pneumonia["true_index"].to_numpy() == 2).astype(int)
    etiology_predicted = (
        pneumonia[f"{key}_probability_bacterial"].to_numpy(float)
        > pneumonia[f"{key}_probability_viral"].to_numpy(float)
    ).astype(int)
    return {
        "normal_vs_pneumonia": {
            "n": int(len(frame)),
            "accuracy": float(accuracy_score(normal_vs_pneumonia_truth, normal_vs_pneumonia_predicted)),
            "balanced_accuracy": float(
                balanced_accuracy_score(normal_vs_pneumonia_truth, normal_vs_pneumonia_predicted)
            ),
            "normal_recall": float(
                recall_score(normal_vs_pneumonia_truth, normal_vs_pneumonia_predicted, pos_label=0)
            ),
            "pneumonia_recall": float(
                recall_score(normal_vs_pneumonia_truth, normal_vs_pneumonia_predicted, pos_label=1)
            ),
        },
        "viral_vs_bacterial_within_true_pneumonia": {
            "n": int(len(pneumonia)),
            "accuracy": float(accuracy_score(etiology_truth, etiology_predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(etiology_truth, etiology_predicted)),
            "viral_recall": float(recall_score(etiology_truth, etiology_predicted, pos_label=0)),
            "bacterial_recall": float(recall_score(etiology_truth, etiology_predicted, pos_label=1)),
        },
    }


def transition_summary(frame: pd.DataFrame, candidate: str) -> dict:
    reference_correct = frame["ft_ref_correct"].to_numpy(bool)
    candidate_correct = frame[f"{candidate}_correct"].to_numpy(bool)
    reference_prediction = frame["ft_ref_predicted_index"].to_numpy(int)
    candidate_prediction = frame[f"{candidate}_predicted_index"].to_numpy(int)
    return {
        "both_correct": int((reference_correct & candidate_correct).sum()),
        "candidate_rescues_reference_error": int((~reference_correct & candidate_correct).sum()),
        "candidate_introduces_new_error": int((reference_correct & ~candidate_correct).sum()),
        "both_wrong_same_prediction": int(
            ((~reference_correct) & (~candidate_correct) & (reference_prediction == candidate_prediction)).sum()
        ),
        "both_wrong_different_prediction": int(
            ((~reference_correct) & (~candidate_correct) & (reference_prediction != candidate_prediction)).sum()
        ),
    }


def correctness_patterns(frame: pd.DataFrame) -> dict:
    result = {}
    for _, row in frame.iterrows():
        pattern = "".join(
            "C" if bool(row[f"{key}_correct"]) else "W"
            for key in ("ft_ref", "ft007_label_smoothing", "ft004_ensemble_s42")
        )
        result[pattern] = result.get(pattern, 0) + 1
    return dict(sorted(result.items()))


def error_decomposition(frame: pd.DataFrame, key: str) -> dict:
    values = [error_type(t, p) for t, p in zip(frame["true_index"], frame[f"{key}_predicted_index"])]
    counts = {name: values.count(name) for name in ("correct", "normal_pneumonia_boundary", "viral_bacterial_etiology")}
    total_errors = len(frame) - counts["correct"]
    counts["total_errors"] = total_errors
    counts["etiology_fraction_of_errors"] = (
        counts["viral_bacterial_etiology"] / total_errors if total_errors else 0.0
    )
    return counts


def save_confusion_plot(metrics: dict[str, dict], target: Path) -> None:
    figure, axes = plt.subplots(1, len(metrics), figsize=(13, 4.2), constrained_layout=True)
    for axis, (key, values) in zip(axes, metrics.items()):
        matrix = np.asarray(values["confusion_matrix"])
        image = axis.imshow(matrix, cmap="Greys", vmin=0, vmax=max(1, matrix.max()))
        for row in range(3):
            for column in range(3):
                color = "white" if matrix[row, column] > matrix.max() * 0.55 else "black"
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color)
        axis.set_title(SOURCES[key]["label"], fontsize=10)
        axis.set_xticks(range(3), CLASS_NAMES, rotation=25, ha="right")
        axis.set_yticks(range(3), CLASS_NAMES)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("FT-009A matched validation confusion matrices", fontsize=13)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_error_decomposition_plot(decomposition: dict[str, dict], target: Path) -> None:
    keys = list(decomposition)
    correct = [decomposition[key]["correct"] for key in keys]
    boundary = [decomposition[key]["normal_pneumonia_boundary"] for key in keys]
    etiology = [decomposition[key]["viral_bacterial_etiology"] for key in keys]
    x = np.arange(len(keys))
    figure, axis = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    axis.bar(x, correct, label="Correct", color="#d9d9d9", edgecolor="black")
    axis.bar(x, boundary, bottom=correct, label="Normal/Pneumonia error", color="#8c8c8c", edgecolor="black")
    axis.bar(
        x,
        etiology,
        bottom=np.asarray(correct) + np.asarray(boundary),
        label="Viral/Bacterial error",
        color="#262626",
        edgecolor="black",
    )
    axis.set_xticks(x, [SOURCES[key]["label"] for key in keys], rotation=12, ha="right")
    axis.set_ylabel("Validation images")
    axis.set_title("Where the three-class systems succeed and fail")
    axis.legend(frameon=False, loc="lower left")
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)


def wrapped_text(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int, font, fill="black") -> int:
    approximate_characters = max(18, int(width / max(1, draw.textlength("M", font=font))))
    lines = textwrap.wrap(
        text,
        width=approximate_characters,
        break_long_words=True,
        break_on_hyphens=False,
    )
    for line in lines[:4]:
        draw.text((x, y), line, font=font, fill=fill)
        y += 16
    return y


def save_contact_sheet(frame: pd.DataFrame, target: Path, title: str, limit: int = 12) -> int:
    selected = frame.head(limit)
    if selected.empty:
        return 0
    columns = 3
    tile_width, tile_height = 420, 430
    rows = int(math.ceil(len(selected) / columns))
    sheet = Image.new("RGB", (columns * tile_width, 46 + rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((12, 14), title, font=font, fill="black")
    for position, (_, row) in enumerate(selected.iterrows()):
        x = (position % columns) * tile_width
        y = 46 + (position // columns) * tile_height
        source = DATASET_ROOT / row["relative_path"]
        with Image.open(source) as opened:
            image = opened.convert("L").convert("RGB")
        image.thumbnail((390, 285), Image.Resampling.LANCZOS)
        paste_x = x + (tile_width - image.width) // 2
        paste_y = y + 6 + (285 - image.height) // 2
        sheet.paste(image, (paste_x, paste_y))
        caption_y = y + 300
        captions = [
            f"{row['sample_id']} | True: {row['true_class']}",
            f"REF: {row['ft_ref_predicted_class']} ({row['ft_ref_confidence']:.3f})",
            f"LS: {row['ft007_label_smoothing_predicted_class']} ({row['ft007_label_smoothing_confidence']:.3f})",
            f"ENS: {row['ft004_ensemble_s42_predicted_class']} ({row['ft004_ensemble_s42_confidence']:.3f})",
            str(row["relative_path"]),
        ]
        for caption in captions:
            caption_y = wrapped_text(draw, caption, x + 12, caption_y, tile_width - 24, font)
        draw.rectangle((x, y, x + tile_width - 1, y + tile_height - 1), outline="black", width=1)
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, quality=94)
    return len(selected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    source_records = {}
    frames = {}
    for key, source in SOURCES.items():
        actual_hash = sha256(source["predictions"])
        if actual_hash != source["expected_sha256"]:
            raise RuntimeError(f"{key} prediction hash changed: {actual_hash}")
        frame = load_predictions(source["predictions"])
        if len(frame) != 874:
            raise RuntimeError(f"{key} has {len(frame)} rows instead of 874")
        frames[key] = frame
        source_records[key] = {
            "label": source["label"],
            "predictions_path": str(source["predictions"]),
            "predictions_sha256": actual_hash,
            "metrics_path": str(source["metrics"]),
            "metrics_sha256": sha256(source["metrics"]),
            "rows": len(frame),
        }

    aligned = align_predictions(frames)
    missing_images = [str(DATASET_ROOT / path) for path in aligned["relative_path"] if not (DATASET_ROOT / path).exists()]
    if missing_images:
        raise FileNotFoundError(f"{len(missing_images)} validation source images are missing")

    for key in SOURCES:
        aligned[f"{key}_error_type"] = [
            error_type(t, p)
            for t, p in zip(aligned["true_index"], aligned[f"{key}_predicted_index"])
        ]

    metrics = {key: multiclass_metrics(aligned, key) for key in SOURCES}
    binary = {key: binary_task_metrics(aligned, key) for key in SOURCES}
    decomposition = {key: error_decomposition(aligned, key) for key in SOURCES}
    transitions = {
        key: transition_summary(aligned, key)
        for key in ("ft007_label_smoothing", "ft004_ensemble_s42")
    }
    patterns = correctness_patterns(aligned)

    for key in SOURCES:
        aligned[f"{key}_error_confidence"] = np.where(
            aligned[f"{key}_correct"], np.nan, aligned[f"{key}_confidence"]
        )
    aligned.to_csv(output / "per_sample_comparison.csv", index=False)

    metric_rows = []
    for key in SOURCES:
        metric_rows.append(
            {
                "system": key,
                "label": SOURCES[key]["label"],
                "macro_f1": metrics[key]["macro_f1"],
                "balanced_accuracy": metrics[key]["balanced_accuracy"],
                "accuracy": metrics[key]["accuracy"],
                "normal_recall": metrics[key]["per_class_recall"]["Normal"],
                "viral_recall": metrics[key]["per_class_recall"]["Viral"],
                "bacterial_recall": metrics[key]["per_class_recall"]["Bacterial"],
                "normal_vs_pneumonia_balanced_accuracy": binary[key]["normal_vs_pneumonia"]["balanced_accuracy"],
                "viral_vs_bacterial_balanced_accuracy": binary[key]["viral_vs_bacterial_within_true_pneumonia"]["balanced_accuracy"],
                "etiology_fraction_of_all_errors": decomposition[key]["etiology_fraction_of_errors"],
            }
        )
    pd.DataFrame(metric_rows).to_csv(output / "metrics_and_error_decomposition.csv", index=False)

    high_confidence = []
    for key in SOURCES:
        subset = aligned[~aligned[f"{key}_correct"]].copy()
        subset.insert(0, "system", key)
        subset = subset.sort_values(f"{key}_confidence", ascending=False).head(30)
        high_confidence.append(subset)
    pd.concat(high_confidence, ignore_index=True).to_csv(output / "high_confidence_errors.csv", index=False)

    common_errors = aligned[
        (~aligned["ft_ref_correct"])
        & (~aligned["ft007_label_smoothing_correct"])
        & (~aligned["ft004_ensemble_s42_correct"])
    ].copy()
    common_errors["mean_error_confidence"] = common_errors[
        ["ft_ref_confidence", "ft007_label_smoothing_confidence", "ft004_ensemble_s42_confidence"]
    ].mean(axis=1)
    common_errors = common_errors.sort_values("mean_error_confidence", ascending=False)
    common_errors.to_csv(output / "common_errors_all_three.csv", index=False)

    ensemble_rescues = aligned[(~aligned["ft_ref_correct"]) & aligned["ft004_ensemble_s42_correct"]].copy()
    ensemble_rescues = ensemble_rescues.sort_values("ft004_ensemble_s42_confidence", ascending=False)
    ensemble_rescues.to_csv(output / "ensemble_rescues_reference_errors.csv", index=False)
    label_smoothing_rescues = aligned[(~aligned["ft_ref_correct"]) & aligned["ft007_label_smoothing_correct"]].copy()
    label_smoothing_rescues = label_smoothing_rescues.sort_values("ft007_label_smoothing_confidence", ascending=False)
    label_smoothing_rescues.to_csv(output / "label_smoothing_rescues_reference_errors.csv", index=False)

    save_confusion_plot(metrics, output / "confusion_matrices.png")
    save_error_decomposition_plot(decomposition, output / "error_decomposition.png")
    contact_sheet_counts = {
        "common_errors": save_contact_sheet(
            common_errors,
            output / "representative_common_errors.jpg",
            "FT-009A: highest-confidence errors shared by all three systems",
        ),
        "ensemble_rescues": save_contact_sheet(
            ensemble_rescues,
            output / "representative_ensemble_rescues.jpg",
            "FT-009A: reference errors corrected by the frozen seed-42 ensemble",
        ),
        "label_smoothing_rescues": save_contact_sheet(
            label_smoothing_rescues,
            output / "representative_label_smoothing_rescues.jpg",
            "FT-009A: reference errors corrected by label smoothing 0.05",
        ),
    }

    systems_passing_dominance = [
        key
        for key in SOURCES
        if decomposition[key]["etiology_fraction_of_errors"] >= ETIOLOGY_ERROR_DOMINANCE_THRESHOLD
    ]
    systems_passing_gap = [
        key
        for key in SOURCES
        if binary[key]["normal_vs_pneumonia"]["balanced_accuracy"]
        - binary[key]["viral_vs_bacterial_within_true_pneumonia"]["balanced_accuracy"]
        >= BINARY_BALANCED_ACCURACY_GAP_THRESHOLD
    ]
    advance = len(systems_passing_dominance) >= 2 and len(systems_passing_gap) >= 2

    summary = {
        "schema_version": 1,
        "experiment_id": "FT-009A-VALIDATION-ERROR-ANALYSIS",
        "status": "completed",
        "completed_at": now(),
        "task": "matched per-image validation error analysis; no model training",
        "source_split": "locked validation assignment only",
        "validation_images": len(aligned),
        "locked_test_accessed": False,
        "model_training_started": False,
        "source_original_folder_note": (
            "relative paths may begin with train/ or test/ from the downloaded source dataset; the locked project "
            "manifest assignment, not that legacy folder name, defines this validation-only analysis"
        ),
        "sources": source_records,
        "metrics": metrics,
        "binary_task_analysis": binary,
        "error_decomposition": decomposition,
        "reference_transitions": transitions,
        "correctness_patterns_order": (
            "characters correspond to FT-REF, FT-007 label smoothing, and FT-004 ensemble seed 42; C=correct, W=wrong"
        ),
        "correctness_patterns": patterns,
        "common_error_count_all_three": len(common_errors),
        "contact_sheet_counts": contact_sheet_counts,
        "ft009b_evidence_gate": {
            "status": "passed_design_gate" if advance else "failed_design_gate",
            "training_authorized": False,
            "etiology_error_dominance_threshold": ETIOLOGY_ERROR_DOMINANCE_THRESHOLD,
            "binary_balanced_accuracy_gap_threshold": BINARY_BALANCED_ACCURACY_GAP_THRESHOLD,
            "required_systems": 2,
            "systems_passing_error_dominance": systems_passing_dominance,
            "systems_passing_binary_gap": systems_passing_gap,
            "interpretation": (
                "A passed design gate authorizes drafting and contract-testing a matched seed-42 hierarchical "
                "experiment. It does not authorize locked-test access or claim that hierarchy will improve results."
            ),
            "threshold_provenance": "project-defined before FT-009A results were computed; not copied from a publication",
        },
        "reference_basis": [
            {
                "url": "https://doi.org/10.1016/j.compeleceng.2022.108402",
                "applicability": "Direct/Partial",
                "supports": "separate pediatric Normal-vs-Pneumonia and Bacterial-vs-Viral CXR scenarios",
            },
            {
                "url": "https://doi.org/10.1002/ppul.25229",
                "applicability": "Direct/Partial",
                "supports": "pediatric CXR modeling specifically for Viral-vs-Bacterial differentiation",
            },
            {
                "url": "https://pubmed.ncbi.nlm.nih.gov/34648114/",
                "applicability": "Background",
                "supports": "strict interpretation because pediatric pneumonia classifiers may generalize poorly externally",
            },
        ],
        "artifacts": {
            "per_sample_comparison": str(output / "per_sample_comparison.csv"),
            "metrics_and_error_decomposition": str(output / "metrics_and_error_decomposition.csv"),
            "high_confidence_errors": str(output / "high_confidence_errors.csv"),
            "common_errors_all_three": str(output / "common_errors_all_three.csv"),
            "ensemble_rescues_reference_errors": str(output / "ensemble_rescues_reference_errors.csv"),
            "label_smoothing_rescues_reference_errors": str(output / "label_smoothing_rescues_reference_errors.csv"),
            "confusion_matrices": str(output / "confusion_matrices.png"),
            "error_decomposition_plot": str(output / "error_decomposition.png"),
            "representative_common_errors": str(output / "representative_common_errors.jpg"),
            "representative_ensemble_rescues": str(output / "representative_ensemble_rescues.jpg"),
            "representative_label_smoothing_rescues": str(output / "representative_label_smoothing_rescues.jpg"),
        },
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"sources", "metrics", "binary_task_analysis", "reference_transitions", "artifacts"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
