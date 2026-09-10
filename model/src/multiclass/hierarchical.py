"""Fail-closed two-stage inference for FT-009B hierarchical classification."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .config import ExperimentConfig
from .data import read_manifest, validate_manifest_bundle
from .evaluation import save_evaluation_artifacts
from .metrics import calculate_metrics
from .models import assert_output_contract, build_model
from .preprocessing import build_transform
from .tracking import atomic_write_json, now_iso
from .trainer import resolve_device, set_deterministic_seed


THREE_CLASS_NAMES = ("Normal", "Viral", "Bacterial")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage_pair(stage1: ExperimentConfig, stage2: ExperimentConfig) -> None:
    """Require an isolated task change and otherwise matched training settings."""

    if stage1.task_mode != "binary" or stage1.resolved_class_names != ("Normal", "Pneumonia"):
        raise ValueError("FT-009B Stage 1 must be binary Normal/Pneumonia")
    if stage2.task_mode != "viral_bacterial" or stage2.resolved_class_names != ("Viral", "Bacterial"):
        raise ValueError("FT-009B Stage 2 must be binary Viral/Bacterial")
    if stage1.seed != stage2.seed:
        raise ValueError("FT-009B stages must use the same seed")

    ignored = {"task_mode", "class_names", "run_id"}
    first = {key: value for key, value in stage1.as_dict().items() if key not in ignored}
    second = {key: value for key, value in stage2.as_dict().items() if key not in ignored}
    if first != second:
        differing = sorted(key for key in first if first.get(key) != second.get(key))
        raise ValueError(f"FT-009B stages differ outside their target definitions: {differing}")
    if stage1.roi.mode != "none" or stage2.roi.mode != "none":
        raise ValueError("FT-009B is a full-image classification experiment; ROI must remain disabled")


def validate_probability_matrix(probabilities: np.ndarray, name: str, rows: int | None = None) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    if rows is not None and matrix.shape[0] != rows:
        raise ValueError(f"{name} row count does not match the validation cohort")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    if (matrix < -1e-8).any() or (matrix > 1.0 + 1e-8).any():
        raise ValueError(f"{name} contains values outside [0, 1]")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(f"{name} rows must sum to one")
    return matrix


def compose_hierarchical_probabilities(
    stage1_probabilities: np.ndarray,
    stage2_probabilities: np.ndarray,
) -> np.ndarray:
    """Compose P(N), P(P)*P(V|P), P(P)*P(B|P) for every image."""

    stage1 = validate_probability_matrix(stage1_probabilities, "stage1_probabilities")
    stage2 = validate_probability_matrix(stage2_probabilities, "stage2_probabilities", rows=len(stage1))
    combined = np.column_stack(
        (
            stage1[:, 0],
            stage1[:, 1] * stage2[:, 0],
            stage1[:, 1] * stage2[:, 1],
        )
    )
    if not np.allclose(combined.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise RuntimeError("hierarchical probability composition violated the unit-sum contract")
    return combined


def assert_exact_path_alignment(expected: Sequence[str], actual: Sequence[str], name: str) -> None:
    if list(expected) != list(actual):
        for index, (left, right) in enumerate(zip(expected, actual)):
            if left != right:
                raise RuntimeError(f"{name} path mismatch at row {index}: {left!r} != {right!r}")
        raise RuntimeError(f"{name} path count mismatch: {len(expected)} != {len(actual)}")


class _AllRowsInferenceDataset:
    def __init__(self, rows: list[dict[str, str]], transform) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(Path(row["filepath"])) as image:
            prepared = image.copy()
        return self.transform(prepared), row["relative_path"]


def _load_checkpoint_model(config: ExperimentConfig, run_dir: Path, device):
    import torch

    checkpoint_path = run_dir / "best_checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"best checkpoint is absent: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("config_hash") != config.config_hash:
        raise RuntimeError(f"checkpoint/config hash mismatch: {checkpoint_path}")
    model, metadata = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    assert_output_contract(model, config.input_size, config.num_classes, device)
    return model, metadata, checkpoint_path


def collect_all_row_probabilities(
    config: ExperimentConfig,
    run_dir: Path,
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Infer one binary stage on every supplied row, including out-of-task Normal rows for Stage 2."""

    import torch
    from torch.utils.data import DataLoader

    if config.roi.mode != "none":
        raise ValueError("all-row hierarchical inference currently permits only the frozen full-image policy")
    set_deterministic_seed(config.seed)
    device = resolve_device(config)
    model, metadata, checkpoint_path = _load_checkpoint_model(config, run_dir, device)
    loader = DataLoader(
        _AllRowsInferenceDataset(rows, build_transform(config, training=False)),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    probabilities: list[np.ndarray] = []
    paths: list[str] = []
    model.eval()
    with torch.inference_mode():
        for inputs, relative_paths in loader:
            inputs = inputs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=config.amp and device.type == "cuda"):
                logits = model(inputs)
            # AMP may leave logits in float16. Compose probabilities only after
            # float32 softmax so the frozen unit-sum tolerance remains meaningful.
            probabilities.append(torch.softmax(logits.float(), dim=1).detach().cpu().numpy())
            paths.extend(relative_paths)
    matrix = validate_probability_matrix(np.vstack(probabilities), config.task_mode, rows=len(rows))
    return matrix, paths, {
        "run_id": config.effective_run_id(),
        "config_hash": config.config_hash,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parameter_count": metadata.parameter_count,
        "trainable_parameter_count": metadata.trainable_parameter_count,
    }


def _balanced_rows(rows: list[dict[str, str]], limit: int | None) -> list[dict[str, str]]:
    if limit is None or limit >= len(rows):
        return rows
    grouped: dict[int, list[dict[str, str]]] = {0: [], 1: [], 2: []}
    for row in rows:
        grouped[int(row["class_index"])].append(row)
    selected: list[dict[str, str]] = []
    cursor = 0
    while len(selected) < limit:
        added = False
        for label in (0, 1, 2):
            if cursor < len(grouped[label]) and len(selected) < limit:
                selected.append(grouped[label][cursor])
                added = True
        if not added:
            break
        cursor += 1
    return selected


def _write_component_probabilities(
    path: Path,
    rows: list[dict[str, str]],
    stage1: np.ndarray,
    stage2: np.ndarray,
    combined: np.ndarray,
) -> None:
    columns = [
        "relative_path", "true_index", "true_class",
        "stage1_probability_normal", "stage1_probability_pneumonia",
        "stage2_probability_viral_given_pneumonia", "stage2_probability_bacterial_given_pneumonia",
        "combined_probability_normal", "combined_probability_viral", "combined_probability_bacterial",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, row in enumerate(rows):
            true_index = int(row["class_index"])
            writer.writerow(
                {
                    "relative_path": row["relative_path"],
                    "true_index": true_index,
                    "true_class": THREE_CLASS_NAMES[true_index],
                    "stage1_probability_normal": float(stage1[index, 0]),
                    "stage1_probability_pneumonia": float(stage1[index, 1]),
                    "stage2_probability_viral_given_pneumonia": float(stage2[index, 0]),
                    "stage2_probability_bacterial_given_pneumonia": float(stage2[index, 1]),
                    "combined_probability_normal": float(combined[index, 0]),
                    "combined_probability_viral": float(combined[index, 1]),
                    "combined_probability_bacterial": float(combined[index, 2]),
                }
            )


def evaluate_hierarchical_validation(
    stage1_config: ExperimentConfig,
    stage2_config: ExperimentConfig,
    stage1_run_dir: Path,
    stage2_run_dir: Path,
    output_dir: Path,
    *,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Evaluate the composed three-class system on validation only; locked test is not addressable here."""

    validate_stage_pair(stage1_config, stage2_config)
    summary = validate_manifest_bundle(stage1_config.manifest_path)
    if stage1_config.manifest_path.resolve() != stage2_config.manifest_path.resolve():
        raise ValueError("FT-009B stages must use the same locked manifest bundle")
    rows = read_manifest(stage1_config.manifest_path / "val_manifest.csv")
    rows = _balanced_rows(rows, max_rows)
    expected_paths = [row["relative_path"] for row in rows]

    stage1, stage1_paths, stage1_metadata = collect_all_row_probabilities(stage1_config, stage1_run_dir, rows)
    stage2, stage2_paths, stage2_metadata = collect_all_row_probabilities(stage2_config, stage2_run_dir, rows)
    assert_exact_path_alignment(expected_paths, stage1_paths, "Stage 1")
    assert_exact_path_alignment(expected_paths, stage2_paths, "Stage 2")
    combined = compose_hierarchical_probabilities(stage1, stage2)
    labels = np.asarray([int(row["class_index"]) for row in rows], dtype=np.int64)
    metrics = calculate_metrics(labels, combined, THREE_CLASS_NAMES)
    metrics.update(
        {
            "experiment_id": "FT-009B-HIERARCHICAL-S42",
            "manifest_version": summary["manifest_version"],
            "probability_rule": "P(N); P(P)*P(V|P); P(P)*P(B|P)",
            "locked_test_accessed": False,
            "validation_rows": len(rows),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_evaluation_artifacts(
        output_dir,
        metrics,
        labels,
        combined,
        expected_paths,
        THREE_CLASS_NAMES,
        split="validation",
    )
    _write_component_probabilities(output_dir / "component_probabilities.csv", rows, stage1, stage2, combined)
    result = {
        "schema_version": 1,
        "created_at": now_iso(),
        "experiment_id": "FT-009B-HIERARCHICAL-S42",
        "status": "completed",
        "locked_test_accessed": False,
        "manifest_version": summary["manifest_version"],
        "validation_rows": len(rows),
        "probability_rule": {
            "normal": "P_stage1(Normal)",
            "viral": "P_stage1(Pneumonia) * P_stage2(Viral | Pneumonia)",
            "bacterial": "P_stage1(Pneumonia) * P_stage2(Bacterial | Pneumonia)",
        },
        "stage1": stage1_metadata,
        "stage2": stage2_metadata,
        "metrics": metrics,
    }
    atomic_write_json(output_dir / "hierarchical_summary.json", result)
    return result
