"""Locked-manifest dataset and data-loader construction."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .preprocessing import build_transform


EXPECTED_MANIFEST_VERSION = "three_class_seed42_3b815cf769c8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest_bundle(manifest_dir: Path) -> dict:
    summary_path = manifest_dir / "split_summary.json"
    mapping_path = manifest_dir / "class_mapping.json"
    if not summary_path.exists() or not mapping_path.exists():
        raise FileNotFoundError(f"locked manifest bundle is incomplete: {manifest_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if summary["manifest_version"] != EXPECTED_MANIFEST_VERSION:
        raise RuntimeError(
            f"unexpected manifest version {summary['manifest_version']}; expected {EXPECTED_MANIFEST_VERSION}"
        )
    if mapping["class_order"] != ["Normal", "Viral", "Bacterial"]:
        raise RuntimeError("locked class order is not Normal, Viral, Bacterial")
    for filename, expected in summary["manifest_hashes"].items():
        path = manifest_dir / filename
        if sha256_file(path) != expected:
            raise RuntimeError(f"manifest digest mismatch: {path}")
    return summary


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"filepath", "relative_path", "assigned_split", "class_name", "class_index", "split_group_id"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise RuntimeError(f"manifest {path} lacks columns: {sorted(missing)}")
    return rows


def filter_rows_for_task(rows: list[dict[str, str]], task_mode: str) -> list[dict[str, str]]:
    """Return only rows that carry a defined target for the requested task."""

    if task_mode == "viral_bacterial":
        return [row for row in rows if int(row["class_index"]) in {1, 2}]
    if task_mode in {"binary", "multiclass"}:
        return rows
    raise ValueError(f"unsupported task_mode: {task_mode}")


def remap_class_index(original_label: int, task_mode: str) -> int:
    """Map the locked three-class label to the target space without changing the manifest."""

    if task_mode == "multiclass":
        return int(original_label)
    if task_mode == "binary":
        return int(original_label > 0)
    if task_mode == "viral_bacterial":
        if original_label not in {1, 2}:
            raise ValueError("Normal has no target in the Viral/Bacterial training task")
        return int(original_label - 1)
    raise ValueError(f"unsupported task_mode: {task_mode}")


def _center_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def apply_normalized_roi_crop(image: Image.Image, normalized_bbox: list[float]) -> Image.Image:
    """Reproduce the audited center-square frame, then apply a normalized ROI box."""

    if len(normalized_bbox) != 4 or any(not np.isfinite(value) for value in normalized_bbox):
        raise RuntimeError(f"invalid normalized ROI bbox: {normalized_bbox}")
    x0, y0, x1, y1 = (float(value) for value in normalized_bbox)
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise RuntimeError(f"normalized ROI bbox is outside [0,1]: {normalized_bbox}")
    square = _center_square(image)
    side = square.size[0]
    pixel_bbox = (
        max(0, min(side - 1, math.floor(x0 * side))),
        max(0, min(side - 1, math.floor(y0 * side))),
        max(1, min(side, math.ceil(x1 * side))),
        max(1, min(side, math.ceil(y1 * side))),
    )
    if pixel_bbox[2] <= pixel_bbox[0] or pixel_bbox[3] <= pixel_bbox[1]:
        raise RuntimeError(f"ROI bbox collapses after pixel conversion: {normalized_bbox}")
    return square.crop(pixel_bbox)


def load_roi_bbox_records(config, expected_rows: list[dict[str, str]], *, split: str) -> dict[str, dict]:
    """Load and strictly validate the frozen train/validation ROI coordinate manifest."""

    if config.roi.mode == "none":
        return {}
    if split == "test":
        raise PermissionError(
            "FT-005 ROI coordinates cover train/validation only; locked-test ROI generation is prohibited before selection"
        )
    manifest_path = config.roi_manifest_path
    if manifest_path is None or not manifest_path.exists():
        raise FileNotFoundError(f"ROI bbox manifest is unavailable: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = payload.get("policy", {})
    if payload.get("status") != "complete" or payload.get("locked_test_accessed") is not False:
        raise RuntimeError("ROI bbox manifest is incomplete or does not certify locked-test isolation")
    if policy.get("mode") != config.roi.mode:
        raise RuntimeError("ROI mode differs from the frozen experiment configuration")
    if policy.get("segmentation_model") != config.roi.segmentation_model:
        raise RuntimeError("ROI segmentation model differs from the frozen experiment configuration")
    if float(policy.get("threshold")) != float(config.roi.threshold):
        raise RuntimeError("ROI threshold differs from the frozen experiment configuration")
    if float(policy.get("margin_fraction")) != float(config.roi.margin_fraction):
        raise RuntimeError("ROI margin differs from the frozen experiment configuration")

    records = {record["relative_path"]: record for record in payload.get("records", [])}
    expected = {row["relative_path"]: row for row in expected_rows}
    missing = sorted(set(expected) - set(records))
    if missing:
        raise RuntimeError(f"ROI bbox manifest lacks {len(missing)} required {split} records; first={missing[0]}")
    selected: dict[str, dict] = {}
    for relative_path, row in expected.items():
        record = records[relative_path]
        if record.get("assigned_split") != split:
            raise RuntimeError(f"ROI split mismatch for {relative_path}")
        if record.get("cryptographic_hash") != row.get("cryptographic_hash"):
            raise RuntimeError(f"ROI source-image digest mismatch for {relative_path}")
        if record.get("bbox_normalized") is None:
            raise RuntimeError(f"ROI bbox is missing for {relative_path}")
        selected[relative_path] = record
    return selected


class ManifestDataset:
    def __init__(
        self,
        rows: list[dict[str, str]],
        transform,
        *,
        task_mode: str,
        roi_records: dict[str, dict] | None = None,
    ) -> None:
        from torch.utils.data import Dataset

        if not issubclass(type(self), Dataset):
            # PyTorch only requires the dataset protocol; this keeps imports lazy for audit tooling.
            pass
        self.rows = rows
        self.transform = transform
        self.task_mode = task_mode
        self.roi_records = roi_records or {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = Path(row["filepath"])
        with Image.open(path) as image:
            prepared = image.copy()
        if self.roi_records:
            record = self.roi_records.get(row["relative_path"])
            if record is None:
                raise RuntimeError(f"ROI record unavailable for {row['relative_path']}")
            prepared = apply_normalized_roi_crop(prepared, record["bbox_normalized"])
        tensor = self.transform(prepared)
        original_label = int(row["class_index"])
        label = remap_class_index(original_label, self.task_mode)
        return tensor, label, row["relative_path"]


def _seed_worker(worker_id: int) -> None:
    import torch

    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _balanced_subset(rows: list[dict[str, str]], limit: int | None, seed: int) -> list[dict[str, str]]:
    if limit is None or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_class.setdefault(row["class_name"], []).append(row)
    for class_rows in by_class.values():
        rng.shuffle(class_rows)
    selected: list[dict[str, str]] = []
    classes = sorted(by_class)
    while len(selected) < limit and classes:
        next_classes: list[str] = []
        for name in classes:
            if by_class[name] and len(selected) < limit:
                selected.append(by_class[name].pop())
            if by_class[name]:
                next_classes.append(name)
        classes = next_classes
    return selected


def build_dataloaders(config):
    import torch
    from torch.utils.data import DataLoader

    summary = validate_manifest_bundle(config.manifest_path)
    train_rows = read_manifest(config.manifest_path / "train_manifest.csv")
    val_rows = read_manifest(config.manifest_path / "val_manifest.csv")
    train_rows = filter_rows_for_task(train_rows, config.task_mode)
    val_rows = filter_rows_for_task(val_rows, config.task_mode)
    train_rows = _balanced_subset(train_rows, config.max_train_samples, config.seed)
    val_rows = _balanced_subset(val_rows, config.max_val_samples, config.seed + 1)

    train_roi = load_roi_bbox_records(config, train_rows, split="train")
    val_roi = load_roi_bbox_records(config, val_rows, split="val")

    train_dataset = ManifestDataset(
        train_rows,
        build_transform(config, training=True),
        task_mode=config.task_mode,
        roi_records=train_roi,
    )
    val_dataset = ManifestDataset(
        val_rows,
        build_transform(config, training=False),
        task_mode=config.task_mode,
        roi_records=val_roi,
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.device == "cuda",
        "worker_init_fn": _seed_worker,
        "generator": generator,
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader, summary


def build_evaluation_loader(config, split: str, *, allow_locked_test: bool = False):
    import torch
    from torch.utils.data import DataLoader

    if split not in {"val", "test"}:
        raise ValueError("evaluation split must be 'val' or 'test'")
    if split == "test" and not allow_locked_test:
        raise PermissionError("locked test loading requires explicit allow_locked_test=True")
    validate_manifest_bundle(config.manifest_path)
    rows = read_manifest(config.manifest_path / f"{split}_manifest.csv")
    rows = filter_rows_for_task(rows, config.task_mode)
    roi_records = load_roi_bbox_records(config, rows, split=split)
    dataset = ManifestDataset(
        rows,
        build_transform(config, training=False),
        task_mode=config.task_mode,
        roi_records=roi_records,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device == "cuda",
        worker_init_fn=_seed_worker,
        persistent_workers=config.num_workers > 0,
    )
