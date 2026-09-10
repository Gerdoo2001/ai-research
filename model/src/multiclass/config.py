"""Experiment configuration, validation, and stable hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .preprocessing import SUPPORTED_PREPROCESSING


REPO_ROOT = Path(__file__).resolve().parents[3]
MULTICLASS_CLASS_NAMES = ("Normal", "Viral", "Bacterial")
BINARY_CLASS_NAMES = ("Normal", "Pneumonia")
VIRAL_BACTERIAL_CLASS_NAMES = ("Viral", "Bacterial")


@dataclass
class AugmentationConfig:
    horizontal_flip_probability: float = 0.5
    rotation_degrees: float = 5.0
    translation_fraction: float = 0.0
    scale_min: float = 1.0
    scale_max: float = 1.0


@dataclass
class RoiConfig:
    """Optional, frozen region-of-interest preprocessing policy."""

    mode: str = "none"
    bbox_manifest: str | None = None
    segmentation_model: str | None = None
    threshold: float | None = None
    margin_fraction: float | None = None


@dataclass
class ExperimentConfig:
    schema_version: int = 1
    task_mode: str = "multiclass"
    class_names: list[str] = field(default_factory=lambda: list(MULTICLASS_CLASS_NAMES))
    architecture: str = "densenet121"
    weights: str = "imagenet"
    freeze_backbone: bool = False
    preprocessing: str = "baseline"
    input_size: int = 224
    batch_size: int = 16
    epochs: int = 12
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "reduce_on_plateau"
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    loss: str = "cross_entropy"
    focal_gamma: float | None = None
    class_weighting: str = "balanced"
    label_smoothing: float = 0.0
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda"
    allow_cpu_fallback: bool = False
    amp: bool = True
    early_stopping_patience: int = 4
    minimum_delta: float = 1e-4
    manifest_dir: str = "model/data_manifests"
    results_dir: str = "model/results/runs"
    tensorboard: bool = True
    smoke_test: bool = False
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    documentation_interval: int = 1
    controlled_stop_after_epoch: int | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.augmentation, dict):
            self.augmentation = AugmentationConfig(**self.augmentation)
        if isinstance(self.roi, dict):
            self.roi = RoiConfig(**self.roi)
        self.architecture = self.architecture.lower()
        self.preprocessing = self.preprocessing.lower()
        self.optimizer = self.optimizer.lower()
        self.scheduler = self.scheduler.lower()
        self.loss = self.loss.lower()
        self.validate()

    @property
    def num_classes(self) -> int:
        return 3 if self.task_mode == "multiclass" else 2

    @property
    def resolved_class_names(self) -> tuple[str, ...]:
        if self.task_mode == "multiclass":
            return MULTICLASS_CLASS_NAMES
        if self.task_mode == "viral_bacterial":
            return VIRAL_BACTERIAL_CLASS_NAMES
        return BINARY_CLASS_NAMES

    @property
    def manifest_path(self) -> Path:
        path = Path(self.manifest_dir)
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def results_path(self) -> Path:
        path = Path(self.results_dir)
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def effective_focal_gamma(self) -> float:
        return 2.0 if self.focal_gamma is None else float(self.focal_gamma)

    @property
    def roi_manifest_path(self) -> Path | None:
        if self.roi.bbox_manifest is None:
            return None
        path = Path(self.roi.bbox_manifest)
        return path if path.is_absolute() else REPO_ROOT / path

    def validate(self) -> None:
        if self.task_mode not in {"binary", "multiclass", "viral_bacterial"}:
            raise ValueError("task_mode must be 'binary', 'multiclass', or 'viral_bacterial'")
        expected = {
            "binary": BINARY_CLASS_NAMES,
            "multiclass": MULTICLASS_CLASS_NAMES,
            "viral_bacterial": VIRAL_BACTERIAL_CLASS_NAMES,
        }[self.task_mode]
        if tuple(self.class_names) != tuple(expected):
            raise ValueError(f"class_names must be exactly {list(expected)} for {self.task_mode}")
        if self.input_size < 64 or self.batch_size < 1 or self.epochs < 1:
            raise ValueError("input_size >= 64, batch_size >= 1, and epochs >= 1 are required")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.controlled_stop_after_epoch is not None and self.controlled_stop_after_epoch < 1:
            raise ValueError("controlled_stop_after_epoch must be positive when set")
        if not 0.0 <= self.augmentation.horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be within [0, 1]")
        if not 0.0 <= self.augmentation.translation_fraction <= 0.15:
            raise ValueError("translation_fraction must be within [0, 0.15]")
        if not 0.8 <= self.augmentation.scale_min <= self.augmentation.scale_max <= 1.2:
            raise ValueError("augmentation scale range must satisfy 0.8 <= scale_min <= scale_max <= 1.2")
        if self.roi.mode not in {"none", "pspnet_bbox"}:
            raise ValueError("roi.mode must be 'none' or 'pspnet_bbox'")
        if self.roi.mode == "none":
            if any(
                value is not None
                for value in (
                    self.roi.bbox_manifest,
                    self.roi.segmentation_model,
                    self.roi.threshold,
                    self.roi.margin_fraction,
                )
            ):
                raise ValueError("ROI fields must be null when roi.mode is 'none'")
        else:
            if not self.roi.bbox_manifest:
                raise ValueError("roi.bbox_manifest is required for pspnet_bbox mode")
            if self.roi.segmentation_model != "torchxrayvision_chestx_det_pspnet":
                raise ValueError("FT-005 requires the frozen TorchXRayVision ChestX-Det PSPNet")
            if self.roi.threshold is None or not 0.0 < self.roi.threshold < 1.0:
                raise ValueError("roi.threshold must be within (0, 1)")
            if self.roi.margin_fraction is None or not 0.0 <= self.roi.margin_fraction <= 0.5:
                raise ValueError("roi.margin_fraction must be within [0, 0.5]")
        if self.preprocessing not in SUPPORTED_PREPROCESSING:
            raise ValueError(f"unsupported preprocessing: {self.preprocessing}")
        if self.optimizer not in {"adam", "adamw", "sgd"}:
            raise ValueError(f"unsupported optimizer: {self.optimizer}")
        if self.scheduler not in {"none", "reduce_on_plateau", "cosine"}:
            raise ValueError(f"unsupported scheduler: {self.scheduler}")
        if self.loss not in {"cross_entropy", "focal"}:
            raise ValueError(f"unsupported loss: {self.loss}")
        if self.loss == "cross_entropy" and self.focal_gamma is not None:
            raise ValueError("focal_gamma must be null when loss is cross_entropy")
        if self.loss == "focal" and self.effective_focal_gamma <= 0.0:
            raise ValueError("focal_gamma must be positive for focal loss")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be within [0, 1)")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash_payload(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload.pop("run_id", None)
        payload.pop("controlled_stop_after_epoch", None)
        # Preserve hashes for all completed configurations created before the
        # optional spatial-augmentation fields were introduced.
        augmentation = payload["augmentation"]
        if augmentation["translation_fraction"] == 0.0:
            augmentation.pop("translation_fraction")
        if augmentation["scale_min"] == 1.0:
            augmentation.pop("scale_min")
        if augmentation["scale_max"] == 1.0:
            augmentation.pop("scale_max")
        # Preserve hashes for every completed experiment that predates optional
        # ROI preprocessing. ROI-enabled experiments retain the complete policy
        # in their hash payload.
        if payload["roi"]["mode"] == "none":
            payload.pop("roi")
        # Historical runs predate an explicit focal-gamma field. When absent,
        # the trainer retains its original gamma=2.0 behavior and old hashes.
        if payload["focal_gamma"] is None:
            payload.pop("focal_gamma")
        payload["manifest_dir"] = str(self.manifest_path.resolve())
        payload["results_dir"] = str(self.results_path.resolve())
        return payload

    @property
    def config_hash(self) -> str:
        encoded = json.dumps(self.hash_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def effective_run_id(self) -> str:
        return self.run_id or f"{self.architecture}_{self.preprocessing}_s{self.seed}_{self.config_hash[:10]}"


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if overrides:
        raw.update({key: value for key, value in overrides.items() if value is not None})
    return ExperimentConfig(**raw)
