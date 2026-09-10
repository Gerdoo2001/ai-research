"""Independent, auditable image preprocessing for chest X-rays."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


# The experiment treats enhancement order as part of the configuration.  Each
# operation appears at most once, giving the baseline plus every ordered subset
# of CLAHE (C), global histogram equalization (G), and VIRIDIS (V): 16 total.
PREPROCESSING_PIPELINES: dict[str, tuple[str, ...]] = {
    "baseline": (),
    "clahe": ("clahe",),
    "histogram_equalization": ("histogram_equalization",),
    "pseudo_color": ("pseudo_color",),
    "clahe_then_histogram_equalization": ("clahe", "histogram_equalization"),
    "histogram_equalization_then_clahe": ("histogram_equalization", "clahe"),
    "clahe_then_pseudo_color": ("clahe", "pseudo_color"),
    "pseudo_color_then_clahe": ("pseudo_color", "clahe"),
    "histogram_equalization_then_pseudo_color": (
        "histogram_equalization",
        "pseudo_color",
    ),
    "pseudo_color_then_histogram_equalization": (
        "pseudo_color",
        "histogram_equalization",
    ),
    "clahe_then_histogram_equalization_then_pseudo_color": (
        "clahe",
        "histogram_equalization",
        "pseudo_color",
    ),
    "clahe_then_pseudo_color_then_histogram_equalization": (
        "clahe",
        "pseudo_color",
        "histogram_equalization",
    ),
    "histogram_equalization_then_clahe_then_pseudo_color": (
        "histogram_equalization",
        "clahe",
        "pseudo_color",
    ),
    "histogram_equalization_then_pseudo_color_then_clahe": (
        "histogram_equalization",
        "pseudo_color",
        "clahe",
    ),
    "pseudo_color_then_clahe_then_histogram_equalization": (
        "pseudo_color",
        "clahe",
        "histogram_equalization",
    ),
    "pseudo_color_then_histogram_equalization_then_clahe": (
        "pseudo_color",
        "histogram_equalization",
        "clahe",
    ),
}
SUPPORTED_PREPROCESSING = frozenset(PREPROCESSING_PIPELINES)

_OPERATION_LABELS = {
    "clahe": "OpenCV CLAHE on grayscale (clip limit 2.0; 8 x 8 tiles)",
    "histogram_equalization": "global PIL ImageOps.equalize on grayscale",
    "pseudo_color": "OpenCV COLORMAP_VIRIDIS applied to grayscale",
}


@dataclass(frozen=True)
class PreprocessingMetadata:
    name: str
    parameters: dict[str, object]


def _clahe(image: Image.Image) -> Image.Image:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("CLAHE requires opencv-python") from exc
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return Image.fromarray(enhanced, mode="L").convert("RGB")


def _histogram_equalization(image: Image.Image) -> Image.Image:
    return ImageOps.equalize(image.convert("L")).convert("RGB")


def _pseudo_color(image: Image.Image) -> Image.Image:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("pseudo-coloring requires opencv-python") from exc
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    colored_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored_rgb, mode="RGB")


def preprocessing_metadata(name: str) -> PreprocessingMetadata:
    if name not in PREPROCESSING_PIPELINES:
        raise ValueError(f"unknown preprocessing pipeline: {name}")
    operations = PREPROCESSING_PIPELINES[name]
    if not operations:
        return PreprocessingMetadata(
            name=name,
            parameters={
                "operation_sequence": [],
                "conversion": "PIL RGB",
                "normalization": "ImageNet mean/std",
            },
        )

    parameters: dict[str, object] = {
        "operation_sequence": list(operations),
        "stages": [_OPERATION_LABELS[operation] for operation in operations],
        "normalization": "ImageNet mean/std",
        "final_channel_behavior": (
            "VIRIDIS RGB color map"
            if operations[-1] == "pseudo_color"
            else "grayscale replicated to RGB"
        ),
    }
    if "clahe" in operations:
        parameters.update(clip_limit=2.0, tile_grid_size=[8, 8])
    if "histogram_equalization" in operations:
        parameters["histogram_equalization"] = "global PIL ImageOps.equalize"
    if "pseudo_color" in operations:
        parameters.update(
            pseudo_color_map="OpenCV COLORMAP_VIRIDIS",
            clinical_information_claim="none",
        )
    return PreprocessingMetadata(name=name, parameters=parameters)


def visual_preprocess(image: Image.Image, name: str, input_size: int) -> Image.Image:
    """Return the pre-normalization RGB image used for visual QA."""

    transform = ImageTransform(name, input_size, training=False)
    enhanced = transform._enhance(image)
    return ImageOps.fit(
        enhanced,
        (input_size, input_size),
        method=Image.Resampling.BILINEAR,
        centering=(0.5, 0.5),
    )


class ImageTransform:
    def __init__(
        self,
        name: str,
        input_size: int,
        *,
        training: bool,
        horizontal_flip_probability: float = 0.0,
        rotation_degrees: float = 0.0,
        translation_fraction: float = 0.0,
        scale_min: float = 1.0,
        scale_max: float = 1.0,
    ) -> None:
        self.name = name
        self.input_size = input_size
        self.training = training
        self.horizontal_flip_probability = horizontal_flip_probability
        self.rotation_degrees = rotation_degrees
        self.translation_fraction = translation_fraction
        self.scale_min = scale_min
        self.scale_max = scale_max
        preprocessing_metadata(name)

    def _apply_spatial_jitter(self, image: Image.Image) -> Image.Image:
        if self.translation_fraction == 0.0 and self.scale_min == self.scale_max == 1.0:
            return image
        width, height = image.size
        translate_x = random.uniform(-self.translation_fraction, self.translation_fraction) * width
        translate_y = random.uniform(-self.translation_fraction, self.translation_fraction) * height
        scale = random.uniform(self.scale_min, self.scale_max)
        center_x, center_y = width / 2.0, height / 2.0
        inverse_scale = 1.0 / scale
        coefficients = (
            inverse_scale,
            0.0,
            center_x - (center_x + translate_x) * inverse_scale,
            0.0,
            inverse_scale,
            center_y - (center_y + translate_y) * inverse_scale,
        )
        return image.transform(
            image.size,
            Image.Transform.AFFINE,
            coefficients,
            resample=Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0),
        )

    def _enhance(self, image: Image.Image) -> Image.Image:
        enhanced = image.convert("RGB")
        for operation in PREPROCESSING_PIPELINES[self.name]:
            if operation == "clahe":
                enhanced = _clahe(enhanced)
            elif operation == "histogram_equalization":
                enhanced = _histogram_equalization(enhanced)
            elif operation == "pseudo_color":
                enhanced = _pseudo_color(enhanced)
            else:  # pragma: no cover - guarded by the declared pipeline table
                raise AssertionError(operation)
        return enhanced

    def __call__(self, image: Image.Image):
        import torch

        image = self._enhance(image)
        if self.training and random.random() < self.horizontal_flip_probability:
            image = ImageOps.mirror(image)
        if self.training and self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
        if self.training:
            image = self._apply_spatial_jitter(image)
        image = ImageOps.fit(
            image,
            (self.input_size, self.input_size),
            method=Image.Resampling.BILINEAR,
            centering=(0.5, 0.5),
        )
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(array))


def build_transform(config, *, training: bool) -> ImageTransform:
    return ImageTransform(
        config.preprocessing,
        config.input_size,
        training=training,
        horizontal_flip_probability=(
            config.augmentation.horizontal_flip_probability if training else 0.0
        ),
        rotation_degrees=config.augmentation.rotation_degrees if training else 0.0,
        translation_fraction=config.augmentation.translation_fraction if training else 0.0,
        scale_min=config.augmentation.scale_min if training else 1.0,
        scale_max=config.augmentation.scale_max if training else 1.0,
    )
