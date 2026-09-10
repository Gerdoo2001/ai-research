"""Build black/gray report figures from one real manifest-listed X-ray."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "model" / "src"))

from multiclass.preprocessing import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    _clahe,
    _histogram_equalization,
    _pseudo_color,
)


OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SAMPLE = (
    ROOT
    / "model"
    / "datasets"
    / "chest_xray"
    / "train"
    / "PNEUMONIA"
    / "person1000_virus_1681.jpeg"
)

BLACK = (0, 0, 0)
DARK = (45, 45, 45)
MID = (120, 120, 120)
LIGHT = (238, 238, 238)
WHITE = (255, 255, 255)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size=size)


F_TITLE = font(42, True)
F_SUB = font(25)
F_LABEL = font(27, True)
F_BODY = font(23)
F_SMALL = font(19)


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(
        image.convert("RGB"), size, method=Image.Resampling.BILINEAR, centering=(0.5, 0.5)
    )


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt, fill=BLACK) -> None:
    box = draw.multiline_textbbox((0, 0), text, font=fnt, align="center", spacing=4)
    w = box[2] - box[0]
    h = box[3] - box[1]
    draw.multiline_text((xy[0] - w / 2, xy[1] - h / 2), text, font=fnt, fill=fill, align="center", spacing=4)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], width: int = 6) -> None:
    draw.line([start, end], fill=BLACK, width=width)
    x, y = end
    if abs(end[0] - start[0]) >= abs(end[1] - start[1]):
        direction = 1 if end[0] > start[0] else -1
        points = [(x, y), (x - direction * 20, y - 13), (x - direction * 20, y + 13)]
    else:
        direction = 1 if end[1] > start[1] else -1
        points = [(x, y), (x - 13, y - direction * 20), (x + 13, y - direction * 20)]
    draw.polygon(points, fill=BLACK)


def framed_tile(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    left: int,
    top: int,
    width: int,
    height: int,
    label: str,
    detail: str,
) -> None:
    draw.rounded_rectangle(
        [left, top, left + width, top + height], radius=18, fill=WHITE, outline=BLACK, width=4
    )
    preview_height = height - 112
    preview = fit(image, (width - 24, preview_height - 12))
    canvas.paste(preview, (left + 12, top + 12))
    text_center(draw, (left + width // 2, top + preview_height + 22), label, F_LABEL)
    text_center(draw, (left + width // 2, top + preview_height + 67), detail, F_SMALL, DARK)


def data_prep_figure(original: Image.Image) -> None:
    canvas = Image.new("RGB", (2000, 900), WHITE)
    draw = ImageDraw.Draw(canvas)
    text_center(draw, (1000, 52), "ONE-IMAGE DATA PREPARATION FLOW", F_TITLE)
    text_center(
        draw,
        (1000, 102),
        "Actual training example: person1000_virus_1681.jpeg  |  Label: Viral Pneumonia",
        F_SUB,
        DARK,
    )

    rgb = original.convert("RGB")
    ge = _histogram_equalization(original)
    cropped = fit(ge, (224, 224))
    tensor = np.asarray(cropped, dtype=np.float32).transpose(2, 0, 1) / 255.0
    normalized = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    preview = normalized.transpose(1, 2, 0)
    preview = (preview - preview.min()) / max(float(preview.max() - preview.min()), 1e-6)
    preview_image = Image.fromarray(
        np.uint8(np.clip(preview * 255.0, 0, 255)), mode="RGB"
    ).convert("L").convert("RGB")

    stages = [
        (original, "1  SOURCE IMAGE", "Manifest verified"),
        (rgb, "2  OPEN + RGB", "L -> replicated RGB"),
        (ge, "3  ENHANCEMENT", "GE champion example"),
        (cropped, "4  RESIZE + CROP", "224 x 224 pixels"),
        (preview_image, "5  TENSOR READY", "float32; normalized; 3 x 224 x 224"),
    ]
    tile_w, tile_h, gap, left0, top = 330, 570, 68, 40, 175
    for i, (img, label, detail) in enumerate(stages):
        left = left0 + i * (tile_w + gap)
        framed_tile(canvas, draw, img, left, top, tile_w, tile_h, label, detail)
        if i < len(stages) - 1:
            arrow(draw, (left + tile_w + 10, top + 280), (left + tile_w + gap - 10, top + 280))
    text_center(
        draw,
        (1000, 810),
        "Duplicate removal and group-aware split assignment happen in the manifest before any pixel transform.",
        F_BODY,
        DARK,
    )
    canvas.save(OUT / "figure_11_1_data_preparation.png", quality=95)


def preprocessing_figure(original: Image.Image) -> None:
    canvas = Image.new("RGB", (1800, 1260), WHITE)
    draw = ImageDraw.Draw(canvas)
    text_center(draw, (900, 50), "SAME IMAGE UNDER EACH TESTED PREPROCESSING CONDITION", F_TITLE)
    text_center(draw, (900, 98), "Only this branch changes; split, label, augmentation controls, and model evaluation stay locked.", F_SUB, DARK)

    variants = [
        (original.convert("RGB"), "BASE / ORIGINAL", "RGB only"),
        (_clahe(original), "CLAHE", "clip 2.0; tiles 8 x 8"),
        (_histogram_equalization(original), "GLOBAL HE", "whole-image redistribution"),
        (_pseudo_color(original), "VIRIDIS", "exploratory pseudo-color"),
        (_clahe(_histogram_equalization(original)), "GE -> CLAHE", "global, then local"),
        (_histogram_equalization(_clahe(original)), "CLAHE -> GE", "local, then global"),
    ]
    tile_w, tile_h = 500, 480
    x_positions = [90, 650, 1210]
    y_positions = [165, 705]
    for i, (img, label, detail) in enumerate(variants):
        framed_tile(canvas, draw, img, x_positions[i % 3], y_positions[i // 3], tile_w, tile_h, label, detail)
    text_center(
        draw,
        (900, 1220),
        "The VIRIDIS tile retains algorithmic color; all document text, lines, and table styling remain black/gray.",
        F_SMALL,
        DARK,
    )
    canvas.save(OUT / "figure_11_2_preprocessing_variants.png", quality=95)


def augmentation_figure(original: Image.Image) -> None:
    canvas = Image.new("RGB", (1900, 850), WHITE)
    draw = ImageDraw.Draw(canvas)
    text_center(draw, (950, 52), "TRAINING-ONLY AUGMENTATION EXAMPLE", F_TITLE)
    text_center(draw, (950, 102), "Illustrative deterministic outcome from the same GE-preprocessed image", F_SUB, DARK)

    ge = _histogram_equalization(original)
    flipped = ImageOps.mirror(ge)
    rotated = flipped.rotate(5.0, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    final = fit(rotated, (224, 224))
    stages = [
        (ge, "1  PREPROCESSED", "GE output"),
        (flipped, "2  HORIZONTAL FLIP", "applied with p = 0.5"),
        (rotated, "3  SMALL ROTATION", "example +5 degrees"),
        (final, "4  FINAL MODEL VIEW", "resize/crop to 224 x 224"),
    ]
    tile_w, tile_h, gap, left0, top = 390, 540, 75, 35, 170
    for i, (img, label, detail) in enumerate(stages):
        left = left0 + i * (tile_w + gap)
        framed_tile(canvas, draw, img, left, top, tile_w, tile_h, label, detail)
        if i < len(stages) - 1:
            arrow(draw, (left + tile_w + 10, top + 270), (left + tile_w + gap - 10, top + 270))
    text_center(draw, (950, 780), "Validation and locked-test images do not receive random flips or rotations.", F_BODY, DARK)
    canvas.save(OUT / "figure_11_3_training_augmentation.png", quality=95)


def pipeline_figure(original: Image.Image) -> None:
    canvas = Image.new("RGB", (1900, 1220), WHITE)
    draw = ImageDraw.Draw(canvas)
    text_center(draw, (950, 52), "END-TO-END TRAINING AND VALIDATION PIPELINE", F_TITLE)
    crop = fit(_histogram_equalization(original), (224, 224))

    boxes = [
        (70, 150, "LOCKED MANIFEST", "Path, class, group,\nand split", crop),
        (660, 150, "IMAGE TRANSFORM", "Selected preprocessing\n+ train-only augmentation", crop),
        (1250, 150, "MODEL INPUT", "float32 tensor\n3 x 224 x 224", crop),
        (1250, 535, "CNN BACKBONE", "DenseNet / ResNet / YOLO classifier", None),
        (660, 535, "THREE LOGITS", "Normal | Viral | Bacterial", None),
        (70, 535, "TRAINING OBJECTIVE", "Class-weighted cross-entropy\nor tuned alternative", None),
        (70, 900, "VALIDATION", "No random augmentation\nsealed test remains unused", None),
        (660, 900, "MODEL SELECTION", "Macro-F1 primary\nbalanced accuracy + recalls", None),
        (1250, 900, "BEST CHECKPOINT", "Save weights from best\nvalidation epoch", None),
    ]
    bw, bh = 500, 250
    for left, top, title, detail, thumb in boxes:
        draw.rounded_rectangle([left, top, left + bw, top + bh], radius=20, fill=LIGHT, outline=BLACK, width=5)
        if thumb is not None:
            im = fit(thumb, (150, 150))
            canvas.paste(im, (left + 18, top + 50))
            text_center(draw, (left + 330, top + 80), title, F_LABEL)
            text_center(draw, (left + 330, top + 152), detail, F_BODY, DARK)
        else:
            text_center(draw, (left + bw // 2, top + 78), title, F_LABEL)
            text_center(draw, (left + bw // 2, top + 157), detail, F_BODY, DARK)

    arrow(draw, (570, 275), (650, 275))
    arrow(draw, (1160, 275), (1240, 275))
    arrow(draw, (1500, 410), (1500, 525))
    arrow(draw, (1240, 660), (1170, 660))
    arrow(draw, (650, 660), (580, 660))
    arrow(draw, (320, 795), (320, 890))
    arrow(draw, (570, 1025), (650, 1025))
    arrow(draw, (1160, 1025), (1240, 1025))
    canvas.save(OUT / "figure_11_4_training_pipeline.png", quality=95)


def main() -> None:
    if not SAMPLE.exists():
        raise FileNotFoundError(SAMPLE)
    original = Image.open(SAMPLE).convert("L")
    data_prep_figure(original)
    preprocessing_figure(original)
    augmentation_figure(original)
    pipeline_figure(original)
    for path in sorted(OUT.glob("*.png")):
        print(path, path.stat().st_size)


if __name__ == "__main__":
    main()
