"""FT-008 screening audit for two legacy U-Net lung-segmentation artifacts.

This script is deliberately an audit, not a classifier trainer. It reuses the
90 validation-only images from FT-002C so the two U-Nets can be compared with
the recorded PSPNet results on the same cases. The set is a screening set and
must not be described as a new independent confirmation cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = ("Normal", "Viral", "Bacterial")
MODEL_A_ID = "classic_unet_cxr_reg"
MODEL_B_ID = "legacy_full_unet"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for attempt in range(12):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(0.2 * (attempt + 1))


def center_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def largest_components(mask: np.ndarray, max_components: int = 2) -> tuple[np.ndarray, int]:
    """Keep up to the two largest foreground regions for a bilateral lung mask."""

    if max_components < 1:
        raise ValueError("max_components must be at least 1")
    raw = mask.astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    foreground_count = component_count - 1
    if foreground_count <= 0:
        return np.zeros_like(raw, dtype=bool), 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    selected = 1 + np.argsort(areas)[::-1][:max_components]
    cleaned = np.isin(labels, selected).astype(np.uint8) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(cleaned)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled.astype(bool), foreground_count


def mask_bbox(mask: np.ndarray, margin_fraction: float) -> tuple[int, int, int, int] | None:
    rows, columns = np.where(mask)
    if not len(rows):
        return None
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    margin_x = int(round((x1 - x0) * margin_fraction))
    margin_y = int(round((y1 - y0) * margin_fraction))
    height, width = mask.shape
    return max(0, x0 - margin_x), max(0, y0 - margin_y), min(width, x1 + margin_x), min(height, y1 + margin_y)


def quality_record(raw_mask: np.ndarray, cleaned: np.ndarray, margin_fraction: float, max_components: int = 2) -> dict:
    _, raw_components = largest_components(raw_mask, max_components=max_components)
    bbox = mask_bbox(cleaned, margin_fraction)
    mask_fraction = float(cleaned.mean())
    height, width = cleaned.shape
    if bbox:
        x0, y0, x1, y1 = bbox
        bbox_width_fraction = (x1 - x0) / float(width)
        bbox_height_fraction = (y1 - y0) / float(height)
        bbox_area_fraction = ((x1 - x0) * (y1 - y0)) / float(width * height)
    else:
        bbox_width_fraction = bbox_height_fraction = bbox_area_fraction = 0.0
    flags: list[str] = []
    if bbox is None:
        flags.append("empty_lung_mask")
    if not 0.18 <= mask_fraction <= 0.62:
        flags.append("lung_area_out_of_range")
    if not 0.45 <= bbox_width_fraction <= 1.0:
        flags.append("bbox_width_out_of_range")
    if not 0.45 <= bbox_height_fraction <= 1.0:
        flags.append("bbox_height_out_of_range")
    if raw_components > 5:
        flags.append("fragmented_raw_mask")
    return {
        "mask_fraction": mask_fraction,
        "raw_components": raw_components,
        "bbox": list(bbox) if bbox else None,
        "bbox_width_fraction": bbox_width_fraction,
        "bbox_height_fraction": bbox_height_fraction,
        "bbox_area_fraction": bbox_area_fraction,
        "automated_pass": not flags,
        "quality_flags": flags,
    }


def build_classic_unet(tf):
    layers = tf.keras.layers
    inputs = layers.Input((256, 256, 1))
    conv1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    conv1 = layers.Conv2D(32, 3, activation="relu", padding="same")(conv1)
    pool1 = layers.MaxPooling2D()(conv1)
    conv2 = layers.Conv2D(64, 3, activation="relu", padding="same")(pool1)
    conv2 = layers.Conv2D(64, 3, activation="relu", padding="same")(conv2)
    pool2 = layers.MaxPooling2D()(conv2)
    conv3 = layers.Conv2D(128, 3, activation="relu", padding="same")(pool2)
    conv3 = layers.Conv2D(128, 3, activation="relu", padding="same")(conv3)
    pool3 = layers.MaxPooling2D()(conv3)
    conv4 = layers.Conv2D(256, 3, activation="relu", padding="same")(pool3)
    conv4 = layers.Conv2D(256, 3, activation="relu", padding="same")(conv4)
    pool4 = layers.MaxPooling2D()(conv4)
    conv5 = layers.Conv2D(512, 3, activation="relu", padding="same")(pool4)
    conv5 = layers.Conv2D(512, 3, activation="relu", padding="same")(conv5)
    up6 = layers.concatenate([layers.Conv2DTranspose(256, 2, strides=2, padding="same")(conv5), conv4])
    conv6 = layers.Conv2D(256, 3, activation="relu", padding="same")(up6)
    conv6 = layers.Conv2D(256, 3, activation="relu", padding="same")(conv6)
    up7 = layers.concatenate([layers.Conv2DTranspose(128, 2, strides=2, padding="same")(conv6), conv3])
    conv7 = layers.Conv2D(128, 3, activation="relu", padding="same")(up7)
    conv7 = layers.Conv2D(128, 3, activation="relu", padding="same")(conv7)
    up8 = layers.concatenate([layers.Conv2DTranspose(64, 2, strides=2, padding="same")(conv7), conv2])
    conv8 = layers.Conv2D(64, 3, activation="relu", padding="same")(up8)
    conv8 = layers.Conv2D(64, 3, activation="relu", padding="same")(conv8)
    up9 = layers.concatenate([layers.Conv2DTranspose(32, 2, strides=2, padding="same")(conv8), conv1])
    conv9 = layers.Conv2D(32, 3, activation="relu", padding="same")(up9)
    conv9 = layers.Conv2D(32, 3, activation="relu", padding="same")(conv9)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(conv9)
    return tf.keras.Model(inputs, outputs, name=MODEL_A_ID)


def load_models(tf, model_a_path: Path, model_b_path: Path) -> tuple[dict, dict]:
    tf.keras.backend.clear_session()
    model_a = build_classic_unet(tf)
    model_a.load_weights(str(model_a_path))
    model_b = tf.keras.models.load_model(str(model_b_path), compile=False)
    for model_id, model in ((MODEL_A_ID, model_a), (MODEL_B_ID, model_b)):
        if tuple(model.input_shape[1:]) != (256, 256, 1):
            raise RuntimeError(f"{model_id} input contract differs from 256x256x1: {model.input_shape}")
        if tuple(model.output_shape[1:]) != (256, 256, 1):
            raise RuntimeError(f"{model_id} output contract differs from 256x256x1: {model.output_shape}")
    return {MODEL_A_ID: model_a, MODEL_B_ID: model_b}, {
        MODEL_A_ID: {
            "artifact_path": str(model_a_path),
            "artifact_sha256": sha256(model_a_path),
            "artifact_type": "weights_only",
            "architecture": "classic U-Net",
            "parameter_count": int(model_a.count_params()),
            "input_shape": list(model_a.input_shape),
            "output_shape": list(model_a.output_shape),
            "provenance_status": "legacy artifact; no in-repository training manifest or validated metric",
        },
        MODEL_B_ID: {
            "artifact_path": str(model_b_path),
            "artifact_sha256": sha256(model_b_path),
            "artifact_type": "complete_keras_model",
            "architecture": "legacy U-Net-like segmentation model",
            "parameter_count": int(model_b.count_params()),
            "input_shape": list(model_b.input_shape),
            "output_shape": list(model_b.output_shape),
            "provenance_status": "legacy artifact; no in-repository training manifest or validated metric",
        },
    }


def render_overlay(image: Image.Image, mask: np.ndarray, bbox: list[int] | None, color: tuple[int, int, int], title: str) -> Image.Image:
    base = np.asarray(image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32)
    resized_mask = cv2.resize(mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST).astype(bool)
    tint = np.array(color, dtype=np.float32)
    base[resized_mask] = base[resized_mask] * 0.62 + tint * 0.38
    edge = resized_mask & ~cv2.erode(resized_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    base[edge] = tint
    rendered = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(rendered)
    if bbox:
        scaled = tuple(int(value * 2) for value in bbox)
        draw.rectangle(scaled, outline=(255, 220, 0), width=4)
    draw.rectangle((0, 0, 512, 40), fill=(0, 0, 0))
    draw.text((8, 11), title, fill=(255, 255, 255))
    return rendered


def make_comparison(original: Image.Image, panel_a: Image.Image, panel_b: Image.Image, caption: str) -> Image.Image:
    original_panel = original.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(original_panel)
    draw.rectangle((0, 0, 512, 40), fill=(0, 0, 0))
    draw.text((8, 11), "Original center-square grayscale", fill=(255, 255, 255))
    canvas = Image.new("RGB", (1536, 562), "black")
    canvas.paste(original_panel, (0, 0))
    canvas.paste(panel_a, (512, 0))
    canvas.paste(panel_b, (1024, 0))
    footer = ImageDraw.Draw(canvas)
    footer.text((10, 530), caption[:220], fill=(255, 255, 255))
    return canvas


def screening_records(source_summary: dict) -> list[dict]:
    records = source_summary.get("records", [])
    if len(records) != 90:
        raise RuntimeError(f"FT-002C source must contain exactly 90 records; found {len(records)}")
    counts = Counter(record["class_name"] for record in records)
    if counts != Counter({name: 30 for name in CLASS_NAMES}):
        raise RuntimeError(f"FT-002C source is not 30-per-class: {dict(counts)}")
    if source_summary.get("locked_test_accessed") is not False:
        raise RuntimeError("source audit does not certify locked-test isolation")
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-summary", default="model/results/roi_pspnet_independent_audit/summary.json")
    parser.add_argument("--output-dir", default="model/results/roi_unet_screening_two_components")
    parser.add_argument("--experiment-id", default="FT-008A2-UNET-ROI-TWO-COMPONENT-SCREENING")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--margin-fraction", type=float, default=0.18)
    parser.add_argument("--max-components", type=int, default=2)
    args = parser.parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be within (0,1)")
    if not 0.0 <= args.margin_fraction <= 0.5:
        raise ValueError("margin_fraction must be within [0,0.5]")
    if not 1 <= args.max_components <= 4:
        raise ValueError("max_components must be within [1,4]")

    output_dir = REPO_ROOT / args.output_dir
    progress_path = output_dir / "progress.json"
    summary_path = output_dir / "summary.json"
    comparison_dir = output_dir / "comparisons"
    mask_dirs = {MODEL_A_ID: output_dir / "masks" / MODEL_A_ID, MODEL_B_ID: output_dir / "masks" / MODEL_B_ID}
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for directory in mask_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    source_path = REPO_ROOT / args.source_summary
    source_summary = read_json(source_path)
    selected = screening_records(source_summary)
    started_at = now()
    progress = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "status": "loading_models",
        "started_at": started_at,
        "updated_at": started_at,
        "locked_test_accessed": False,
        "source_split": "validation only",
        "screening_cohort": "same 90 images as FT-002C for matched head-to-head comparison",
        "independent_confirmation": False,
        "total_images": len(selected),
        "completed_images": 0,
        "threshold": args.threshold,
        "margin_fraction": args.margin_fraction,
        "max_components": args.max_components,
    }
    atomic_json(progress_path, progress)

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    model_a_path = REPO_ROOT / "model/outputs/cxr_reg_weights.best.hdf5"
    model_b_path = REPO_ROOT / "model/outputs/trained_model.hdf5"
    models, model_details = load_models(tf, model_a_path, model_b_path)
    runtime = {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "gpu_devices": [device.name for device in gpus],
        "input_preprocessing": "center-square grayscale; resize 256x256; float32 [0,1]",
        "output_postprocessing": (
            f"threshold {args.threshold}; retain up to {args.max_components} largest components for bilateral lungs; "
            f"fill/close; bbox margin {args.margin_fraction}"
        ),
    }
    progress.update(status="auditing", updated_at=now(), models=model_details, runtime=runtime)
    atomic_json(progress_path, progress)

    records: list[dict] = []
    started_clock = time.perf_counter()
    for index, source_record in enumerate(selected, start=1):
        row_started = time.perf_counter()
        supplied_filepath = source_record.get("filepath")
        image_path = Path(supplied_filepath) if supplied_filepath else Path()
        if not supplied_filepath or not image_path.is_file():
            relative_path = source_record["relative_path"]
            manifest_match = next(
                (record for record in source_summary["records"] if record["relative_path"] == relative_path),
                None,
            )
            if manifest_match and manifest_match.get("filepath"):
                image_path = Path(manifest_match["filepath"])
            else:
                import csv

                manifest_path = REPO_ROOT / "model/data_manifests/val_manifest.csv"
                with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
                    match = next((row for row in csv.DictReader(handle) if row["relative_path"] == relative_path), None)
                if match is None:
                    raise RuntimeError(f"screening image is absent from validation manifest: {relative_path}")
                image_path = Path(match["filepath"])
        with Image.open(image_path) as opened:
            square = center_square(opened.convert("L"))
        array = np.asarray(square.resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        batch = array[None, ..., None]
        model_records: dict[str, dict] = {}
        panels: dict[str, Image.Image] = {}
        safe_stem = f"{index:03d}_{source_record['class_name'].lower()}_{hashlib.sha1(source_record['relative_path'].encode()).hexdigest()[:10]}"
        for model_id, color in ((MODEL_A_ID, (64, 160, 255)), (MODEL_B_ID, (64, 255, 128))):
            prediction = np.asarray(models[model_id](batch, training=False).numpy()).squeeze()
            if prediction.shape != (256, 256) or not np.isfinite(prediction).all():
                raise RuntimeError(f"{model_id} returned invalid prediction {prediction.shape}")
            raw_mask = prediction >= args.threshold
            cleaned, _ = largest_components(raw_mask, max_components=args.max_components)
            quality = quality_record(raw_mask, cleaned, args.margin_fraction, max_components=args.max_components)
            mask_path = mask_dirs[model_id] / f"{safe_stem}.png"
            Image.fromarray(cleaned.astype(np.uint8) * 255, mode="L").save(mask_path)
            quality.update(
                probability_min=float(prediction.min()),
                probability_max=float(prediction.max()),
                probability_mean=float(prediction.mean()),
                mask_path=str(mask_path),
            )
            model_records[model_id] = quality
            panels[model_id] = render_overlay(
                square,
                cleaned,
                quality["bbox"],
                color,
                f"{model_id} pass={quality['automated_pass']} area={quality['mask_fraction']:.3f}",
            )
        comparison_path = comparison_dir / f"{safe_stem}.jpg"
        make_comparison(
            square,
            panels[MODEL_A_ID],
            panels[MODEL_B_ID],
            f"{index:03d}/090 | {source_record['class_name']} | {source_record['relative_path']} | Yellow = proposed 18% margin crop",
        ).save(comparison_path, quality=92)
        record = {
            "audit_index": index,
            "class_name": source_record["class_name"],
            "class_index": source_record["class_index"],
            "relative_path": source_record["relative_path"],
            "cryptographic_hash": source_record["cryptographic_hash"],
            "comparison_path": str(comparison_path),
            "models": model_records,
            "elapsed_seconds": time.perf_counter() - row_started,
        }
        records.append(record)
        elapsed = time.perf_counter() - started_clock
        progress.update(
            status="auditing",
            updated_at=now(),
            completed_images=index,
            current_image=source_record["relative_path"],
            current_class=source_record["class_name"],
            current_result=record,
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=(elapsed / index) * (len(selected) - index),
        )
        atomic_json(progress_path, progress)
        print(f"FT-008 U-Net audit {index:02d}/90 {source_record['class_name']}: {source_record['relative_path']}", flush=True)

    per_model = {}
    for model_id in (MODEL_A_ID, MODEL_B_ID):
        model_rows = [record["models"][model_id] for record in records]
        class_summary = {}
        for class_name in CLASS_NAMES:
            subset = [record["models"][model_id] for record in records if record["class_name"] == class_name]
            class_summary[class_name] = {
                "count": len(subset),
                "automated_passes": sum(bool(row["automated_pass"]) for row in subset),
                "automated_flags": sum(not bool(row["automated_pass"]) for row in subset),
                "automated_pass_rate": float(np.mean([row["automated_pass"] for row in subset])),
                "mean_mask_fraction": float(np.mean([row["mask_fraction"] for row in subset])),
            }
        flag_counts = Counter(flag for row in model_rows for flag in row["quality_flags"])
        per_model[model_id] = {
            "automated_passes": sum(bool(row["automated_pass"]) for row in model_rows),
            "automated_flags": sum(not bool(row["automated_pass"]) for row in model_rows),
            "automated_pass_rate": float(np.mean([row["automated_pass"] for row in model_rows])),
            "empty_masks": sum(row["bbox"] is None for row in model_rows),
            "flag_counts": dict(flag_counts),
            "class_summary": class_summary,
        }
    summary = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "status": "awaiting_manual_review",
        "started_at": started_at,
        "completed_at": now(),
        "locked_test_accessed": False,
        "classifier_training_started": False,
        "source_split": "validation only",
        "screening_cohort": "same 90 images as FT-002C for matched PSPNet comparison",
        "independent_confirmation": False,
        "source_summary_path": str(source_path),
        "source_summary_sha256": sha256(source_path),
        "threshold": args.threshold,
        "margin_fraction": args.margin_fraction,
        "max_components": args.max_components,
        "models": model_details,
        "runtime": runtime,
        "per_model": per_model,
        "records": records,
        "manual_review_required": True,
        "manual_review_status": "pending",
        "interpretation_boundary": (
            "This screening cohort was already used for FT-002C. Automated geometry cannot establish anatomical "
            "correctness and no Dice score is available without pixel-level ground truth. Select one U-Net/policy "
            "only after manual review, then confirm it on a new validation-only cohort before a 4,950-image "
            "feasibility audit or any DenseNet ROI training."
        ),
    }
    atomic_json(summary_path, summary)
    progress.update(
        status="awaiting_manual_review",
        updated_at=now(),
        completed_images=len(records),
        current_image=None,
        estimated_remaining_seconds=0.0,
        summary_path=str(summary_path),
        per_model=per_model,
    )
    atomic_json(progress_path, progress)
    print(summary_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
