"""Generate a stratified pediatric lung-mask audit without test-set access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


REPO_ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = ("Normal", "Viral", "Bacterial")


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
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2 * (attempt + 1))


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_stratified(rows: list[dict[str, str]], count_per_class: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for row in rows:
        by_class[row["class_name"]].append(row)
    selected = []
    for class_name in CLASS_NAMES:
        candidates = sorted(by_class[class_name], key=lambda row: row["relative_path"])
        rng.shuffle(candidates)
        if len(candidates) < count_per_class:
            raise RuntimeError(f"not enough {class_name} validation images for the ROI audit")
        selected.extend(candidates[:count_per_class])
    return selected


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, component_count = ndimage.label(mask)
    if component_count == 0:
        return np.zeros_like(mask, dtype=bool), 0
    areas = ndimage.sum(mask, labels, index=np.arange(1, component_count + 1))
    largest_label = int(np.argmax(areas)) + 1
    cleaned = labels == largest_label
    cleaned = ndimage.binary_fill_holes(cleaned)
    cleaned = ndimage.binary_closing(cleaned, iterations=2)
    return cleaned.astype(bool), int(component_count)


def center_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


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


def create_overlay(
    image: Image.Image,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    caption: str,
) -> Image.Image:
    base = np.asarray(image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32)
    for mask, color in ((left_mask, np.array([255, 64, 64])), (right_mask, np.array([64, 255, 64]))):
        base[mask] = base[mask] * 0.62 + color * 0.38
        edge = mask ^ ndimage.binary_erosion(mask)
        base[edge] = color
    rendered = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(rendered)
    if bbox:
        draw.rectangle(bbox, outline=(255, 220, 0), width=4)
    draw.rectangle((0, 0, 512, 38), fill=(0, 0, 0))
    draw.text((8, 10), caption[:80], fill=(255, 255, 255))
    return rendered


def make_contact_sheets(overlays: list[Path], output_dir: Path) -> list[str]:
    paths = []
    per_page, columns, rows = 15, 3, 5
    tile_width, tile_height = 320, 320
    for page_index in range(0, len(overlays), per_page):
        page_paths = overlays[page_index : page_index + per_page]
        canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "black")
        for index, path in enumerate(page_paths):
            with Image.open(path) as image:
                tile = image.convert("RGB").resize((tile_width, tile_height), Image.Resampling.LANCZOS)
            x = (index % columns) * tile_width
            y = (index // columns) * tile_height
            canvas.paste(tile, (x, y))
        target = output_dir / f"roi_audit_contact_sheet_{page_index // per_page + 1:02d}.jpg"
        canvas.save(target, quality=92)
        paths.append(str(target))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count-per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--margin-fraction", type=float, default=0.08)
    parser.add_argument("--output-dir", default="model/results/roi_pspnet_audit")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--experiment-id", default="FT-002-ROI-AUDIT")
    parser.add_argument(
        "--exclude-summary",
        action="append",
        default=[],
        help="Prior audit summary whose relative paths must be excluded; may be repeated",
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTHONUTF8", "1")
    import torch
    import torchxrayvision as xrv

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ROI audit requested CUDA but CUDA is unavailable")
    device = torch.device(args.device)
    output_dir = REPO_ROOT / args.output_dir
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    started_at = now()
    progress = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "status": "loading_model",
        "started_at": started_at,
        "updated_at": started_at,
        "locked_test_accessed": False,
        "source_split": "validation only",
        "requested_per_class": args.count_per_class,
        "total_images": args.count_per_class * len(CLASS_NAMES),
        "completed_images": 0,
        "current_image": None,
        "threshold": args.threshold,
        "margin_fraction": args.margin_fraction,
        "selection_seed": args.seed,
        "device": str(device),
    }
    atomic_json(progress_path, progress)

    manifest = REPO_ROOT / "model" / "data_manifests" / "val_manifest.csv"
    manifest_rows = read_manifest(manifest)
    excluded_paths: set[str] = set()
    exclusion_sources = []
    for supplied_path in args.exclude_summary:
        exclusion_path = Path(supplied_path)
        if not exclusion_path.is_absolute():
            exclusion_path = REPO_ROOT / exclusion_path
        exclusion_summary = json.loads(exclusion_path.read_text(encoding="utf-8"))
        source_paths = {record["relative_path"] for record in exclusion_summary["records"]}
        excluded_paths.update(source_paths)
        exclusion_sources.append(
            {
                "path": str(exclusion_path),
                "sha256": sha256(exclusion_path),
                "record_count": len(exclusion_summary["records"]),
                "unique_relative_path_count": len(source_paths),
            }
        )
    eligible_rows = [row for row in manifest_rows if row["relative_path"] not in excluded_paths]
    selected = select_stratified(eligible_rows, args.count_per_class, args.seed)
    selected_overlap = sorted({row["relative_path"] for row in selected} & excluded_paths)
    if selected_overlap:
        raise RuntimeError(f"excluded images entered the new audit: {selected_overlap[:5]}")
    progress.update(
        updated_at=now(),
        validation_manifest_path=str(manifest),
        validation_manifest_sha256=sha256(manifest),
        validation_manifest_rows=len(manifest_rows),
        excluded_unique_paths=len(excluded_paths),
        eligible_validation_rows=len(eligible_rows),
        exclusion_sources=exclusion_sources,
        selected_overlap_count=len(selected_overlap),
    )
    atomic_json(progress_path, progress)
    model = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
    left_index = model.targets.index("Left Lung")
    right_index = model.targets.index("Right Lung")
    weight_path = Path(model.weights_filename_local)
    model_details = {
        "package": "torchxrayvision",
        "package_version": version("torchxrayvision"),
        "model": repr(model),
        "architecture": "ChestX-Det PSPNet",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "native_resolution": [512, 512],
        "input_channels": 1,
        "input_normalization": "TorchXRayVision [-1024, 1024] then model-internal ImageNet normalization",
        "output_shape": [1, 14, 512, 512],
        "lung_channels": {"left": left_index, "right": right_index},
        "targets": list(model.targets),
        "weights_path": str(weight_path),
        "weights_sha256": sha256(weight_path),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }
    progress.update(status="auditing", model=model_details, updated_at=now())
    atomic_json(progress_path, progress)

    records: list[dict[str, object]] = []
    overlay_paths: list[Path] = []
    start_clock = time.perf_counter()
    for index, row in enumerate(selected, start=1):
        image_start = time.perf_counter()
        path = Path(row["filepath"])
        with Image.open(path) as source:
            square = center_square(source.convert("L"))
            array = np.asarray(square, dtype=np.float32)
        normalized = xrv.datasets.normalize(array, 255)
        tensor = torch.from_numpy(normalized[None, None, ...]).float().to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(tensor)
        probabilities = torch.sigmoid(logits.float())[0].cpu().numpy()
        raw_left = probabilities[left_index] >= args.threshold
        raw_right = probabilities[right_index] >= args.threshold
        left, left_components = largest_component(raw_left)
        right, right_components = largest_component(raw_right)
        union = left | right
        bbox = mask_bbox(union, args.margin_fraction)
        union_fraction = float(union.mean())
        left_fraction = float(left.mean())
        right_fraction = float(right.mean())
        if bbox:
            x0, y0, x1, y1 = bbox
            bbox_width_fraction = (x1 - x0) / 512.0
            bbox_height_fraction = (y1 - y0) / 512.0
            bbox_area_fraction = ((x1 - x0) * (y1 - y0)) / float(512 * 512)
        else:
            bbox_width_fraction = bbox_height_fraction = bbox_area_fraction = 0.0
        flags = []
        if not 0.08 <= left_fraction <= 0.38:
            flags.append("left_lung_area_out_of_range")
        if not 0.08 <= right_fraction <= 0.38:
            flags.append("right_lung_area_out_of_range")
        if not 0.18 <= union_fraction <= 0.62:
            flags.append("union_area_out_of_range")
        if not 0.45 <= bbox_width_fraction <= 1.0:
            flags.append("bbox_width_out_of_range")
        if not 0.45 <= bbox_height_fraction <= 1.0:
            flags.append("bbox_height_out_of_range")
        if left_components > 5 or right_components > 5:
            flags.append("fragmented_raw_mask")
        automated_pass = not flags
        safe_name = f"{index:03d}_{row['class_name'].lower()}_{hashlib.sha1(row['relative_path'].encode()).hexdigest()[:10]}.jpg"
        overlay_path = overlay_dir / safe_name
        caption = f"{index:03d} {row['class_name']} pass={automated_pass} union={union_fraction:.3f}"
        create_overlay(square, left, right, bbox, caption).save(overlay_path, quality=92)
        overlay_paths.append(overlay_path)
        duration = time.perf_counter() - image_start
        record = {
            "audit_index": index,
            "class_name": row["class_name"],
            "class_index": int(row["class_index"]),
            "relative_path": row["relative_path"],
            "cryptographic_hash": row["cryptographic_hash"],
            "original_width": int(row["image_width"]),
            "original_height": int(row["image_height"]),
            "left_fraction": left_fraction,
            "right_fraction": right_fraction,
            "union_fraction": union_fraction,
            "left_raw_components": left_components,
            "right_raw_components": right_components,
            "bbox": list(bbox) if bbox else None,
            "bbox_width_fraction": bbox_width_fraction,
            "bbox_height_fraction": bbox_height_fraction,
            "bbox_area_fraction": bbox_area_fraction,
            "automated_pass": automated_pass,
            "quality_flags": flags,
            "overlay_path": str(overlay_path),
            "inference_and_render_seconds": duration,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        }
        records.append(record)
        elapsed = time.perf_counter() - start_clock
        progress.update(
            status="auditing",
            updated_at=now(),
            completed_images=index,
            current_image=row["relative_path"],
            current_class=row["class_name"],
            current_result=record,
            passed_images=sum(bool(item["automated_pass"]) for item in records),
            flagged_images=sum(not bool(item["automated_pass"]) for item in records),
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=(elapsed / index) * (len(selected) - index),
        )
        atomic_json(progress_path, progress)

    manifest_path = output_dir / "audit_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(records[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            serialized = dict(record)
            serialized["quality_flags"] = "|".join(record["quality_flags"])
            serialized["bbox"] = json.dumps(record["bbox"])
            writer.writerow(serialized)
    contact_sheets = make_contact_sheets(overlay_paths, output_dir)
    class_summary = {}
    for class_name in CLASS_NAMES:
        subset = [record for record in records if record["class_name"] == class_name]
        class_summary[class_name] = {
            "count": len(subset),
            "passed": sum(bool(record["automated_pass"]) for record in subset),
            "flagged": sum(not bool(record["automated_pass"]) for record in subset),
            "pass_rate": float(np.mean([record["automated_pass"] for record in subset])),
            "mean_union_fraction": float(np.mean([record["union_fraction"] for record in subset])),
        }
    flagged = [record for record in records if not record["automated_pass"]]
    flag_counts = Counter(flag for record in flagged for flag in record["quality_flags"])
    pass_rate = sum(bool(record["automated_pass"]) for record in records) / len(records)
    summary = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "started_at": started_at,
        "completed_at": now(),
        "status": "awaiting_manual_review",
        "locked_test_accessed": False,
        "source_split": "validation only",
        "selection_seed": args.seed,
        "validation_manifest_path": str(manifest),
        "validation_manifest_sha256": sha256(manifest),
        "excluded_unique_paths": len(excluded_paths),
        "exclusion_sources": exclusion_sources,
        "selected_overlap_count": len(selected_overlap),
        "count_per_class": args.count_per_class,
        "total_images": len(records),
        "automated_pass_count": len(records) - len(flagged),
        "automated_flagged_count": len(flagged),
        "automated_pass_rate": pass_rate,
        "automated_gate_rule": "pass rate >= 95%, zero empty bilateral masks, then mandatory visual review",
        "automated_gate_passed": pass_rate >= 0.95 and all(record["bbox"] is not None for record in records),
        "manual_review_required": True,
        "manual_review_approved": False,
        "class_summary": class_summary,
        "flag_counts": dict(flag_counts),
        "model": model_details,
        "threshold": args.threshold,
        "margin_fraction": args.margin_fraction,
        "manifest_path": str(manifest_path),
        "contact_sheets": contact_sheets,
        "overlay_directory": str(overlay_dir),
        "records": records,
        "interpretation": (
            "Automated geometric checks cannot establish anatomical or clinical correctness. "
            "A human must review all contact sheets before ROI classification training is authorized."
        ),
    }
    atomic_json(output_dir / "summary.json", summary)
    approval = {
        "experiment_id": args.experiment_id,
        "approved": False,
        "reviewer": None,
        "reviewed_at": None,
        "review_notes": "Review every contact sheet; confirm lungs are not truncated and peripheral opacities remain inside the padded crop.",
        "summary_sha256": sha256(output_dir / "summary.json"),
    }
    atomic_json(output_dir / "manual_approval.json", approval)
    progress.update(
        status="awaiting_manual_review",
        updated_at=now(),
        completed_images=len(records),
        current_image=None,
        passed_images=len(records) - len(flagged),
        flagged_images=len(flagged),
        estimated_remaining_seconds=0.0,
        summary_path=str(output_dir / "summary.json"),
        contact_sheets=contact_sheets,
        automated_gate_passed=summary["automated_gate_passed"],
        manual_review_required=True,
    )
    atomic_json(progress_path, progress)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
