"""Build three PSPNet recalibration variants for the manually rejected masks.

This experiment uses only the 21 validation-audit images that were marked "no"
during FT-002.  It never reads the locked-test manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_pspnet_lung_roi import (  # noqa: E402
    center_square,
    create_overlay,
    largest_component,
    mask_bbox,
    sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PROGRESS_PATH = (
    REPO_ROOT / "model" / "results" / "roi_pspnet_audit" / "manual_review" / "progress.json"
)
OUTPUT_ROOT = REPO_ROOT / "model" / "results" / "roi_pspnet_recalibration"
IMAGE_ROOT = OUTPUT_ROOT / "images"
PROGRESS_PATH = OUTPUT_ROOT / "progress.json"

CANDIDATES = (
    {
        "choice": "1",
        "name": "Mild expansion",
        "short_name": "MILD",
        "threshold": 0.40,
        "margin_fraction": 0.12,
        "reason": "Slightly expands the predicted lungs and preserves 12% padding around the crop.",
    },
    {
        "choice": "2",
        "name": "Moderate expansion",
        "short_name": "MODERATE",
        "threshold": 0.30,
        "margin_fraction": 0.15,
        "reason": "Expands weak lung predictions further and preserves 15% crop padding.",
    },
    {
        "choice": "3",
        "name": "Aggressive expansion",
        "short_name": "AGGRESSIVE",
        "threshold": 0.20,
        "margin_fraction": 0.18,
        "reason": "Uses the broadest lung mask and 18% padding; it may retain more anatomy but also more background.",
    },
)
VALID_CHOICES = {candidate["choice"] for candidate in CANDIDATES} | {"none"}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def source_path(relative_path: str) -> Path:
    return REPO_ROOT / "model" / "datasets" / "chest_xray" / relative_path


def candidate_public_definitions() -> list[dict[str, object]]:
    return [dict(candidate) for candidate in CANDIDATES]


def summarize(records: list[dict]) -> dict[str, object]:
    counts = {choice: 0 for choice in ("1", "2", "3", "none")}
    for record in records:
        choice = record.get("user_choice")
        if choice in counts:
            counts[choice] += 1
    reviewed = sum(counts.values())
    pending = [record for record in records if record.get("user_choice") not in VALID_CHOICES]
    return {
        "status": "completed" if records and not pending else "in_progress",
        "total_images": len(records),
        "reviewed_images": reviewed,
        "pending_count": len(pending),
        "choice_counts": counts,
        "none_count": counts["none"],
        "next_pending_review_order": pending[0]["review_order"] if pending else None,
        "next_pending_audit_index": pending[0]["audit_index"] if pending else None,
    }


def load_progress() -> dict:
    if not PROGRESS_PATH.exists():
        raise FileNotFoundError(
            f"Recalibration images have not been generated yet: {PROGRESS_PATH}. "
            "Run prepare_roi_recalibration_review.py first."
        )
    return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))


def record_choice(audit_index: int, choice: str, note: str | None = None) -> dict:
    if choice not in VALID_CHOICES:
        raise ValueError(f"invalid choice {choice!r}; expected 1, 2, 3, or none")
    payload = load_progress()
    matches = [record for record in payload["records"] if int(record["audit_index"]) == audit_index]
    if not matches:
        raise ValueError(f"unknown recalibration audit index: {audit_index}")
    matches[0]["user_choice"] = choice
    matches[0]["reviewed_at"] = now()
    matches[0]["review_note"] = note
    payload.update(summarize(payload["records"]))
    payload["updated_at"] = now()
    atomic_json(PROGRESS_PATH, payload)
    return payload


def variant_metrics(
    probabilities: np.ndarray,
    left_index: int,
    right_index: int,
    candidate: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    threshold = float(candidate["threshold"])
    left, left_components = largest_component(probabilities[left_index] >= threshold)
    right, right_components = largest_component(probabilities[right_index] >= threshold)
    union = left | right
    bbox = mask_bbox(union, float(candidate["margin_fraction"]))
    height, width = union.shape
    if bbox:
        x0, y0, x1, y1 = bbox
        bbox_area_fraction = ((x1 - x0) * (y1 - y0)) / float(height * width)
    else:
        bbox_area_fraction = 0.0
    metrics = {
        **candidate,
        "left_fraction": float(left.mean()),
        "right_fraction": float(right.mean()),
        "union_fraction": float(union.mean()),
        "left_raw_components": left_components,
        "right_raw_components": right_components,
        "bbox": list(bbox) if bbox else None,
        "bbox_area_fraction": bbox_area_fraction,
        "mask_height": int(height),
        "mask_width": int(width),
    }
    return left, right, metrics


def build_comparison(
    image: Image.Image,
    audit_index: int,
    review_order: int,
    total: int,
    class_name: str,
    relative_path: str,
    variants: list[tuple[np.ndarray, np.ndarray, dict[str, object]]],
    target: Path,
) -> None:
    panel_size = 512
    header_height = 92
    panels: list[Image.Image] = []

    original = image.convert("RGB").resize((panel_size, panel_size), Image.Resampling.LANCZOS)
    original_draw = ImageDraw.Draw(original)
    original_draw.rectangle((0, 0, panel_size, 42), fill="black")
    original_draw.text((10, 12), "ORIGINAL (no mask overlay)", fill="white")
    panels.append(original)

    for left, right, metrics in variants:
        caption = (
            f"{metrics['choice']} {metrics['short_name']} | threshold={metrics['threshold']:.2f} | "
            f"margin={100 * metrics['margin_fraction']:.0f}%"
        )
        rendered = create_overlay(image, left, right, tuple(metrics["bbox"]) if metrics["bbox"] else None, caption)
        panels.append(rendered)

    canvas = Image.new("RGB", (panel_size * len(panels), panel_size + header_height), "black")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (panel_size * index, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 10),
        f"RECALIBRATION {review_order}/{total} | Original audit index {audit_index}/90 | {class_name}",
        fill="white",
    )
    draw.text((12, 34), f"File: {relative_path}", fill=(215, 215, 215))
    draw.text(
        (12, 59),
        "Choose the best anatomically acceptable crop: 1=Mild  2=Moderate  3=Aggressive  N=None. "
        "Red/green=lungs; yellow=crop.",
        fill=(150, 255, 150),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=94)


def generate(device_name: str = "cuda") -> dict:
    os.environ.setdefault("PYTHONUTF8", "1")
    import torch
    import torchxrayvision as xrv

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ROI recalibration requested CUDA but CUDA is unavailable")
    device = torch.device(device_name)

    source_progress = json.loads(SOURCE_PROGRESS_PATH.read_text(encoding="utf-8"))
    rejected = sorted(
        (record for record in source_progress["records"] if record.get("user_decision") == "no"),
        key=lambda record: int(record["audit_index"]),
    )
    if len(rejected) != 21:
        raise RuntimeError(f"expected the 21 saved manual rejections, found {len(rejected)}")

    existing = json.loads(PROGRESS_PATH.read_text(encoding="utf-8")) if PROGRESS_PATH.exists() else {}
    prior_decisions = {
        int(record["audit_index"]): record
        for record in existing.get("records", [])
        if record.get("user_choice") in VALID_CHOICES
    }
    started_at = existing.get("started_at", now())
    initial = {
        "schema_version": 1,
        "experiment_id": "FT-002B-ROI-RECALIBRATION",
        "status": "loading_model",
        "started_at": started_at,
        "updated_at": now(),
        "locked_test_accessed": False,
        "source_split": "validation only",
        "source_rule": "Only images marked no in the completed FT-002 manual review",
        "source_manual_review_path": str(SOURCE_PROGRESS_PATH),
        "source_manual_review_sha256": sha256(SOURCE_PROGRESS_PATH),
        "original_rejected_configuration": {"threshold": 0.50, "margin_fraction": 0.08},
        "candidate_definitions": candidate_public_definitions(),
        "total_images": len(rejected),
        "completed_inference_images": 0,
        "records": [],
    }
    atomic_json(PROGRESS_PATH, initial)

    print("Loading TorchXRayVision ChestX-Det PSPNet on GPU...", flush=True)
    model = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
    left_index = model.targets.index("Left Lung")
    right_index = model.targets.index("Right Lung")
    weight_path = Path(model.weights_filename_local)
    model_details = {
        "package": "torchxrayvision",
        "package_version": version("torchxrayvision"),
        "architecture": "ChestX-Det PSPNet",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "input_channels": 1,
        "input_normalization": "TorchXRayVision [-1024, 1024] then model-internal ImageNet normalization",
        "lung_channels": {"left": left_index, "right": right_index},
        "weights_path": str(weight_path),
        "weights_sha256": sha256(weight_path),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }

    records: list[dict] = []
    clock = time.perf_counter()
    for review_order, rejected_record in enumerate(rejected, start=1):
        audit_index = int(rejected_record["audit_index"])
        path = source_path(rejected_record["relative_path"])
        with Image.open(path) as source:
            square = center_square(source.convert("L"))
            array = np.asarray(square, dtype=np.float32)
        normalized = xrv.datasets.normalize(array, 255)
        tensor = torch.from_numpy(normalized[None, None, ...]).float().to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(tensor)
        probabilities = torch.sigmoid(logits.float())[0].cpu().numpy()
        variants = [
            variant_metrics(probabilities, left_index, right_index, candidate)
            for candidate in CANDIDATES
        ]
        comparison_path = IMAGE_ROOT / f"{audit_index:03d}_{rejected_record['class_name'].lower()}.jpg"
        build_comparison(
            square,
            audit_index,
            review_order,
            len(rejected),
            rejected_record["class_name"],
            rejected_record["relative_path"],
            variants,
            comparison_path,
        )
        prior = prior_decisions.get(audit_index, {})
        record = {
            "review_order": review_order,
            "audit_index": audit_index,
            "class_name": rejected_record["class_name"],
            "relative_path": rejected_record["relative_path"],
            "original_user_decision": "no",
            "original_threshold": 0.50,
            "original_margin_fraction": 0.08,
            "comparison_image_path": str(comparison_path),
            "candidates": [metrics for _, _, metrics in variants],
            "user_choice": prior.get("user_choice"),
            "reviewed_at": prior.get("reviewed_at"),
            "review_note": prior.get("review_note"),
        }
        records.append(record)
        elapsed = time.perf_counter() - clock
        payload = {
            **initial,
            **summarize(records),
            "status": "generating_comparisons",
            "updated_at": now(),
            "model": model_details,
            "completed_inference_images": review_order,
            "current_audit_index": audit_index,
            "elapsed_seconds": elapsed,
            "estimated_remaining_seconds": (elapsed / review_order) * (len(rejected) - review_order),
            "records": records,
        }
        atomic_json(PROGRESS_PATH, payload)
        print(
            f"[{review_order:02d}/{len(rejected)}] audit {audit_index:02d} "
            f"{rejected_record['class_name']}: comparison ready",
            flush=True,
        )

    payload.update(summarize(records))
    payload["status"] = "completed" if payload["pending_count"] == 0 else "awaiting_recalibration_review"
    payload["updated_at"] = now()
    payload["completed_inference_images"] = len(records)
    payload["current_audit_index"] = None
    payload["estimated_remaining_seconds"] = 0.0
    payload["interpretation"] = (
        "This is an exploratory calibration on previously rejected validation-audit images. "
        "The preferred global policy must pass a new, independent validation-only audit before ROI classifier training."
    )
    atomic_json(PROGRESS_PATH, payload)
    print(f"Prepared {len(records)} comparisons at {IMAGE_ROOT}", flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--record", type=int, help="Original audit index to update")
    parser.add_argument("--choice", choices=("1", "2", "3", "none"))
    parser.add_argument("--note")
    args = parser.parse_args()
    if (args.record is None) != (args.choice is None):
        parser.error("--record and --choice must be used together")
    payload = (
        record_choice(args.record, args.choice, args.note)
        if args.record is not None
        else generate(args.device)
    )
    print(json.dumps({key: value for key, value in payload.items() if key not in {"records", "model"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
