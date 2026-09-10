"""Precompute the frozen FT-005 PSPNet lung boxes for locked train/validation rows.

Only coordinates and audit metadata are persisted. No cropped images are saved,
and no locked-test image is opened. The partial checkpoint makes the operation
restartable after an ordinary interruption or computer shutdown.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model/src"))

from audit_pspnet_lung_roi import atomic_json, center_square, largest_component, mask_bbox, sha256  # noqa: E402
from multiclass.data import read_manifest, validate_manifest_bundle  # noqa: E402


EXPERIMENT_ID = "FT-005-PSPNET-ROI-BBOX-PRECOMPUTE"
MODEL_ID = "torchxrayvision_chestx_det_pspnet"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def policy_payload(threshold: float, margin_fraction: float) -> dict[str, object]:
    return {
        "mode": "pspnet_bbox",
        "segmentation_model": MODEL_ID,
        "threshold": threshold,
        "margin_fraction": margin_fraction,
        "source_frame": "center square of original image",
        "lung_channels": ["Left Lung", "Right Lung"],
        "mask_cleanup": "largest connected component per lung, binary hole fill, binary closing (2 iterations)",
        "crop_box": "union of cleaned left/right masks plus proportional margin",
        "empty_mask_behavior": "fail closed; abort manifest construction",
        "manual_filtering": False,
        "class_dependent_processing": False,
        "saved_artifact": "normalized bounding-box coordinates only; no cropped images",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--margin-fraction", type=float, default=0.18)
    parser.add_argument(
        "--output",
        default="model/data_manifests/roi_pspnet_t020_m018_train_val.json",
    )
    parser.add_argument(
        "--progress",
        default="model/results/roi_pspnet_bbox_precompute/progress.json",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be within (0,1)")
    if not 0.0 <= args.margin_fraction <= 0.5:
        raise ValueError("margin fraction must be within [0,0.5]")

    os.environ.setdefault("PYTHONUTF8", "1")
    import torch
    import torchxrayvision as xrv

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("FT-005 bbox precomputation requested CUDA, but CUDA is unavailable")
    device = torch.device(args.device)
    output_path = Path(args.output)
    progress_path = Path(args.progress)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    if not progress_path.is_absolute():
        progress_path = REPO_ROOT / progress_path

    manifest_dir = REPO_ROOT / "model/data_manifests"
    manifest_summary = validate_manifest_bundle(manifest_dir)
    train_path = manifest_dir / "train_manifest.csv"
    val_path = manifest_dir / "val_manifest.csv"
    rows = read_manifest(train_path) + read_manifest(val_path)
    if not rows or {row["assigned_split"] for row in rows} != {"train", "val"}:
        raise RuntimeError("FT-005 input must contain exactly the locked train and validation splits")
    rows = sorted(rows, key=lambda row: (row["assigned_split"], row["relative_path"]))
    expected_paths = {row["relative_path"] for row in rows}
    if len(expected_paths) != len(rows):
        raise RuntimeError("relative paths are not unique across locked train/validation manifests")

    policy = policy_payload(args.threshold, args.margin_fraction)
    started_at = now()
    records_by_path: dict[str, dict] = {}
    if progress_path.exists():
        existing = json.loads(progress_path.read_text(encoding="utf-8"))
        if existing.get("experiment_id") == EXPERIMENT_ID and existing.get("policy") == policy:
            if existing.get("manifest_hashes") == {
                "train_manifest.csv": sha256(train_path),
                "val_manifest.csv": sha256(val_path),
            }:
                started_at = existing.get("started_at", started_at)
                records_by_path = {
                    record["relative_path"]: record
                    for record in existing.get("records", [])
                    if record.get("relative_path") in expected_paths
                }
                for record in records_by_path.values():
                    record.setdefault("left_empty", float(record.get("left_mask_fraction", 0.0)) == 0.0)
                    record.setdefault("right_empty", float(record.get("right_mask_fraction", 0.0)) == 0.0)
                    record.setdefault("union_empty", float(record.get("union_mask_fraction", 0.0)) == 0.0)
                    record.setdefault(
                        "usable_under_frozen_policy",
                        not record["left_empty"]
                        and not record["right_empty"]
                        and record.get("bbox_normalized") is not None,
                    )

    base_progress = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "loading_model",
        "started_at": started_at,
        "updated_at": now(),
        "locked_test_images_accessed": False,
        "source_splits": ["train", "val"],
        "total_images": len(rows),
        "completed_images": len(records_by_path),
        "current_image": None,
        "policy": policy,
        "manifest_version": manifest_summary["manifest_version"],
        "manifest_hashes": {
            "train_manifest.csv": sha256(train_path),
            "val_manifest.csv": sha256(val_path),
        },
        "output_path": str(output_path),
        "device": str(device),
        "records": list(records_by_path.values()),
    }
    atomic_json(progress_path, base_progress)

    model = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
    left_index = model.targets.index("Left Lung")
    right_index = model.targets.index("Right Lung")
    weight_path = Path(model.weights_filename_local)
    model_details = {
        "package": "torchxrayvision",
        "package_version": version("torchxrayvision"),
        "model_id": MODEL_ID,
        "architecture": "ChestX-Det PSPNet",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "native_output_resolution": [512, 512],
        "input_channels": 1,
        "input_normalization": "TorchXRayVision [-1024,1024], followed by model-internal normalization",
        "left_lung_channel": left_index,
        "right_lung_channel": right_index,
        "weights_path": str(weight_path),
        "weights_sha256": sha256(weight_path),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    base_progress.update(status="precomputing", updated_at=now(), model=model_details)
    atomic_json(progress_path, base_progress)

    run_started = time.perf_counter()
    newly_completed = 0
    for row in rows:
        relative_path = row["relative_path"]
        if relative_path in records_by_path:
            continue
        image_started = time.perf_counter()
        path = Path(row["filepath"])
        if sha256(path) != row["cryptographic_hash"]:
            raise RuntimeError(f"source image digest mismatch: {relative_path}")
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
        left, left_components = largest_component(probabilities[left_index] >= args.threshold)
        right, right_components = largest_component(probabilities[right_index] >= args.threshold)
        union = left | right
        bbox = mask_bbox(union, args.margin_fraction)
        if bbox is None:
            pixel_bbox = None
            normalized_bbox = None
        else:
            x0, y0, x1, y1 = bbox
            pixel_bbox = [x0, y0, x1, y1]
            normalized_bbox = [x0 / 512.0, y0 / 512.0, x1 / 512.0, y1 / 512.0]
        record = {
            "relative_path": relative_path,
            "assigned_split": row["assigned_split"],
            "class_name": row["class_name"],
            "class_index": int(row["class_index"]),
            "cryptographic_hash": row["cryptographic_hash"],
            "source_width": int(row["image_width"]),
            "source_height": int(row["image_height"]),
            "bbox_pixels_512": pixel_bbox,
            "bbox_normalized": normalized_bbox,
            "left_empty": not left.any(),
            "right_empty": not right.any(),
            "union_empty": not union.any(),
            "usable_under_frozen_policy": bool(left.any() and right.any() and bbox is not None),
            "left_mask_fraction": float(left.mean()),
            "right_mask_fraction": float(right.mean()),
            "union_mask_fraction": float(union.mean()),
            "left_raw_component_count": left_components,
            "right_raw_component_count": right_components,
            "inference_seconds": time.perf_counter() - image_started,
        }
        records_by_path[relative_path] = record
        newly_completed += 1
        completed = len(records_by_path)
        elapsed = time.perf_counter() - run_started
        remaining = len(rows) - completed
        base_progress.update(
            status="precomputing",
            updated_at=now(),
            completed_images=completed,
            current_image=relative_path,
            current_split=row["assigned_split"],
            elapsed_this_session_seconds=elapsed,
            estimated_remaining_seconds=(elapsed / newly_completed) * remaining if newly_completed else None,
        )
        if newly_completed % args.checkpoint_every == 0 or completed == len(rows):
            base_progress["records"] = list(records_by_path.values())
            atomic_json(progress_path, base_progress)
        percent = completed * 100.0 / len(rows)
        eta = base_progress.get("estimated_remaining_seconds")
        eta_text = "calculating" if eta is None else f"{float(eta) / 60.0:.1f} min"
        print(
            f"\rFT-005 PSPNet ROI boxes: {completed:4d}/{len(rows)} "
            f"({percent:6.2f}%) | {row['assigned_split']:<5} | ETA {eta_text:<12}",
            end="",
            flush=True,
        )

    records = [records_by_path[row["relative_path"]] for row in rows]
    split_counts = Counter(record["assigned_split"] for record in records)
    class_counts = Counter(f"{record['assigned_split']}:{record['class_name']}" for record in records)
    completed_at = now()
    unusable = [record for record in records if not record["usable_under_frozen_policy"]]
    if unusable:
        failure_summary = {
            "unusable_record_count": len(unusable),
            "left_empty_count": sum(record["left_empty"] for record in records),
            "right_empty_count": sum(record["right_empty"] for record in records),
            "union_empty_count": sum(record["union_empty"] for record in records),
            "by_split": dict(Counter(record["assigned_split"] for record in unusable)),
            "by_class": dict(Counter(record["class_name"] for record in unusable)),
            "records": unusable,
        }
        feasibility_path = progress_path.parent / "feasibility_summary.json"
        feasibility = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "status": "failed_closed_empty_masks",
            "started_at": started_at,
            "completed_at": completed_at,
            "locked_test_images_accessed": False,
            "source_splits": ["train", "val"],
            "manifest_version": manifest_summary["manifest_version"],
            "manifest_hashes": base_progress["manifest_hashes"],
            "policy": policy,
            "model": model_details,
            "total_record_count": len(records),
            "usable_record_count": len(records) - len(unusable),
            "usable_record_rate": (len(records) - len(unusable)) / len(records),
            "split_counts": dict(split_counts),
            "split_class_counts": dict(class_counts),
            "failure_summary": failure_summary,
            "classifier_training_started": False,
            "decision": (
                "Do not create the ROI training manifest and do not start classification training; "
                "the frozen bilateral-mask policy cannot preprocess every development image uniformly."
            ),
        }
        atomic_json(feasibility_path, feasibility)
        base_progress.update(
            status="failed_closed_empty_masks",
            updated_at=completed_at,
            completed_images=len(records),
            current_image=None,
            estimated_remaining_seconds=0.0,
            failure_summary=failure_summary,
            feasibility_summary_path=str(feasibility_path),
            feasibility_summary_sha256=sha256(feasibility_path),
            records=records,
        )
        atomic_json(progress_path, base_progress)
        print()
        print(json.dumps({"status": base_progress["status"], **failure_summary}, indent=2))
        print("FT-005 classifier training was not authorized because the frozen policy lacks complete bilateral boxes.")
        return 2
    output = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "started_at": started_at,
        "completed_at": completed_at,
        "locked_test_accessed": False,
        "locked_test_images_accessed": False,
        "source_splits": ["train", "val"],
        "manifest_version": manifest_summary["manifest_version"],
        "manifest_hashes": base_progress["manifest_hashes"],
        "policy": policy,
        "model": model_details,
        "record_count": len(records),
        "split_counts": dict(split_counts),
        "split_class_counts": dict(class_counts),
        "records": records,
    }
    atomic_json(output_path, output)
    base_progress.update(
        status="complete",
        updated_at=completed_at,
        completed_images=len(records),
        current_image=None,
        estimated_remaining_seconds=0.0,
        output_sha256=sha256(output_path),
        records=records,
    )
    atomic_json(progress_path, base_progress)
    print()
    print(json.dumps({key: value for key, value in output.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
