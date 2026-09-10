"""Keyboard GUI for persistent FT-008A2 two-model U-Net anatomical review."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_unet_roi_manual_review import PROGRESS_PATH, initialize, record_decision  # noqa: E402


WINDOW_NAME = "FT-008A2 U-Net Review | 1=Classic 2=Legacy 3=Both 0=Neither B=Back Q=Pause"


def load_progress() -> dict:
    initialize()
    return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))


def pending_or_last(progress: dict) -> int:
    pending = [record["audit_index"] for record in progress["records"] if not record.get("user_decision")]
    return pending[0] if pending else progress["records"][-1]["audit_index"]


def render(record: dict, progress: dict) -> np.ndarray:
    image = cv2.imread(record["review_image_path"], cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(record["review_image_path"])
    width = min(1500, image.shape[1])
    height = int(round(image.shape[0] * width / image.shape[1]))
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    footer_height = 130
    canvas = np.zeros((height + footer_height, width, 3), dtype=np.uint8)
    canvas[:height] = image

    classic = record["models"]["classic_unet_cxr_reg"]
    legacy = record["models"]["legacy_full_unet"]
    saved = (record.get("user_decision") or "pending").replace("_", " ").upper()
    line1 = f"Image {record['audit_index']}/90 | {record['class_name']} | Saved: {saved}"
    line2 = (
        f"Classic (middle): {'AUTO PASS' if classic['automated_pass'] else 'AUTO FLAG'}; "
        f"area={100 * classic['mask_fraction']:.1f}% | "
        f"Legacy (right): {'AUTO PASS' if legacy['automated_pass'] else 'AUTO FLAG'}; "
        f"area={100 * legacy['mask_fraction']:.1f}%"
    )
    line3 = "Choose anatomical acceptability: 1=CLASSIC ONLY   2=LEGACY ONLY   3=BOTH   0=NEITHER"
    line4 = (
        f"Saved {progress['reviewed_images']}/90 | Pending {progress['pending_count']} | "
        "B=previous image   Q/Esc=pause safely"
    )
    colors = ((255, 255, 255), (0, 215, 255), (120, 255, 120), (210, 210, 210))
    for index, (line, color) in enumerate(zip((line1, line2, line3, line4), colors)):
        cv2.putText(
            canvas,
            line,
            (12, height + 25 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.61,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def print_details(record: dict, progress: dict) -> None:
    classic = record["models"]["classic_unet_cxr_reg"]
    legacy = record["models"]["legacy_full_unet"]
    print("\n" + "=" * 112, flush=True)
    print(f"IMAGE {record['audit_index']}/90 | CLASS {record['class_name']}", flush=True)
    print(f"Source file: {record['relative_path']}", flush=True)
    for label, details in (("Classic U-Net", classic), ("Legacy U-Net", legacy)):
        flags = ", ".join(details["quality_flags"]) or "none"
        print(
            f"{label:<15} auto={'PASS' if details['automated_pass'] else 'FLAGGED':<7} "
            f"area={100 * details['mask_fraction']:.2f}% components={details['raw_components']} "
            f"flags={flags}",
            flush=True,
        )
    print(f"Saved decision: {(record.get('user_decision') or 'pending').upper()}", flush=True)
    print(
        f"Progress: {progress['reviewed_images']}/90 reviewed, {progress['pending_count']} pending",
        flush=True,
    )
    print("Controls: 1=Classic only, 2=Legacy only, 3=both, 0=neither, B=back, Q/Esc=pause", flush=True)


def main() -> int:
    progress = load_progress()
    current_index = pending_or_last(progress)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1500, 760)
    key_to_decision = {
        ord("1"): "classic_only",
        ord("2"): "legacy_only",
        ord("3"): "both",
        ord("0"): "neither",
    }

    while True:
        progress = load_progress()
        records_by_index = {record["audit_index"]: record for record in progress["records"]}
        record = records_by_index[current_index]
        print_details(record, progress)
        cv2.imshow(WINDOW_NAME, render(record, progress))
        key = cv2.waitKeyEx(0)
        if key in key_to_decision:
            decision = key_to_decision[key]
            progress = record_decision(current_index, decision, "keyboard GUI anatomical review")
            print(f"RECORDED: Image {current_index} = {decision.upper()}", flush=True)
        elif key in (ord("b"), ord("B")):
            current_index = max(1, current_index - 1)
            continue
        elif key in (ord("q"), ord("Q"), 27):
            print("Review paused. Every recorded decision is saved.", flush=True)
            break
        else:
            print("Unknown key. Use 1, 2, 3, 0, B, or Q.", flush=True)
            continue

        if progress["pending_count"] == 0:
            print("All 90 images have been reviewed. Closing the image window.", flush=True)
            break
        pending = [
            item["audit_index"]
            for item in progress["records"]
            if not item.get("user_decision")
        ]
        later = [index for index in pending if index > current_index]
        current_index = later[0] if later else pending[0]

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
