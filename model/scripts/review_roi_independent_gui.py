"""Keyboard GUI for the frozen-policy independent 90-image ROI audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_roi_independent_review import PROGRESS_PATH, initialize, record_decision  # noqa: E402


WINDOW_NAME = "Independent ROI Audit - Y=Accept N=Reject B=Back Q=Pause"


def load_progress() -> dict:
    initialize()
    return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))


def pending_or_last(progress: dict) -> int:
    pending = [record["audit_index"] for record in progress["records"] if record.get("user_decision") not in {"yes", "no"}]
    return pending[0] if pending else progress["records"][-1]["audit_index"]


def render(record: dict, progress: dict) -> np.ndarray:
    image = cv2.imread(record["review_image_path"], cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(record["review_image_path"])
    width = min(1500, image.shape[1])
    height = int(round(image.shape[0] * width / image.shape[1]))
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    footer_height = 116
    canvas = np.zeros((height + footer_height, width, 3), dtype=np.uint8)
    canvas[:height] = image
    decision = record.get("user_decision") or "PENDING"
    flags = ", ".join(record.get("quality_flags") or []) or "none"
    line1 = (
        f"NEW IMAGE {record['audit_index']}/90 | {record['class_name']} | "
        f"Automatic: {'PASS' if record['automated_pass'] else 'FLAGGED'} | Saved: {decision.upper()}"
    )
    line2 = "Frozen policy: PSPNet threshold 0.20 + 18% crop margin | Red/green=lungs | Yellow=crop"
    line3 = f"Flags: {flags}"
    line4 = (
        f"Reviewed {progress['reviewed_images']}/90 | YES {progress['yes_count']} | NO {progress['no_count']} | "
        "Y=acceptable  N=unacceptable  B=previous  Q/Esc=pause"
    )
    cv2.putText(canvas, line1, (12, height + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, line2, (12, height + 49), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (120, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(canvas, line3[:150], (12, height + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, line4, (12, height + 101), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (180, 220, 255), 1, cv2.LINE_AA)
    return canvas


def print_details(record: dict, progress: dict) -> None:
    print("\n" + "=" * 110, flush=True)
    print(f"INDEPENDENT ROI AUDIT IMAGE {record['audit_index']}/90", flush=True)
    print(f"Class:              {record['class_name']}", flush=True)
    print(f"File:               {record['relative_path']}", flush=True)
    print("Frozen policy:      PSPNet probability threshold 0.20; crop margin 18%", flush=True)
    print(f"Automated result:   {'PASS' if record['automated_pass'] else 'FLAGGED'}", flush=True)
    print(f"Automated flags:    {', '.join(record['quality_flags']) if record['quality_flags'] else 'none'}", flush=True)
    print(
        f"Mask fractions:     left={100*record['left_fraction']:.2f}%  "
        f"right={100*record['right_fraction']:.2f}%  union={100*record['union_fraction']:.2f}%",
        flush=True,
    )
    print(f"Proposed crop:      {record['bbox']}", flush=True)
    print(f"Saved decision:     {(record.get('user_decision') or 'pending').upper()}", flush=True)
    print(
        f"Running totals:     reviewed={progress['reviewed_images']}/90  "
        f"YES={progress['yes_count']}  NO={progress['no_count']}  pending={progress['pending_count']}",
        flush=True,
    )
    print("Controls: Y=acceptable, N=unacceptable, B=previous, Q or Esc=pause", flush=True)


def main() -> int:
    progress = load_progress()
    current_index = pending_or_last(progress)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1400, 880)
    while True:
        progress = load_progress()
        records_by_index = {record["audit_index"]: record for record in progress["records"]}
        record = records_by_index[current_index]
        print_details(record, progress)
        cv2.imshow(WINDOW_NAME, render(record, progress))
        key = cv2.waitKeyEx(0)
        if key in (ord("y"), ord("Y")):
            progress = record_decision(current_index, "yes", "independent frozen-policy keyboard review")
            print(f"RECORDED: Image {current_index} = YES", flush=True)
        elif key in (ord("n"), ord("N")):
            progress = record_decision(current_index, "no", "independent frozen-policy keyboard review")
            print(f"RECORDED: Image {current_index} = NO", flush=True)
        elif key in (ord("b"), ord("B")):
            current_index = max(1, current_index - 1)
            continue
        elif key in (ord("q"), ord("Q"), 27):
            print("Independent review paused. Every completed decision is saved.", flush=True)
            break
        else:
            print("Unknown key. Use Y, N, B, or Q.", flush=True)
            continue

        if progress["pending_count"] == 0:
            print("All 90 independent images have been reviewed. Closing the image window.", flush=True)
            break
        pending = [item["audit_index"] for item in progress["records"] if item.get("user_decision") not in {"yes", "no"}]
        later = [index for index in pending if index > current_index]
        current_index = later[0] if later else pending[0]
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
