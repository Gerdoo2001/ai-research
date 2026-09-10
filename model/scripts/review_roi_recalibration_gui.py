"""Keyboard GUI for comparing three persistent PSPNet recalibration variants."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_roi_recalibration_review import (  # noqa: E402
    PROGRESS_PATH,
    VALID_CHOICES,
    load_progress,
    record_choice,
)


WINDOW_NAME = "ROI Recalibration - 1=Mild 2=Moderate 3=Aggressive N=None B=Back Q=Pause"


def first_pending_or_last(progress: dict) -> int:
    pending = [
        index
        for index, record in enumerate(progress["records"])
        if record.get("user_choice") not in VALID_CHOICES
    ]
    return pending[0] if pending else len(progress["records"]) - 1


def render(record: dict, progress: dict) -> np.ndarray:
    image = cv2.imread(record["comparison_image_path"], cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(record["comparison_image_path"])
    max_width, max_height = 1850, 820
    scale = min(max_width / image.shape[1], max_height / image.shape[0], 1.0)
    width = int(round(image.shape[1] * scale))
    height = int(round(image.shape[0] * scale))
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    footer_height = 112
    canvas = np.zeros((height + footer_height, width, 3), dtype=np.uint8)
    canvas[:height] = image
    counts = progress["choice_counts"]
    saved = (record.get("user_choice") or "pending").upper()
    line1 = (
        f"Review {record['review_order']}/{progress['total_images']} | Original audit {record['audit_index']}/90 | "
        f"Class: {record['class_name']} | Saved choice: {saved}"
    )
    line2 = (
        f"Reviewed {progress['reviewed_images']}/{progress['total_images']} | "
        f"Mild={counts['1']} Moderate={counts['2']} Aggressive={counts['3']} None={counts['none']} | "
        f"Pending={progress['pending_count']}"
    )
    line3 = "Press 1, 2, or 3 for the best acceptable crop; N if none are acceptable; B=previous; Q/Esc=save and pause"
    line4 = "This tunes candidates only. The winning global setting still requires a new independent 90-image audit."
    cv2.putText(canvas, line1, (12, height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, line2, (12, height + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.59, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, line3, (12, height + 77), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(canvas, line4, (12, height + 102), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (180, 180, 255), 1, cv2.LINE_AA)
    return canvas


def print_details(record: dict, progress: dict) -> None:
    print("\n" + "=" * 112, flush=True)
    print(
        f"RECALIBRATION REVIEW {record['review_order']}/{progress['total_images']}  "
        f"(original audit index {record['audit_index']}/90)",
        flush=True,
    )
    print(f"Class:              {record['class_name']}", flush=True)
    print(f"File:               {record['relative_path']}", flush=True)
    for candidate in record["candidates"]:
        print(
            f"Option {candidate['choice']} {candidate['name']:<22} "
            f"threshold={candidate['threshold']:.2f}  crop margin={100*candidate['margin_fraction']:.0f}%  "
            f"left={100*candidate['left_fraction']:.2f}%  right={100*candidate['right_fraction']:.2f}%  "
            f"union={100*candidate['union_fraction']:.2f}%  bbox={candidate['bbox']}",
            flush=True,
        )
    counts = progress["choice_counts"]
    print(f"Saved choice:       {(record.get('user_choice') or 'pending').upper()}", flush=True)
    print(
        f"Running totals:     reviewed={progress['reviewed_images']}/{progress['total_images']}  "
        f"1={counts['1']}  2={counts['2']}  3={counts['3']}  NONE={counts['none']}  "
        f"pending={progress['pending_count']}",
        flush=True,
    )
    print("Controls: 1=mild, 2=moderate, 3=aggressive, N=none, B=previous, Q or Esc=pause", flush=True)


def main() -> int:
    progress = load_progress()
    if not progress.get("records"):
        raise RuntimeError(f"No recalibration records found in {PROGRESS_PATH}")
    current_position = first_pending_or_last(progress)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1800, 930)
    while True:
        progress = load_progress()
        record = progress["records"][current_position]
        print_details(record, progress)
        cv2.imshow(WINDOW_NAME, render(record, progress))
        key = cv2.waitKeyEx(0)
        choice = None
        if key in (ord("1"), 65457):
            choice = "1"
        elif key in (ord("2"), 65458):
            choice = "2"
        elif key in (ord("3"), 65459):
            choice = "3"
        elif key in (ord("n"), ord("N")):
            choice = "none"
        elif key in (ord("b"), ord("B")):
            current_position = max(0, current_position - 1)
            continue
        elif key in (ord("q"), ord("Q"), 27):
            print("Recalibration review paused. Every recorded choice is saved.", flush=True)
            break
        else:
            print("Unknown key. Use 1, 2, 3, N, B, or Q.", flush=True)
            continue

        progress = record_choice(record["audit_index"], choice, "keyboard GUI recalibration review")
        print(f"RECORDED: Original audit {record['audit_index']} = {choice.upper()}", flush=True)
        if progress["pending_count"] == 0:
            print("All 21 rejected masks have been compared. Closing the image window.", flush=True)
            break
        pending_positions = [
            index
            for index, item in enumerate(progress["records"])
            if item.get("user_choice") not in VALID_CHOICES
        ]
        later = [index for index in pending_positions if index > current_position]
        current_position = later[0] if later else pending_positions[0]
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
