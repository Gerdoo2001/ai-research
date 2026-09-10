"""Prepare and persist an image-by-image manual review of PSPNet ROI masks."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = REPO_ROOT / "model" / "results" / "roi_pspnet_audit"
SUMMARY_PATH = AUDIT_ROOT / "summary.json"
REVIEW_ROOT = AUDIT_ROOT / "manual_review"
IMAGE_ROOT = REVIEW_ROOT / "images"
PROGRESS_PATH = REVIEW_ROOT / "progress.json"


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


def center_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def source_path(relative_path: str) -> Path:
    return REPO_ROOT / "model" / "datasets" / "chest_xray" / relative_path


def build_pair(record: dict, target: Path) -> None:
    with Image.open(source_path(record["relative_path"])) as source:
        original = center_square(source.convert("L")).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    with Image.open(record["overlay_path"]) as overlay_source:
        overlay = overlay_source.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    header_height = 66
    canvas = Image.new("RGB", (1024, 512 + header_height), "black")
    canvas.paste(original, (0, header_height))
    canvas.paste(overlay, (512, header_height))
    draw = ImageDraw.Draw(canvas)
    status = "AUTO PASS" if record["automated_pass"] else "AUTO FLAGGED"
    flags = ", ".join(record["quality_flags"]) if record["quality_flags"] else "none"
    draw.text((12, 8), f"Image {record['audit_index']:02d}/90 | {record['class_name']} | {status}", fill="white")
    draw.text((12, 30), f"File: {record['relative_path']}", fill=(210, 210, 210))
    draw.text((524, 8), "OVERLAY: red=left lung, green=right lung, yellow=proposed crop", fill="white")
    draw.text((524, 30), f"Automated flags: {flags}"[:92], fill=(255, 210, 80) if flags != "none" else (180, 255, 180))
    canvas.save(target, quality=94)


def initialize() -> dict:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    existing = json.loads(PROGRESS_PATH.read_text(encoding="utf-8")) if PROGRESS_PATH.exists() else {}
    existing_decisions = {
        int(record["audit_index"]): record
        for record in existing.get("records", [])
        if record.get("user_decision") in {"yes", "no"}
    }
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for audit_record in summary["records"]:
        index = int(audit_record["audit_index"])
        target = IMAGE_ROOT / f"{index:03d}_{audit_record['class_name'].lower()}.jpg"
        if not target.exists():
            build_pair(audit_record, target)
        prior = existing_decisions.get(index, {})
        records.append(
            {
                "audit_index": index,
                "class_name": audit_record["class_name"],
                "relative_path": audit_record["relative_path"],
                "automated_pass": bool(audit_record["automated_pass"]),
                "quality_flags": audit_record["quality_flags"],
                "left_fraction": audit_record["left_fraction"],
                "right_fraction": audit_record["right_fraction"],
                "union_fraction": audit_record["union_fraction"],
                "bbox": audit_record["bbox"],
                "review_image_path": str(target),
                "user_decision": prior.get("user_decision"),
                "reviewed_at": prior.get("reviewed_at"),
                "review_note": prior.get("review_note"),
            }
        )
    payload = summarize(records)
    payload.update(
        schema_version=1,
        experiment_id="FT-002-ROI-MANUAL-REVIEW",
        initialized_at=existing.get("initialized_at", now()),
        updated_at=now(),
        decision_meaning={
            "yes": "lung masks and proposed crop are visually acceptable",
            "no": "one or both lung masks or the proposed crop are visually unacceptable",
        },
        records=records,
    )
    atomic_json(PROGRESS_PATH, payload)
    return payload


def summarize(records: list[dict]) -> dict:
    yes_count = sum(record.get("user_decision") == "yes" for record in records)
    no_count = sum(record.get("user_decision") == "no" for record in records)
    pending = [record["audit_index"] for record in records if record.get("user_decision") not in {"yes", "no"}]
    return {
        "status": "completed" if not pending else "in_progress",
        "total_images": len(records),
        "reviewed_images": yes_count + no_count,
        "yes_count": yes_count,
        "no_count": no_count,
        "pending_count": len(pending),
        "next_pending_index": pending[0] if pending else None,
        "manual_acceptance_rate_reviewed": yes_count / (yes_count + no_count) if yes_count + no_count else None,
    }


def record_decision(index: int, decision: str, note: str | None) -> dict:
    payload = initialize()
    matching = [record for record in payload["records"] if record["audit_index"] == index]
    if not matching:
        raise ValueError(f"unknown audit index: {index}")
    record = matching[0]
    record["user_decision"] = decision
    record["reviewed_at"] = now()
    record["review_note"] = note
    payload.update(summarize(payload["records"]))
    payload["updated_at"] = now()
    atomic_json(PROGRESS_PATH, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=int, help="Audit image index to record")
    parser.add_argument("--decision", choices=("yes", "no"))
    parser.add_argument("--note")
    args = parser.parse_args()
    if (args.record is None) != (args.decision is None):
        parser.error("--record and --decision must be used together")
    payload = record_decision(args.record, args.decision, args.note) if args.record else initialize()
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
