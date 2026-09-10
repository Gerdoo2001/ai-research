"""Prepare a persistent manual review for the independent PSPNet ROI audit."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = REPO_ROOT / "model" / "results" / "roi_pspnet_independent_audit"
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
    header_height = 72
    canvas = Image.new("RGB", (1024, 512 + header_height), "black")
    canvas.paste(original, (0, header_height))
    canvas.paste(overlay, (512, header_height))
    draw = ImageDraw.Draw(canvas)
    status = "AUTO PASS" if record["automated_pass"] else "AUTO FLAGGED"
    flags = ", ".join(record["quality_flags"]) if record["quality_flags"] else "none"
    draw.text((12, 8), f"INDEPENDENT AUDIT {record['audit_index']:02d}/90 | {record['class_name']} | {status}", fill="white")
    draw.text((12, 34), f"File: {record['relative_path']}", fill=(210, 210, 210))
    draw.text((524, 8), "FROZEN POLICY: threshold=0.20, margin=18%", fill=(150, 255, 150))
    draw.text((524, 31), "Red=left lung, green=right lung, yellow=proposed crop", fill="white")
    draw.text((524, 52), f"Automated flags: {flags}"[:94], fill=(255, 210, 80) if flags != "none" else (180, 255, 180))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=94)


def summarize(records: list[dict]) -> dict:
    yes_count = sum(record.get("user_decision") == "yes" for record in records)
    no_count = sum(record.get("user_decision") == "no" for record in records)
    pending = [record["audit_index"] for record in records if record.get("user_decision") not in {"yes", "no"}]
    by_class = {}
    for class_name in ("Normal", "Viral", "Bacterial"):
        subset = [record for record in records if record["class_name"] == class_name]
        class_yes = sum(record.get("user_decision") == "yes" for record in subset)
        class_no = sum(record.get("user_decision") == "no" for record in subset)
        by_class[class_name] = {
            "total": len(subset),
            "reviewed": class_yes + class_no,
            "yes": class_yes,
            "no": class_no,
            "acceptance_rate_reviewed": class_yes / (class_yes + class_no) if class_yes + class_no else None,
        }
    return {
        "status": "completed" if records and not pending else "in_progress",
        "total_images": len(records),
        "reviewed_images": yes_count + no_count,
        "yes_count": yes_count,
        "no_count": no_count,
        "pending_count": len(pending),
        "next_pending_index": pending[0] if pending else None,
        "manual_acceptance_rate_reviewed": yes_count / (yes_count + no_count) if yes_count + no_count else None,
        "class_summary": by_class,
    }


def initialize() -> dict:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if summary.get("selected_overlap_count") != 0:
        raise RuntimeError("independent audit overlaps the original audit")
    if float(summary["threshold"]) != 0.20 or float(summary["margin_fraction"]) != 0.18:
        raise RuntimeError("independent audit does not use the frozen threshold 0.20 / margin 18% policy")
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
        experiment_id="FT-002C-ROI-INDEPENDENT-MANUAL-REVIEW",
        initialized_at=existing.get("initialized_at", now()),
        updated_at=now(),
        locked_test_accessed=False,
        source_split="new disjoint validation-only audit",
        frozen_policy={"threshold": 0.20, "margin_fraction": 0.18},
        selected_overlap_count=summary["selected_overlap_count"],
        decision_meaning={
            "yes": "both lung masks and the frozen proposed crop are anatomically acceptable",
            "no": "one or both masks or the frozen crop are anatomically unacceptable",
        },
        records=records,
    )
    atomic_json(PROGRESS_PATH, payload)
    return payload


def record_decision(index: int, decision: str, note: str | None) -> dict:
    payload = initialize()
    matches = [record for record in payload["records"] if record["audit_index"] == index]
    if not matches:
        raise ValueError(f"unknown audit index: {index}")
    matches[0]["user_decision"] = decision
    matches[0]["reviewed_at"] = now()
    matches[0]["review_note"] = note
    payload.update(summarize(payload["records"]))
    payload["updated_at"] = now()
    atomic_json(PROGRESS_PATH, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=int)
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
