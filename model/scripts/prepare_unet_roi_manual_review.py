"""Prepare and persist the corrected FT-008A2 two-model U-Net manual review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = REPO_ROOT / "model" / "results" / "roi_unet_screening_two_components"
SUMMARY_PATH = AUDIT_ROOT / "summary.json"
REVIEW_ROOT = AUDIT_ROOT / "manual_review"
PROGRESS_PATH = REVIEW_ROOT / "progress.json"
FINAL_SUMMARY_PATH = REVIEW_ROOT / "summary.json"
QUEUE_PROGRESS_PATH = REPO_ROOT / "model" / "results" / "ft008a2_unet_queue" / "progress.json"

DECISIONS = ("classic_only", "legacy_only", "both", "neither")
MODEL_KEYS = ("classic_unet_cxr_reg", "legacy_full_unet")


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


def model_is_accepted(decision: str | None, model_key: str) -> bool:
    if model_key == "classic_unet_cxr_reg":
        return decision in {"classic_only", "both"}
    if model_key == "legacy_full_unet":
        return decision in {"legacy_only", "both"}
    raise ValueError(f"unknown model key: {model_key}")


def summarize(records: list[dict]) -> dict:
    reviewed = [record for record in records if record.get("user_decision") in DECISIONS]
    pending = [record["audit_index"] for record in records if record.get("user_decision") not in DECISIONS]
    decision_counts = {decision: 0 for decision in DECISIONS}
    for record in reviewed:
        decision_counts[record["user_decision"]] += 1

    per_model = {}
    for model_key in MODEL_KEYS:
        accepted = sum(model_is_accepted(record.get("user_decision"), model_key) for record in reviewed)
        class_summary = {}
        for class_name in ("Normal", "Viral", "Bacterial"):
            subset = [record for record in reviewed if record["class_name"] == class_name]
            class_accepted = sum(model_is_accepted(record.get("user_decision"), model_key) for record in subset)
            class_summary[class_name] = {
                "reviewed": len(subset),
                "accepted": class_accepted,
                "rejected": len(subset) - class_accepted,
                "acceptance_rate_reviewed": class_accepted / len(subset) if subset else None,
            }
        per_model[model_key] = {
            "reviewed": len(reviewed),
            "accepted": accepted,
            "rejected": len(reviewed) - accepted,
            "acceptance_rate_reviewed": accepted / len(reviewed) if reviewed else None,
            "class_summary": class_summary,
        }

    return {
        "status": "completed" if records and not pending else "in_progress",
        "total_images": len(records),
        "reviewed_images": len(reviewed),
        "pending_count": len(pending),
        "next_pending_index": pending[0] if pending else None,
        "decision_counts": decision_counts,
        "per_model": per_model,
    }


def automatic_manual_agreement(records: list[dict]) -> dict:
    agreement = {}
    for model_key in MODEL_KEYS:
        counts = {
            "automatic_pass_human_accept": 0,
            "automatic_pass_human_reject": 0,
            "automatic_flag_human_accept": 0,
            "automatic_flag_human_reject": 0,
        }
        for record in records:
            decision = record.get("user_decision")
            human_accept = model_is_accepted(decision, model_key)
            automatic_pass = bool(record["models"][model_key]["automated_pass"])
            if automatic_pass and human_accept:
                counts["automatic_pass_human_accept"] += 1
            elif automatic_pass:
                counts["automatic_pass_human_reject"] += 1
            elif human_accept:
                counts["automatic_flag_human_accept"] += 1
            else:
                counts["automatic_flag_human_reject"] += 1
        agreement[model_key] = counts
    return agreement


def validate_summary(summary: dict) -> None:
    if summary.get("experiment_id") != "FT-008A2-UNET-ROI-TWO-COMPONENT-SCREENING":
        raise RuntimeError("unexpected source experiment")
    if summary.get("status") != "awaiting_manual_review":
        raise RuntimeError("FT-008A2 is not at the manual-review boundary")
    if summary.get("locked_test_accessed") is not False:
        raise RuntimeError("source audit does not verify locked-test isolation")
    if summary.get("classifier_training_started") is not False:
        raise RuntimeError("source audit unexpectedly reports classifier training")
    if int(summary.get("max_components", -1)) != 2:
        raise RuntimeError("source audit is not the corrected bilateral two-component run")
    if len(summary.get("records", [])) != 90:
        raise RuntimeError("manual review requires the complete 90-image screen")


def initialize() -> dict:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    validate_summary(summary)
    existing = json.loads(PROGRESS_PATH.read_text(encoding="utf-8")) if PROGRESS_PATH.exists() else {}
    existing_decisions = {
        int(record["audit_index"]): record
        for record in existing.get("records", [])
        if record.get("user_decision") in DECISIONS
    }

    records = []
    for audit_record in summary["records"]:
        index = int(audit_record["audit_index"])
        comparison_path = Path(audit_record["comparison_path"])
        if not comparison_path.exists():
            raise FileNotFoundError(comparison_path)
        prior = existing_decisions.get(index, {})
        model_details = {}
        for model_key in MODEL_KEYS:
            model_record = audit_record["models"][model_key]
            model_details[model_key] = {
                "automated_pass": bool(model_record["automated_pass"]),
                "quality_flags": list(model_record["quality_flags"]),
                "mask_fraction": float(model_record["mask_fraction"]),
                "bbox": model_record["bbox"],
                "raw_components": int(model_record["raw_components"]),
                "mask_path": model_record["mask_path"],
            }
        records.append(
            {
                "audit_index": index,
                "class_name": audit_record["class_name"],
                "relative_path": audit_record["relative_path"],
                "cryptographic_hash": audit_record["cryptographic_hash"],
                "review_image_path": str(comparison_path),
                "models": model_details,
                "user_decision": prior.get("user_decision"),
                "reviewed_at": prior.get("reviewed_at"),
                "review_note": prior.get("review_note"),
            }
        )

    payload = summarize(records)
    payload.update(
        schema_version=1,
        experiment_id="FT-008A2-UNET-ROI-TWO-COMPONENT-MANUAL-REVIEW",
        initialized_at=existing.get("initialized_at", now()),
        updated_at=now(),
        source_summary_path=str(SUMMARY_PATH),
        source_summary_sha256=sha256(SUMMARY_PATH),
        locked_test_accessed=False,
        classifier_training_authorized=False,
        source_split="matched validation-only screening cohort",
        decision_meaning={
            "classic_only": "Classic U-Net mask and proposed crop are acceptable; Legacy U-Net is unacceptable",
            "legacy_only": "Legacy U-Net mask and proposed crop are acceptable; Classic U-Net is unacceptable",
            "both": "both U-Net masks and proposed crops are acceptable",
            "neither": "neither U-Net mask and proposed crop is acceptable",
        },
        records=records,
    )
    atomic_json(PROGRESS_PATH, payload)
    return payload


def record_decision(index: int, decision: str, note: str | None) -> dict:
    if decision not in DECISIONS:
        raise ValueError(f"invalid decision: {decision}")
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


def finalize() -> dict:
    progress = initialize()
    if progress["status"] != "completed" or progress["pending_count"] != 0:
        raise RuntimeError("cannot finalize until all 90 images have a valid manual decision")
    indices = [int(record["audit_index"]) for record in progress["records"]]
    if sorted(indices) != list(range(1, 91)):
        raise RuntimeError("manual-review records are incomplete or duplicated")
    invalid = [record for record in progress["records"] if record.get("user_decision") not in DECISIONS]
    if invalid:
        raise RuntimeError("manual-review records contain invalid decisions")

    result = {
        "schema_version": 1,
        "experiment_id": "FT-008A2-UNET-ROI-TWO-COMPONENT-MANUAL-REVIEW",
        "status": "completed_no_advance",
        "finalized_at": now(),
        "locked_test_accessed": False,
        "classifier_training_authorized": False,
        "source_summary_path": str(SUMMARY_PATH),
        "source_summary_sha256": progress["source_summary_sha256"],
        "manual_progress_path": str(PROGRESS_PATH),
        "manual_progress_sha256": sha256(PROGRESS_PATH),
        "total_images": progress["total_images"],
        "reviewed_images": progress["reviewed_images"],
        "decision_counts": progress["decision_counts"],
        "per_model": progress["per_model"],
        "automatic_manual_agreement": automatic_manual_agreement(progress["records"]),
        "advancement_decision": {
            "classic_unet_cxr_reg": "do_not_advance",
            "legacy_full_unet": "do_not_advance",
            "reason": (
                "Neither model produced any manually acceptable Viral or Bacterial mask/crop in the matched "
                "90-image screening cohort. This severe class-dependent failure prohibits independent "
                "confirmation, full-development feasibility processing, and ROI classifier training."
            ),
        },
        "interpretation_boundary": (
            "These are manual anatomical acceptability counts on a validation-only screening cohort already "
            "used in FT-002C. They are not segmentation Dice/IoU measurements and are not locked-test results."
        ),
    }
    atomic_json(FINAL_SUMMARY_PATH, result)

    if QUEUE_PROGRESS_PATH.exists():
        queue = json.loads(QUEUE_PROGRESS_PATH.read_text(encoding="utf-8"))
        queue.update(
            status="manual_review_completed_no_advance",
            updated_at=now(),
            manual_review_summary=str(FINAL_SUMMARY_PATH),
            manual_review_snapshot={
                "reviewed_images": result["reviewed_images"],
                "decision_counts": result["decision_counts"],
                "per_model": result["per_model"],
                "classifier_training_authorized": False,
            },
        )
        atomic_json(QUEUE_PROGRESS_PATH, queue)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=int)
    parser.add_argument("--decision", choices=DECISIONS)
    parser.add_argument("--note")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize and (args.record is not None or args.decision is not None):
        parser.error("--finalize cannot be combined with --record/--decision")
    if (args.record is None) != (args.decision is None):
        parser.error("--record and --decision must be used together")
    if args.finalize:
        payload = finalize()
    else:
        payload = record_decision(args.record, args.decision, args.note) if args.record else initialize()
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
