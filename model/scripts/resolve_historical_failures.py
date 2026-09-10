"""Mark preserved historical failure rows resolved only when artifacts prove recovery."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "model" / "results"
FAILURES = RESULTS / "failures.csv"


def load_status(run_id: str) -> dict:
    path = RESULTS / "runs" / run_id / "status.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    with FAILURES.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        columns = list(rows[0]) if rows else []
    if not rows:
        raise RuntimeError("failure ledger is empty")

    resume = load_status("smoke_resume_contract")
    yolo_retry = load_status("smoke_yolov8_cls_e2e_retry2")
    if resume.get("status") != "completed" or int(resume.get("current_epoch", 0)) != 2:
        raise RuntimeError("resume contract does not have a completed two-epoch artifact")
    if yolo_retry.get("status") != "completed":
        raise RuntimeError("YOLOv8 corrected retry is not completed")

    resolutions = {
        "smoke_resume_contract": {
            "issue_id": "CONTROLLED-RESUME-CONTRACT",
            "resolution": "The interruption was intentional; the same run resumed from last_checkpoint.pt and completed epoch 2.",
            "evidence": [
                "model/results/runs/smoke_resume_contract/status.json",
                "model/results/runs/smoke_resume_contract/last_checkpoint.pt",
                "model/results/runs/smoke_resume_contract/best_checkpoint.pt",
                "model/docs/TRAINING_JOURNAL.md",
            ],
        },
        "smoke_yolov8_cls_e2e": {
            "issue_id": "ISSUE-006",
            "resolution": "A classification-logits adapter now selects raw logits from Ultralytics evaluation tuples; the corrected retry completed with artifacts and checkpoint reload verification.",
            "evidence": [
                "model/src/multiclass/models.py",
                "model/results/runs/smoke_yolov8_cls_e2e_retry2/status.json",
                "model/results/runs/smoke_yolov8_cls_e2e_retry2/best_checkpoint.pt",
                "model/docs/ISSUES_AND_FIXES.md",
            ],
        },
    }
    changed = 0
    for row in rows:
        resolution = resolutions.get(row.get("run_id", ""))
        if resolution:
            if row.get("resolved", "").lower() != "true":
                changed += 1
            row["resolved"] = "true"
            row["issue_id"] = resolution["issue_id"]

    temp = FAILURES.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, FAILURES)

    payload = {
        "schema_version": 1,
        "updated_at": datetime.now(ZoneInfo("Asia/Manila")).isoformat(timespec="seconds"),
        "ledger_rows_preserved": len(rows),
        "rows_changed_this_invocation": changed,
        "active_unresolved_count": sum(row.get("resolved", "").lower() != "true" for row in rows),
        "resolutions": resolutions,
    }
    (RESULTS / "audits" / "historical_failure_resolutions.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
