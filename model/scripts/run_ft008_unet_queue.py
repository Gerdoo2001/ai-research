"""Persistent FT-008A2 queue for the corrected bilateral U-Net ROI audit."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_PYTHON = REPO_ROOT / "venv" / "Scripts" / "python.exe"
FT007_STATUS = REPO_ROOT / "model/results/runs/fine_tuning_densenet201_viridis_ge_clahe_label_smoothing005_s42/status.json"
OUTPUT_ROOT = REPO_ROOT / "model/results/ft008a2_unet_queue"
PROGRESS_PATH = OUTPUT_ROOT / "progress.json"
STDOUT_PATH = OUTPUT_ROOT / "ft008_audit.stdout.log"
STDERR_PATH = OUTPUT_ROOT / "ft008_audit.stderr.log"
AUDIT_PROGRESS = REPO_ROOT / "model/results/roi_unet_screening_two_components/progress.json"
AUDIT_SUMMARY = REPO_ROOT / "model/results/roi_unet_screening_two_components/summary.json"
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    for _ in range(8):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (PermissionError, json.JSONDecodeError):
            time.sleep(0.15)
    return {}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for attempt in range(12):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(0.2 * (attempt + 1))


def write_state(state: dict, **updates: object) -> None:
    state.update(updates, updated_at=now())
    atomic_json(PROGRESS_PATH, state)


def initial_state() -> dict:
    existing = read_json(PROGRESS_PATH)
    if existing:
        return existing
    created = now()
    return {
        "schema_version": 1,
        "queue_id": "FT-008A2-UNET-ROI-TWO-COMPONENT-AUDIT",
        "status": "waiting_for_ft007",
        "created_at": created,
        "updated_at": created,
        "dependency": {
            "experiment": "FT-007-LS005-S42",
            "status_path": str(FT007_STATUS),
            "required_terminal_status": "completed",
        },
        "stage": {
            "experiment": "FT-008A2-UNET-ROI-TWO-COMPONENT-SCREENING",
            "description": "corrected bilateral comparison of two frozen U-Net artifacts on the FT-002C cohort",
            "threshold": 0.5,
            "margin_fraction": 0.18,
            "max_components": 2,
            "classifier_training_authorized": False,
            "manual_review_required": True,
        },
        "locked_test_accessed": False,
    }


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    state = initial_state()
    summary = read_json(AUDIT_SUMMARY)
    if summary.get("status") == "awaiting_manual_review":
        write_state(state, status="awaiting_manual_review", audit_summary=summary)
        return 0
    write_state(state, status="waiting_for_ft007")
    while True:
        dependency = read_json(FT007_STATUS)
        write_state(state, dependency_snapshot=dependency, audit_snapshot=read_json(AUDIT_PROGRESS))
        dependency_status = dependency.get("status")
        if dependency_status == "completed":
            break
        if dependency_status in {"failed", "cancelled"}:
            write_state(
                state,
                status="blocked_by_ft007",
                failure_reason=f"FT-007 ended with status {dependency_status}; FT-008 was not started",
            )
            return 2
        time.sleep(5)

    write_state(state, status="gpu_release_cooldown", dependency_snapshot=read_json(FT007_STATUS))
    time.sleep(20)
    if read_json(FT007_STATUS).get("status") != "completed":
        write_state(state, status="blocked_by_ft007", failure_reason="FT-007 completion changed during cooldown")
        return 2
    command = [
        str(LEGACY_PYTHON),
        str(REPO_ROOT / "model/scripts/audit_unet_lung_roi.py"),
        "--source-summary",
        "model/results/roi_pspnet_independent_audit/summary.json",
        "--output-dir",
        "model/results/roi_unet_screening_two_components",
        "--experiment-id",
        "FT-008A2-UNET-ROI-TWO-COMPONENT-SCREENING",
        "--threshold",
        "0.5",
        "--margin-fraction",
        "0.18",
        "--max-components",
        "2",
    ]
    write_state(
        state,
        status="running_unet_audit",
        command=command,
        stdout_log=str(STDOUT_PATH),
        stderr_log=str(STDERR_PATH),
        launched_at=now(),
    )
    with STDOUT_PATH.open("a", encoding="utf-8") as stdout, STDERR_PATH.open("a", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=stdout,
            stderr=stderr,
            creationflags=CREATE_NO_WINDOW,
        )
        while process.poll() is None:
            write_state(
                state,
                status="running_unet_audit",
                audit_process_id=process.pid,
                audit_snapshot=read_json(AUDIT_PROGRESS),
            )
            time.sleep(3)
        return_code = int(process.returncode)
    summary = read_json(AUDIT_SUMMARY)
    if return_code != 0:
        write_state(
            state,
            status="failed",
            return_code=return_code,
            failure_reason=f"FT-008 audit exited with code {return_code}; inspect {STDERR_PATH}",
            audit_snapshot=read_json(AUDIT_PROGRESS),
        )
        return return_code
    if summary.get("status") != "awaiting_manual_review":
        write_state(state, status="failed", failure_reason="FT-008 ended without a manual-review summary")
        return 3
    write_state(
        state,
        status="awaiting_manual_review",
        completed_at=now(),
        return_code=0,
        audit_summary=summary,
        next_step="manual image-by-image comparison; no classifier training is authorized",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
