"""Attach to an already-running 320px trainer and recover its final result file."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "tuning_densenet201_viridis_ge_clahe_320_b8_s42"
RUN_DIR = PROJECT_ROOT / "model" / "results" / "runs" / RUN_ID
OUTPUT_ROOT = PROJECT_ROOT / "model" / "results" / "densenet201_320_tuning"
RECOVERY_PATH = OUTPUT_ROOT / "recovery_status.json"
RESULT_PATH = OUTPUT_ROOT / "result.json"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_json_with_retry(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(20):
        temporary = path.with_name(f"{path.name}.{os.getpid()}.{attempt}.tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary.replace(path)
            return
        except PermissionError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            time.sleep(0.25)
    raise PermissionError(f"Could not publish {path} after retries")


def main() -> None:
    while True:
        status = read_json(RUN_DIR / "status.json")
        state = status.get("status", "waiting_for_status")
        write_json_with_retry(
            RECOVERY_PATH,
            {
                "updated_at": now_iso(),
                "status": "attached" if state == "running" else state,
                "run_id": RUN_ID,
                "trainer_status": status,
                "live_progress": read_json(RUN_DIR / "live_progress.json"),
                "locked_test_accessed": False,
            },
        )
        if state == "completed":
            metrics = read_json(RUN_DIR / "metrics.json")
            payload = {
                "updated_at": now_iso(),
                "experiment": "DenseNet-201 + VIRIDIS -> GE -> CLAHE at 320x320",
                "seed": 42,
                "selected_batch_size": 8,
                "locked_test_accessed": False,
                "run_id": RUN_ID,
                "run_dir": str(RUN_DIR),
                "status": status,
                "metrics": metrics,
                "recovered_after_controller_detach": True,
            }
            write_json_with_retry(RESULT_PATH, payload)
            write_json_with_retry(
                RECOVERY_PATH,
                {
                    "updated_at": now_iso(),
                    "status": "completed",
                    "run_id": RUN_ID,
                    "result_path": str(RESULT_PATH),
                    "locked_test_accessed": False,
                },
            )
            return
        if state == "failed":
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
