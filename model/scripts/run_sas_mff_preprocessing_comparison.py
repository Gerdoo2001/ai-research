"""Run the matched DenseNet-201 SAS/MFF preprocessing experiments in sequence.

The runner is restart-safe at the experiment level: it waits for an already-running
experiment, skips an experiment that completed, and stops if an experiment failed.
It intentionally does not resume a failed run automatically because that decision
should be made after inspecting the recorded failure reason.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "model" / "results" / "runs"
TRAIN_SCRIPT = PROJECT_ROOT / "model" / "scripts" / "train_multiclass.py"

EXPERIMENTS = (
    (
        "experimental_densenet201_sas_mff_histogram_equalization_s42",
        PROJECT_ROOT
        / "model"
        / "configs"
        / "densenet201_sas_mff_histogram_equalization.json",
    ),
    (
        "experimental_densenet201_sas_mff_clahe_s42",
        PROJECT_ROOT / "model" / "configs" / "densenet201_sas_mff_clahe.json",
    ),
)


def read_status(run_id: str) -> dict[str, object] | None:
    status_path = RESULTS_ROOT / run_id / "status.json"
    if not status_path.exists():
        return None
    with status_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def wait_for_existing_run(run_id: str) -> str:
    while True:
        status = read_status(run_id)
        if status is None:
            return "missing"
        state = str(status.get("status", "unknown"))
        epoch = status.get("current_epoch", 0)
        total = status.get("total_epochs", "?")
        best_f1 = status.get("best_validation_macro_f1")
        print(
            f"{run_id}: status={state}, epoch={epoch}/{total}, "
            f"best_validation_macro_f1={best_f1}",
            flush=True,
        )
        if state != "running":
            return state
        time.sleep(60)


def run_experiment(run_id: str, config_path: Path) -> None:
    state = wait_for_existing_run(run_id)
    if state == "completed":
        return
    if state != "missing":
        status = read_status(run_id) or {}
        raise RuntimeError(
            f"Refusing to continue after {run_id} ended with status={state}: "
            f"{status.get('failure_reason')}"
        )

    command = [sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path)]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{run_id} exited with code {completed.returncode}")

    final_status = read_status(run_id) or {}
    if final_status.get("status") != "completed":
        raise RuntimeError(
            f"{run_id} did not record completed status: {final_status}"
        )


def main() -> int:
    for run_id, config_path in EXPERIMENTS:
        run_experiment(run_id, config_path)
    print("DenseNet-201 SAS/MFF preprocessing comparison completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
