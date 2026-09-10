"""Run the two ordered HE/CLAHE DenseNet-201 experiments in sequence."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "model" / "results" / "runs"
TRAIN_SCRIPT = PROJECT_ROOT / "model" / "scripts" / "train_multiclass.py"

EXPERIMENTS = (
    (
        "ablation_densenet201_histogram_equalization_then_clahe_s42",
        PROJECT_ROOT
        / "model"
        / "configs"
        / "densenet201_histogram_equalization_then_clahe.json",
    ),
    (
        "ablation_densenet201_clahe_then_histogram_equalization_s42",
        PROJECT_ROOT
        / "model"
        / "configs"
        / "densenet201_clahe_then_histogram_equalization.json",
    ),
)


def read_status(run_id: str) -> dict[str, object] | None:
    status_path = RESULTS_ROOT / run_id / "status.json"
    if not status_path.exists():
        return None
    with status_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_experiment(run_id: str, config_path: Path) -> None:
    status = read_status(run_id)
    if status and status.get("status") == "completed":
        print(f"{run_id}: already completed; skipping.", flush=True)
        return
    if status:
        raise RuntimeError(
            f"Refusing to overwrite {run_id} with status={status.get('status')}: "
            f"{status.get('failure_reason')}"
        )

    command = [sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path)]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{run_id} exited with code {completed.returncode}")

    final_status = read_status(run_id) or {}
    if final_status.get("status") != "completed":
        raise RuntimeError(f"{run_id} did not complete: {final_status}")


def main() -> int:
    for run_id, config_path in EXPERIMENTS:
        run_experiment(run_id, config_path)
    print("Ordered HE/CLAHE comparison completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
