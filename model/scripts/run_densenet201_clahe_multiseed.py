"""Run DenseNet-201 + CLAHE at seeds 123 and 2026 and aggregate with seed 42.

Only the locked training and validation manifests are used. The locked test set
is not evaluated by this queue.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
TRAIN_SCRIPT = PROJECT_ROOT / "model" / "scripts" / "train_multiclass.py"
RUNS_ROOT = PROJECT_ROOT / "model" / "results" / "runs"
OUTPUT_ROOT = PROJECT_ROOT / "model" / "results" / "densenet201_clahe_multiseed"
PROGRESS_PATH = OUTPUT_ROOT / "progress.json"
SUMMARY_JSON = OUTPUT_ROOT / "summary.json"
SUMMARY_CSV = OUTPUT_ROOT / "summary.csv"

RUNS = [
    {
        "seed": 42,
        "run_id": "ablation_densenet201_clahe_s42",
        "config": None,
        "reference": True,
    },
    {
        "seed": 123,
        "run_id": "robustness_densenet201_clahe_s123",
        "config": PROJECT_ROOT
        / "model"
        / "configs"
        / "clahe_multiseed"
        / "densenet201_clahe_s123.json",
        "reference": False,
    },
    {
        "seed": 2026,
        "run_id": "robustness_densenet201_clahe_s2026",
        "config": PROJECT_ROOT
        / "model"
        / "configs"
        / "clahe_multiseed"
        / "densenet201_clahe_s2026.json",
        "reference": False,
    },
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_row(item: dict[str, Any]) -> dict[str, Any]:
    run_dir = RUNS_ROOT / item["run_id"]
    status = read_json(run_dir / "status.json")
    metrics = read_json(run_dir / "metrics.json")
    viral = metrics.get("per_class", {}).get("Viral", {})
    return {
        "seed": item["seed"],
        "run_id": item["run_id"],
        "status": status.get("status", "pending"),
        "best_epoch": metrics.get("best_epoch", status.get("best_epoch")),
        "macro_f1": metrics.get("macro_f1"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "accuracy": metrics.get("accuracy"),
        "viral_recall": viral.get("recall"),
        "training_duration_seconds": metrics.get("training_duration_seconds"),
    }


def save_summary() -> dict[str, Any]:
    rows = [run_row(item) for item in RUNS]
    complete = [
        row
        for row in rows
        if row["status"] == "completed" and row["macro_f1"] is not None
    ]

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in complete if row[key] is not None]

    macro_f1 = values("macro_f1")
    payload = {
        "updated_at": now_iso(),
        "architecture": "densenet201",
        "preprocessing": "clahe",
        "target_seeds": [42, 123, 2026],
        "selection_split": "validation",
        "locked_test_accessed": False,
        "completed_seed_count": len(complete),
        "mean_macro_f1": statistics.mean(macro_f1) if macro_f1 else None,
        "sample_std_macro_f1": statistics.stdev(macro_f1)
        if len(macro_f1) > 1
        else None,
        "mean_balanced_accuracy": statistics.mean(values("balanced_accuracy"))
        if values("balanced_accuracy")
        else None,
        "mean_accuracy": statistics.mean(values("accuracy"))
        if values("accuracy")
        else None,
        "mean_viral_recall": statistics.mean(values("viral_recall"))
        if values("viral_recall")
        else None,
        "runs": rows,
    }
    write_json(SUMMARY_JSON, payload)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return payload


def update_progress(
    *, status: str, current: dict[str, Any] | None = None, trainer_pid: int | None = None
) -> None:
    payload: dict[str, Any] = {
        "updated_at": now_iso(),
        "status": status,
        "controller_pid": os.getpid(),
        "trainer_pid": trainer_pid,
        "architecture": "densenet201",
        "preprocessing": "clahe",
        "target_seeds": [42, 123, 2026],
        "locked_test_accessed": False,
    }
    if current is not None:
        run_dir = RUNS_ROOT / current["run_id"]
        payload.update(
            {
                "current_seed": current["seed"],
                "current_run_id": current["run_id"],
                "training_status": read_json(run_dir / "status.json"),
                "live_progress": read_json(run_dir / "live_progress.json"),
            }
        )
    write_json(PROGRESS_PATH, payload)
    save_summary()


def train(item: dict[str, Any]) -> None:
    run_dir = RUNS_ROOT / item["run_id"]
    if read_json(run_dir / "status.json").get("status") == "completed":
        print(f"Seed {item['seed']} is already completed; skipping.", flush=True)
        return

    stdout_path = OUTPUT_ROOT / f"seed_{item['seed']}.stdout.log"
    stderr_path = OUTPUT_ROOT / f"seed_{item['seed']}.stderr.log"
    command = [str(PYTHON_EXE), str(TRAIN_SCRIPT), "--config", str(item["config"])]
    if (run_dir / "last_checkpoint.pt").exists():
        command.append("--resume")

    print(f"Starting seed {item['seed']}: {' '.join(command)}", flush=True)
    with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        while process.poll() is None:
            update_progress(status="running", current=item, trainer_pid=process.pid)
            time.sleep(10)

    update_progress(status="running", current=item, trainer_pid=None)
    if process.returncode != 0:
        raise RuntimeError(
            f"Training failed for seed {item['seed']} with exit code {process.returncode}; "
            f"see {stderr_path}"
        )
    if read_json(run_dir / "status.json").get("status") != "completed":
        raise RuntimeError(f"Seed {item['seed']} exited without completed status")
    print(f"Seed {item['seed']} completed.", flush=True)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    reference_status = read_json(
        RUNS_ROOT / RUNS[0]["run_id"] / "status.json"
    ).get("status")
    if reference_status != "completed":
        raise RuntimeError("The seed-42 CLAHE reference run is not completed")

    try:
        update_progress(status="starting")
        for item in RUNS[1:]:
            train(item)
        summary = save_summary()
        if summary["completed_seed_count"] != 3:
            raise RuntimeError("Queue ended without all three completed seeds")
        update_progress(status="completed")
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as exc:
        update_progress(status="failed")
        print(f"ERROR: {exc}", flush=True)
        raise


if __name__ == "__main__":
    main()
