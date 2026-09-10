"""Run and visibly monitor DenseNet-201 + VIRIDIS -> GE -> CLAHE at new seeds.

This confirmation queue deliberately uses only the locked training and validation
manifests. It does not call the locked-test evaluation command.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
TRAIN_SCRIPT = PROJECT_ROOT / "model" / "scripts" / "train_multiclass.py"
RUNS_ROOT = PROJECT_ROOT / "model" / "results" / "runs"
OUTPUT_ROOT = PROJECT_ROOT / "model" / "results" / "winner_multiseed_confirmation"
PROGRESS_PATH = OUTPUT_ROOT / "progress.json"
SUMMARY_JSON = OUTPUT_ROOT / "summary.json"
SUMMARY_CSV = OUTPUT_ROOT / "summary.csv"

REFERENCE_RUN = {
    "seed": 42,
    "run_id": "staged_densenet201_pseudo_color_then_histogram_equalization_then_clahe_s42",
}

TARGETS = [
    {
        "seed": 123,
        "run_id": "robustness_densenet201_pseudo_color_then_histogram_equalization_then_clahe_s123",
        "config": PROJECT_ROOT
        / "model"
        / "configs"
        / "winner_multiseed"
        / "densenet201_viridis_ge_clahe_s123.json",
    },
    {
        "seed": 2026,
        "run_id": "robustness_densenet201_pseudo_color_then_histogram_equalization_then_clahe_s2026",
        "config": PROJECT_ROOT
        / "model"
        / "configs"
        / "winner_multiseed"
        / "densenet201_viridis_ge_clahe_s2026.json",
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


def read_last_history(path: Path) -> dict[str, str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return rows[-1] if rows else {}
    except (OSError, csv.Error):
        return {}


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int:
    number = as_float(value)
    return int(number) if number is not None else 0


def percent(value: Any) -> str:
    number = as_float(value)
    return "Waiting for first epoch" if number is None else f"{number * 100:.3f}%"


def decimal(value: Any, places: int = 4) -> str:
    number = as_float(value)
    return "-" if number is None else f"{number:.{places}f}"


def duration(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return "-"
    seconds = max(0, int(round(number)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def progress_bar(fraction: float, width: int = 38) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = min(width, int(round(fraction * width)))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {fraction * 100:5.1f}%"


def gpu_status() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "GPU status unavailable"
    if result.returncode != 0 or not result.stdout.strip():
        return "GPU status unavailable"
    parts = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    if len(parts) != 5:
        return result.stdout.strip()
    name, usage, used, total, temperature = parts
    return f"{name} | usage {usage}% | memory {used}/{total} MiB | {temperature} C"


def last_error_line(path: Path) -> str:
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return ""
    return lines[-1] if lines else ""


def queue_rows() -> list[dict[str, Any]]:
    rows = [REFERENCE_RUN, *TARGETS]
    output = []
    for item in rows:
        run_dir = RUNS_ROOT / item["run_id"]
        status = read_json(run_dir / "status.json")
        history = read_last_history(run_dir / "history.csv")
        output.append(
            {
                "seed": item["seed"],
                "run_id": item["run_id"],
                "status": status.get("status", "pending"),
                "current_epoch": status.get("current_epoch", 0),
                "best_epoch": status.get("best_epoch") or history.get("best_epoch") or "-",
                "best_validation_macro_f1": status.get("best_validation_macro_f1")
                if as_float(status.get("best_validation_macro_f1")) is not None
                else history.get("best_validation_macro_f1"),
            }
        )
    return output


def save_summary() -> None:
    rows = queue_rows()
    complete_f1 = [
        float(row["best_validation_macro_f1"])
        for row in rows
        if row["status"] == "completed" and as_float(row["best_validation_macro_f1"]) is not None
    ]
    payload = {
        "updated_at": now_iso(),
        "architecture": "densenet201",
        "pipeline_id": "pseudo_color_then_histogram_equalization_then_clahe",
        "pipeline_short_name": "VIRIDIS -> GE -> CLAHE",
        "locked_test_accessed": False,
        "runs": rows,
        "completed_seed_count": len(complete_f1),
        "mean_validation_macro_f1": sum(complete_f1) / len(complete_f1) if complete_f1 else None,
    }
    write_json(SUMMARY_JSON, payload)
    with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(target: dict[str, Any], process_state: str, stderr_path: Path) -> None:
    run_dir = RUNS_ROOT / target["run_id"]
    status = read_json(run_dir / "status.json")
    live = read_json(run_dir / "live_progress.json")
    history = read_last_history(run_dir / "history.csv")

    total_epochs = as_int(status.get("total_epochs")) or 12
    completed_epoch = max(as_int(status.get("current_epoch")), as_int(history.get("epoch")))
    active_epoch = as_int(live.get("epoch")) or min(completed_epoch + 1, total_epochs)
    stage = str(live.get("stage", "initializing"))
    fraction = as_float(live.get("fraction")) or 0.0
    batch = as_int(live.get("batch"))
    total_batches = as_int(live.get("total_batches"))
    best_f1 = status.get("best_validation_macro_f1")
    if as_float(best_f1) is None:
        best_f1 = history.get("best_validation_macro_f1")
    best_epoch = status.get("best_epoch") or history.get("best_epoch") or "-"

    lines = [
        "DENSENET-201 WINNER - MULTI-SEED CONFIRMATION",
        "=" * 92,
        f"Updated:              {now_iso()}",
        f"GPU:                  {gpu_status()}",
        "Purpose:              Confirm that the seed-42 winner is repeatable at seeds 123 and 2026",
        "Locked test:          NOT ACCESSED (training/validation manifests only)",
        "",
        "FOCUSED CONFIGURATION",
        "-" * 92,
        "Model:                DenseNet-201",
        "Pipeline ID:          pseudo_color_then_histogram_equalization_then_clahe",
        "Short order:          VIRIDIS -> GE -> CLAHE",
        "Preprocessing order:",
        "  Step 1: VIRIDIS pseudo-coloring (VIRIDIS)",
        "  Step 2: Global Histogram Equalization (GE)",
        "  Step 3: Contrast Limited Adaptive Histogram Equalization (CLAHE)",
        "Fixed training:       ImageNet weights, 224x224, batch 8, AdamW, LR 0.0001",
        "Augmentation:         Horizontal flip p=0.5; random rotation +/-5 degrees (training only)",
        "",
        "CURRENT TRAINING RUN",
        "-" * 92,
        f"Seed:                 {target['seed']}",
        f"Run ID:               {target['run_id']}",
        f"Process state:         {process_state}",
        f"Training status:       {status.get('status', 'initializing')}",
        f"Completed epochs:      {completed_epoch}/{total_epochs}",
        f"Active epoch:          {active_epoch}/{total_epochs} ({stage})",
        f"Current epoch:         {progress_bar(fraction)}",
        f"Current batch:         {batch}/{total_batches}" if total_batches else "Current batch:         Waiting for data loader",
        f"Epoch elapsed:         {duration(live.get('elapsed_seconds'))}",
        f"Epoch time left*:      {duration(live.get('estimated_remaining_seconds'))}",
        f"Best epoch:            {best_epoch}",
        f"Best val macro F1:     {percent(best_f1)}",
        "",
        "LATEST COMPLETED EPOCH",
        "-" * 92,
        f"Train loss:            {decimal(history.get('train_loss'))}",
        f"Validation loss:       {decimal(history.get('val_loss'))}",
        f"Validation macro F1:   {percent(history.get('val_macro_f1'))}",
        f"Balanced accuracy:     {percent(history.get('val_balanced_accuracy'))}",
        f"Accuracy:              {percent(history.get('val_accuracy'))}",
        f"Viral recall:          {percent(history.get('recall_viral_or_pneumonia'))}",
        f"Learning rate:         {decimal(history.get('learning_rate'), 7)}",
        f"Epoch duration:        {duration(history.get('elapsed_seconds'))}",
        f"Run time left*:        {duration(history.get('estimated_remaining_seconds'))}",
        "",
        "SEED QUEUE",
        "-" * 92,
    ]
    for row in queue_rows():
        marker = "REFERENCE" if row["seed"] == 42 else ("CURRENT" if row["seed"] == target["seed"] else "QUEUED")
        lines.append(
            f"Seed {row['seed']:<4} | {marker:<9} | {row['status']:<10} | "
            f"epoch {row['current_epoch']}/12 | best F1 {percent(row['best_validation_macro_f1'])}"
        )
    error = last_error_line(stderr_path)
    if error:
        lines.extend(["", f"Latest trainer message: {error}"])
    lines.extend(
        [
            "",
            "*Time estimates become more reliable after the first completed epoch.",
            "Keep the laptop plugged in and prevent sleep/shutdown while training.",
        ]
    )
    os.system("cls" if os.name == "nt" else "clear")
    print("\n".join(lines), flush=True)


def run_target(target: dict[str, Any]) -> bool:
    run_dir = RUNS_ROOT / target["run_id"]
    status = read_json(run_dir / "status.json")
    if status.get("status") == "completed":
        save_summary()
        return True

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stdout_path = OUTPUT_ROOT / f"seed_{target['seed']}.stdout.log"
    stderr_path = OUTPUT_ROOT / f"seed_{target['seed']}.stderr.log"
    command = [str(PYTHON_EXE), str(TRAIN_SCRIPT), "--config", str(target["config"])]
    if (run_dir / "last_checkpoint.pt").exists():
        command.append("--resume")

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
        write_json(
            PROGRESS_PATH,
            {
                "status": "running",
                "controller_pid": os.getpid(),
                "trainer_pid": process.pid,
                "current_seed": target["seed"],
                "current_run_id": target["run_id"],
                "started_at": now_iso(),
                "locked_test_accessed": False,
            },
        )
        while process.poll() is None:
            render(target, f"RUNNING (trainer PID {process.pid})", stderr_path)
            save_summary()
            time.sleep(2)
        return_code = process.returncode

    save_summary()
    render(target, f"EXITED (code {return_code})", stderr_path)
    return return_code == 0 and read_json(run_dir / "status.json").get("status") == "completed"


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    os.system("title DenseNet-201 + VIRIDIS - GE - CLAHE Multi-Seed Confirmation")
    for target in TARGETS:
        if not run_target(target):
            write_json(
                PROGRESS_PATH,
                {
                    "status": "failed",
                    "controller_pid": os.getpid(),
                    "current_seed": target["seed"],
                    "current_run_id": target["run_id"],
                    "updated_at": now_iso(),
                    "locked_test_accessed": False,
                },
            )
            print("\nTraining stopped because the current seed did not complete successfully.", flush=True)
            return 1

    save_summary()
    write_json(
        PROGRESS_PATH,
        {
            "status": "completed",
            "controller_pid": os.getpid(),
            "completed_at": now_iso(),
            "locked_test_accessed": False,
        },
    )
    os.system("cls" if os.name == "nt" else "clear")
    print("DENSENET-201 + VIRIDIS -> GE -> CLAHE CONFIRMATION COMPLETE")
    print("=" * 76)
    for row in queue_rows():
        print(
            f"Seed {row['seed']}: {row['status']} | best epoch {row['best_epoch']} | "
            f"best validation macro F1 {percent(row['best_validation_macro_f1'])}"
        )
    print(f"\nSummary: {SUMMARY_JSON}")
    print("Locked test was not accessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
