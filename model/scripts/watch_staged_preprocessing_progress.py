"""Display the staged preprocessing experiment's live progress in a console."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "model" / "results" / "staged_preprocessing_experiment"
RUNS_ROOT = PROJECT_ROOT / "model" / "results" / "runs"
PROGRESS_PATH = OUTPUT_ROOT / "progress.json"
LOCK_PATH = OUTPUT_ROOT / "runner.lock.json"
RUNNER_SCRIPT = PROJECT_ROOT / "model" / "scripts" / "run_staged_preprocessing_experiment.py"
PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

_last_supervisor_launch = 0.0

MODEL_NAMES = {
    "densenet121": "DenseNet-121",
    "densenet169": "DenseNet-169",
    "densenet201": "DenseNet-201",
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    "resnet152": "ResNet-152",
    "yolov5_cls": "YOLOv5 Classification",
    "yolov8_cls": "YOLOv8 Classification",
}

PREPROCESSING_OPERATION_NAMES = {
    "clahe": "Contrast Limited Adaptive Histogram Equalization (CLAHE)",
    "histogram_equalization": "Global Histogram Equalization (GE)",
    "pseudo_color": "VIRIDIS pseudo-coloring (VIRIDIS)",
}

PREPROCESSING_ABBREVIATIONS = {
    "clahe": "CLAHE",
    "histogram_equalization": "GE",
    "pseudo_color": "VIRIDIS",
}


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def read_last_csv_row(path: Path) -> dict[str, str]:
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


def percent(value: Any) -> str:
    number = as_float(value)
    return "Waiting for first epoch" if number is None else f"{number * 100:.3f}%"


def decimal(value: Any, places: int = 4) -> str:
    number = as_float(value)
    return "-" if number is None else f"{number:.{places}f}"


def duration(value: Any) -> str:
    seconds = as_float(value)
    if seconds is None:
        return "-"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def progress_bar(fraction: float, width: int = 32) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = min(width, int(round(fraction * width)))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {fraction * 100:5.1f}%"


def seconds_since(timestamp: Any) -> float | None:
    if not timestamp:
        return None
    try:
        moment = datetime.fromisoformat(str(timestamp))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return max(0.0, (datetime.now().astimezone() - moment).total_seconds())


def preprocessing_steps(pipeline: str) -> list[str]:
    if not pipeline or pipeline == "baseline":
        return ["Base image pipeline (Base; no additional contrast enhancement)"]
    operation_ids = pipeline.split("_then_")
    return [
        PREPROCESSING_OPERATION_NAMES.get(operation_id, operation_id.replace("_", " ").title())
        for operation_id in operation_ids
    ]


def short_preprocessing_order(pipeline: str) -> str:
    if not pipeline or pipeline == "baseline":
        return "Base"
    return " -> ".join(
        PREPROCESSING_ABBREVIATIONS.get(operation_id, operation_id)
        for operation_id in pipeline.split("_then_")
    )


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, pid
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def gpu_status() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
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
    name, utilization, used, total, temperature = parts
    return f"{name} | usage {utilization}% | memory {used}/{total} MiB | {temperature} C"


def gpu_python_training_active() -> bool:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and any(
        "python" in line.lower() for line in result.stdout.splitlines()
    )


def maybe_restart_queue_controller() -> str:
    """Keep the sequential queue alive without duplicating an active GPU run."""

    global _last_supervisor_launch
    lock = read_json(LOCK_PATH)
    runner_pid = int(lock.get("pid", 0) or 0)
    if pid_is_running(runner_pid):
        return f"ACTIVE (PID {runner_pid})"
    if gpu_python_training_active():
        return "WAITING - training is active; controller restarts after this run"
    if time.monotonic() - _last_supervisor_launch < 20:
        return "STARTING"

    stdout_handle = (OUTPUT_ROOT / "runner.stdout.log").open("a", encoding="utf-8")
    stderr_handle = (OUTPUT_ROOT / "runner.stderr.log").open("a", encoding="utf-8")
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    subprocess.Popen(
        [str(PYTHON_EXE), str(RUNNER_SCRIPT), "--skip-smoke"],
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=stdout_handle,
        stderr=stderr_handle,
        close_fds=True,
        creationflags=creationflags,
    )
    stdout_handle.close()
    stderr_handle.close()
    _last_supervisor_launch = time.monotonic()
    return "STARTED BY MONITOR"


def last_nonempty_line(path: Path) -> str:
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return ""
    return lines[-1] if lines else ""


def render(*, controller_text: str | None = None) -> str:
    progress = read_json(PROGRESS_PATH)
    current = progress.get("current_target") or {}
    run_id = str(current.get("run_id", ""))
    run_dir = RUNS_ROOT / run_id if run_id else None
    status = read_json(run_dir / "status.json") if run_dir else {}
    history = read_last_csv_row(run_dir / "history.csv") if run_dir else {}
    live = read_json(run_dir / "live_progress.json") if run_dir else {}
    lock = read_json(LOCK_PATH)
    runner_pid = int(lock.get("pid", progress.get("runner_pid", 0)) or 0)
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024**3)

    phase = str(progress.get("phase", "Preparing"))
    completed = int(progress.get("phase_completed", 0) or 0)
    targets = int(progress.get("phase_targets", 0) or 0)
    remaining = int(progress.get("phase_remaining", max(0, targets - completed)) or 0)
    status_epoch = int(as_float(status.get("current_epoch")) or 0)
    history_epoch = int(as_float(history.get("epoch")) or 0)
    current_epoch = max(status_epoch, history_epoch)
    total_epochs = status.get("total_epochs", 12)
    architecture = str(current.get("architecture", ""))
    pipeline = str(current.get("preprocessing", ""))
    operation_lines = ["Preprocessing order:"]
    for index, operation in enumerate(preprocessing_steps(pipeline), start=1):
        operation_lines.append(f"  Step {index}: {operation}")

    best_epoch = status.get("best_epoch") or history.get("best_epoch") or "-"
    best_f1 = status.get("best_validation_macro_f1")
    if as_float(best_f1) is None:
        best_f1 = history.get("best_validation_macro_f1")

    active_epoch = min(current_epoch + 1, int(total_epochs))
    epoch_stage = "waiting"
    epoch_fraction = 0.0
    epoch_elapsed = None
    epoch_remaining = None
    batch_text = "-"
    progress_source = "Waiting for training to begin."
    live_is_current = live.get("run_id") == run_id and live.get("stage") in {
        "initializing", "training", "validation", "epoch_completed"
    }
    if live_is_current:
        active_epoch = int(as_float(live.get("epoch")) or active_epoch)
        epoch_stage = str(live.get("stage", "training"))
        epoch_fraction = as_float(live.get("fraction")) or 0.0
        epoch_elapsed = as_float(live.get("elapsed_seconds"))
        epoch_remaining = as_float(live.get("estimated_remaining_seconds"))
        batch = int(as_float(live.get("batch")) or 0)
        total_batches = int(as_float(live.get("total_batches")) or 0)
        batch_text = f"{batch}/{total_batches}" if total_batches else "-"
        progress_source = "Exact batch-by-batch measurement."
    elif status.get("status") == "running" and current_epoch < int(total_epochs):
        epoch_stage = "training (estimated)"
        epoch_elapsed = seconds_since(status.get("updated_at") or status.get("started_at"))
        expected_seconds = as_float(history.get("elapsed_seconds"))
        if epoch_elapsed is not None and expected_seconds and expected_seconds > 0:
            epoch_fraction = min(0.99, epoch_elapsed / expected_seconds)
            epoch_remaining = max(0.0, expected_seconds - epoch_elapsed)
        progress_source = "Time-based estimate; exact batch tracking begins with the next trainer process."

    lines = [
        "STAGED PREPROCESSING EXPERIMENT - LIVE MONITOR",
        "=" * 88,
        f"Updated:           {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Queue controller:  {controller_text or ('ACTIVE' if pid_is_running(runner_pid) else 'NOT ACTIVE')}",
        f"GPU training:      {'ACTIVE' if gpu_python_training_active() else 'NOT ACTIVE'}",
        f"Experiment state:  {progress.get('state', 'waiting')}",
        f"Current phase:     {phase}",
        f"Phase progress:    {completed}/{targets} complete; {remaining} remaining",
        "",
        "CURRENT TRAINING RUN",
        "-" * 88,
        f"Model:             {MODEL_NAMES.get(architecture, architecture or '-')}",
        f"Pipeline ID:       {pipeline or '-'}",
        f"Short order:       {short_preprocessing_order(pipeline)}",
        *operation_lines,
        f"Run ID:            {run_id or '-'}",
        f"Training status:   {status.get('status', 'waiting')}",
        f"Completed epochs:  {current_epoch}/{total_epochs}",
        f"Working on epoch:  {active_epoch}/{total_epochs}",
        f"Epoch stage:       {epoch_stage}",
        f"Epoch progress:    {progress_bar(epoch_fraction)}",
        f"Batch progress:    {batch_text}",
        f"Epoch elapsed:     {duration(epoch_elapsed)}",
        f"Epoch time left:   {duration(epoch_remaining)}",
        f"Progress source:   {progress_source}",
        f"Best epoch:        {best_epoch}",
        f"Best val macro F1: {percent(best_f1)}",
        "",
        "LATEST COMPLETED EPOCH",
        "-" * 88,
        f"Train loss:        {decimal(history.get('train_loss'))}",
        f"Validation loss:   {decimal(history.get('val_loss'))}",
        f"Validation F1:     {percent(history.get('val_macro_f1'))}",
        f"Balanced accuracy: {percent(history.get('val_balanced_accuracy'))}",
        f"Accuracy:          {percent(history.get('val_accuracy'))}",
        f"Viral recall:      {percent(history.get('recall_viral_or_pneumonia'))}",
        f"Learning rate:     {decimal(history.get('learning_rate'), 7)}",
        f"Epoch duration:    {duration(history.get('elapsed_seconds'))}",
        f"Run time left*:    {duration(history.get('estimated_remaining_seconds'))}",
        "",
        "SYSTEM",
        "-" * 88,
        f"GPU:               {gpu_status()}",
        f"Free disk space:   {free_gb:.2f} GiB",
        f"Locked test used:  {progress.get('locked_test_accessed', False)}",
        "",
        "*Estimate covers the current run only and updates after each epoch.",
        "This window is a monitor. Press Ctrl+C or close it without stopping training.",
    ]

    error = read_json(OUTPUT_ROOT / "runner_error.json")
    if error and not pid_is_running(runner_pid):
        lines.extend(("", "RUNNER ERROR", "-" * 88, str(error.get("error", "Unknown error"))))
    stderr_line = last_nonempty_line(OUTPUT_ROOT / "runner.stderr.log")
    if stderr_line and "profile.ps1" not in stderr_line:
        lines.extend(("", f"Latest runner message: {stderr_line}"))
    return "\n".join(lines)


def main() -> int:
    if "--once" in sys.argv[1:]:
        print(render(), flush=True)
        return 0
    supervise = "--supervise" in sys.argv[1:]
    try:
        while True:
            controller_text = maybe_restart_queue_controller() if supervise else None
            os.system("cls")
            print(render(controller_text=controller_text), flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nMonitor closed. Training was not stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
