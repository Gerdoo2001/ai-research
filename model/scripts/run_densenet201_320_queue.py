"""Queue a 320px DenseNet-201 VIRIDIS -> GE -> CLAHE tuning run.

The controller waits for the active CLAHE multi-seed queue to release the GPU,
benchmarks the largest safe batch from 8, 6, and 4, then runs one seed-42
training/validation experiment. It never evaluates the locked test manifest.
"""

from __future__ import annotations

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
CONFIG_PATH = (
    PROJECT_ROOT
    / "model"
    / "configs"
    / "fine_tuning"
    / "densenet201_viridis_ge_clahe_320_s42.json"
)
BLOCKING_PROGRESS = (
    PROJECT_ROOT
    / "model"
    / "results"
    / "densenet201_clahe_multiseed"
    / "progress.json"
)
OUTPUT_ROOT = PROJECT_ROOT / "model" / "results" / "densenet201_320_tuning"
PROGRESS_PATH = OUTPUT_ROOT / "progress.json"
RESULT_PATH = OUTPUT_ROOT / "result.json"
RUNS_ROOT = PROJECT_ROOT / "model" / "results" / "runs"


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


def pid_exists(pid: Any) -> bool:
    try:
        pid_value = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_value <= 0:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid_value)
    except ImportError:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid_value}", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid_value) in result.stdout


def report(status: str, **extra: Any) -> None:
    payload = {
        "updated_at": now_iso(),
        "status": status,
        "controller_pid": os.getpid(),
        "architecture": "densenet201",
        "preprocessing": "pseudo_color_then_histogram_equalization_then_clahe",
        "pipeline_short_name": "VIRIDIS -> GE -> CLAHE",
        "input_size": 320,
        "seed": 42,
        "locked_test_accessed": False,
        **extra,
    }
    write_json(PROGRESS_PATH, payload)


def wait_for_gpu_queue() -> None:
    while True:
        blocker = read_json(BLOCKING_PROGRESS)
        blocker_status = blocker.get("status")
        live_pids = [
            pid
            for pid in (blocker.get("controller_pid"), blocker.get("trainer_pid"))
            if pid_exists(pid)
        ]
        if blocker_status not in {"starting", "running"} or not live_pids:
            return
        report(
            "queued_waiting_for_gpu",
            blocking_queue_status=blocker_status,
            blocking_seed=blocker.get("current_seed"),
            blocking_run_id=blocker.get("current_run_id"),
            blocking_pids=live_pids,
        )
        time.sleep(15)


def benchmark_batch_size() -> tuple[int, list[dict[str, Any]]]:
    sys.path.insert(0, str(PROJECT_ROOT / "model" / "src"))
    import torch
    import torch.nn.functional as functional

    from multiclass.config import load_config
    from multiclass.models import build_model

    config = load_config(CONFIG_PATH)
    device = torch.device("cuda")
    attempts: list[dict[str, Any]] = []
    for batch_size in (8, 6, 4):
        model = optimizer = scaler = inputs = targets = logits = loss = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            model, _ = build_model(config)
            model.to(device).train()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            inputs = torch.randn(
                batch_size, 3, config.input_size, config.input_size, device=device
            )
            targets = torch.arange(batch_size, device=device) % config.num_classes
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(inputs)
                loss = functional.cross_entropy(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            attempts.append(
                {
                    "batch_size": batch_size,
                    "status": "passed",
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                }
            )
            return batch_size, attempts
        except torch.OutOfMemoryError as exc:
            attempts.append(
                {"batch_size": batch_size, "status": "out_of_memory", "error": str(exc)}
            )
        finally:
            del model, optimizer, scaler, inputs, targets, logits, loss
            torch.cuda.empty_cache()
    raise RuntimeError("No safe 320x320 batch size was found among 8, 6, and 4")


def run_training(batch_size: int) -> dict[str, Any]:
    run_id = f"tuning_densenet201_viridis_ge_clahe_320_b{batch_size}_s42"
    run_dir = RUNS_ROOT / run_id
    command = [
        str(PYTHON_EXE),
        str(TRAIN_SCRIPT),
        "--config",
        str(CONFIG_PATH),
        "--batch-size",
        str(batch_size),
        "--run-id",
        run_id,
    ]
    if (run_dir / "last_checkpoint.pt").exists():
        command.append("--resume")

    stdout_path = OUTPUT_ROOT / "trainer.stdout.log"
    stderr_path = OUTPUT_ROOT / "trainer.stderr.log"
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
            report(
                "running",
                trainer_pid=process.pid,
                selected_batch_size=batch_size,
                run_id=run_id,
                training_status=read_json(run_dir / "status.json"),
                live_progress=read_json(run_dir / "live_progress.json"),
            )
            time.sleep(10)
    if process.returncode != 0:
        raise RuntimeError(
            f"Training exited with code {process.returncode}; see {stderr_path}"
        )
    status = read_json(run_dir / "status.json")
    if status.get("status") != "completed":
        raise RuntimeError("Trainer exited without a completed status")
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": status,
        "metrics": read_json(run_dir / "metrics.json"),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        report("queued_waiting_for_gpu")
        wait_for_gpu_queue()
        report("benchmarking_320_batch_size")
        batch_size, attempts = benchmark_batch_size()
        report(
            "starting_training",
            selected_batch_size=batch_size,
            benchmark_attempts=attempts,
        )
        result = run_training(batch_size)
        payload = {
            "updated_at": now_iso(),
            "experiment": "DenseNet-201 + VIRIDIS -> GE -> CLAHE at 320x320",
            "seed": 42,
            "selected_batch_size": batch_size,
            "locked_test_accessed": False,
            "benchmark_attempts": attempts,
            **result,
        }
        write_json(RESULT_PATH, payload)
        report(
            "completed",
            selected_batch_size=batch_size,
            run_id=result["run_id"],
            result_path=str(RESULT_PATH),
        )
        print(json.dumps(payload, indent=2), flush=True)
    except Exception as exc:
        report("failed", failure_reason=str(exc))
        print(f"ERROR: {exc}", flush=True)
        raise


if __name__ == "__main__":
    main()
