"""Confirm the two closest seed-42 competitors across seeds 123 and 2026.

The queue compares the resulting three-seed means with the already completed
DenseNet-201 candidate. It uses only locked training/validation manifests and
never invokes locked-test evaluation.
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
OUTPUT_ROOT = PROJECT_ROOT / "model" / "results" / "top_competitor_multiseed_confirmation"
PROGRESS_PATH = OUTPUT_ROOT / "progress.json"
SUMMARY_JSON = OUTPUT_ROOT / "summary.json"
SUMMARY_CSV = OUTPUT_ROOT / "summary.csv"


CANDIDATES = [
    {
        "key": "densenet201_viridis_ge_clahe",
        "model": "DenseNet-201",
        "pipeline": "VIRIDIS -> GE -> CLAHE",
        "runs": {
            42: "staged_densenet201_pseudo_color_then_histogram_equalization_then_clahe_s42",
            123: "robustness_densenet201_pseudo_color_then_histogram_equalization_then_clahe_s123",
            2026: "robustness_densenet201_pseudo_color_then_histogram_equalization_then_clahe_s2026",
        },
    },
    {
        "key": "densenet169_viridis_ge_clahe",
        "model": "DenseNet-169",
        "pipeline": "VIRIDIS -> GE -> CLAHE",
        "runs": {
            42: "staged_densenet169_pseudo_color_then_histogram_equalization_then_clahe_s42",
            123: "robustness_densenet169_pseudo_color_then_histogram_equalization_then_clahe_s123",
            2026: "robustness_densenet169_pseudo_color_then_histogram_equalization_then_clahe_s2026",
        },
    },
    {
        "key": "resnet50_clahe_viridis_ge",
        "model": "ResNet-50",
        "pipeline": "CLAHE -> VIRIDIS -> GE",
        "runs": {
            42: "staged_resnet50_clahe_then_pseudo_color_then_histogram_equalization_s42",
            123: "robustness_resnet50_clahe_then_pseudo_color_then_histogram_equalization_s123",
            2026: "robustness_resnet50_clahe_then_pseudo_color_then_histogram_equalization_s2026",
        },
    },
]


TARGETS = [
    {
        "model": "DenseNet-169",
        "architecture": "densenet169",
        "pipeline": "VIRIDIS -> GE -> CLAHE",
        "pipeline_id": "pseudo_color_then_histogram_equalization_then_clahe",
        "operations": [
            "VIRIDIS pseudo-coloring (VIRIDIS)",
            "Global Histogram Equalization (GE)",
            "Contrast Limited Adaptive Histogram Equalization (CLAHE)",
        ],
        "batch_size": 12,
        "source_config": PROJECT_ROOT
        / "model"
        / "configs"
        / "staged_preprocessing"
        / "densenet169"
        / "pseudo_color_then_histogram_equalization_then_clahe.json",
        "seed": seed,
        "run_id": f"robustness_densenet169_pseudo_color_then_histogram_equalization_then_clahe_s{seed}",
    }
    for seed in (123, 2026)
] + [
    {
        "model": "ResNet-50",
        "architecture": "resnet50",
        "pipeline": "CLAHE -> VIRIDIS -> GE",
        "pipeline_id": "clahe_then_pseudo_color_then_histogram_equalization",
        "operations": [
            "Contrast Limited Adaptive Histogram Equalization (CLAHE)",
            "VIRIDIS pseudo-coloring (VIRIDIS)",
            "Global Histogram Equalization (GE)",
        ],
        "batch_size": 16,
        "source_config": PROJECT_ROOT
        / "model"
        / "configs"
        / "staged_preprocessing"
        / "resnet50"
        / "clahe_then_pseudo_color_then_histogram_equalization.json",
        "seed": seed,
        "run_id": f"robustness_resnet50_clahe_then_pseudo_color_then_histogram_equalization_s{seed}",
    }
    for seed in (123, 2026)
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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
    return "Waiting" if number is None else f"{number * 100:.3f}%"


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


def run_record(candidate: dict[str, Any], seed: int, run_id: str) -> dict[str, Any]:
    run_dir = RUNS_ROOT / run_id
    status = read_json(run_dir / "status.json")
    metrics = read_json(run_dir / "metrics.json")
    viral = metrics.get("per_class", {}).get("Viral", {})
    return {
        "candidate_key": candidate["key"],
        "model": candidate["model"],
        "pipeline": candidate["pipeline"],
        "seed": seed,
        "run_id": run_id,
        "status": status.get("status", "pending"),
        "best_epoch": metrics.get("best_epoch", status.get("best_epoch")),
        "macro_f1": metrics.get("macro_f1"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "accuracy": metrics.get("accuracy"),
        "viral_recall": viral.get("recall"),
        "training_duration_seconds": metrics.get("training_duration_seconds"),
    }


def comparison_payload() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate_summaries = []
    flat_rows = []
    for candidate in CANDIDATES:
        runs = [run_record(candidate, seed, run_id) for seed, run_id in candidate["runs"].items()]
        flat_rows.extend(runs)
        completed = [row for row in runs if row["status"] == "completed" and row["macro_f1"] is not None]

        def values(key: str) -> list[float]:
            return [float(row[key]) for row in completed if row[key] is not None]

        macro_values = values("macro_f1")
        candidate_summaries.append(
            {
                "candidate_key": candidate["key"],
                "model": candidate["model"],
                "pipeline": candidate["pipeline"],
                "completed_seed_count": len(completed),
                "target_seed_count": 3,
                "mean_macro_f1": statistics.mean(macro_values) if macro_values else None,
                "sample_std_macro_f1": statistics.stdev(macro_values) if len(macro_values) > 1 else None,
                "mean_balanced_accuracy": statistics.mean(values("balanced_accuracy")) if values("balanced_accuracy") else None,
                "mean_accuracy": statistics.mean(values("accuracy")) if values("accuracy") else None,
                "mean_viral_recall": statistics.mean(values("viral_recall")) if values("viral_recall") else None,
                "runs": runs,
            }
        )
    ranked = sorted(
        candidate_summaries,
        key=lambda row: (
            row["completed_seed_count"] < 3,
            -(row["mean_macro_f1"] if row["mean_macro_f1"] is not None else -1.0),
        ),
    )
    return (
        {
            "updated_at": now_iso(),
            "selection_split": "validation",
            "locked_test_accessed": False,
            "primary_metric": "mean macro F1 across seeds 42, 123, and 2026",
            "candidates": ranked,
        },
        flat_rows,
    )


def save_comparison() -> dict[str, Any]:
    payload, rows = comparison_payload()
    atomic_write_json(SUMMARY_JSON, payload)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return payload


def render(target: dict[str, Any], queue_index: int, process_state: str) -> None:
    run_dir = RUNS_ROOT / target["run_id"]
    status = read_json(run_dir / "status.json")
    live = read_json(run_dir / "live_progress.json")
    history = read_last_history(run_dir / "history.csv")
    summary = save_comparison()

    total_epochs = as_int(status.get("total_epochs")) or 12
    completed_epoch = max(as_int(status.get("current_epoch")), as_int(history.get("epoch")))
    active_epoch = as_int(live.get("epoch")) or min(completed_epoch + 1, total_epochs)
    fraction = as_float(live.get("fraction")) or 0.0
    batch = as_int(live.get("batch"))
    total_batches = as_int(live.get("total_batches"))
    best_f1 = status.get("best_validation_macro_f1")
    if as_float(best_f1) is None:
        best_f1 = history.get("best_validation_macro_f1")

    lines = [
        "TOP MODEL COMPETITORS - MULTI-SEED CONFIRMATION",
        "=" * 94,
        f"Updated:              {now_iso()}",
        f"GPU:                  {gpu_status()}",
        f"Queue progress:       Run {queue_index + 1}/4",
        "Locked test:          NOT ACCESSED (training/validation manifests only)",
        "",
        "CURRENT TRAINING RUN",
        "-" * 94,
        f"Model:                {target['model']}",
        f"Pipeline ID:          {target['pipeline_id']}",
        f"Short order:          {target['pipeline']}",
        "Preprocessing order:",
    ]
    for index, operation in enumerate(target["operations"], start=1):
        lines.append(f"  Step {index}: {operation}")
    lines.extend(
        [
            f"Seed:                 {target['seed']}",
            f"Run ID:               {target['run_id']}",
            f"Fixed training:       ImageNet, 224x224, batch {target['batch_size']}, AdamW, LR 0.0001",
            "Augmentation:         Horizontal flip p=0.5; random rotation +/-5 degrees",
            f"Process state:         {process_state}",
            f"Training status:       {status.get('status', 'initializing')}",
            f"Completed epochs:      {completed_epoch}/{total_epochs}",
            f"Active epoch:          {active_epoch}/{total_epochs} ({live.get('stage', 'initializing')})",
            f"Current epoch:         {progress_bar(fraction)}",
            f"Current batch:         {batch}/{total_batches}" if total_batches else "Current batch:         Waiting for data loader",
            f"Epoch elapsed:         {duration(live.get('elapsed_seconds'))}",
            f"Epoch time left*:      {duration(live.get('estimated_remaining_seconds'))}",
            f"Best val macro F1:     {percent(best_f1)}",
            "",
            "LATEST COMPLETED EPOCH",
            "-" * 94,
            f"Train loss:            {decimal(history.get('train_loss'))}",
            f"Validation loss:       {decimal(history.get('val_loss'))}",
            f"Validation macro F1:   {percent(history.get('val_macro_f1'))}",
            f"Balanced accuracy:     {percent(history.get('val_balanced_accuracy'))}",
            f"Accuracy:              {percent(history.get('val_accuracy'))}",
            f"Viral recall:          {percent(history.get('recall_viral_or_pneumonia'))}",
            f"Epoch duration:        {duration(history.get('elapsed_seconds'))}",
            f"Run time left*:        {duration(history.get('estimated_remaining_seconds'))}",
            "",
            "THREE-SEED COMPARISON SNAPSHOT",
            "-" * 94,
        ]
    )
    for candidate in summary["candidates"]:
        std_text = percent(candidate["sample_std_macro_f1"]) if candidate["sample_std_macro_f1"] is not None else "-"
        lines.append(
            f"{candidate['model']:<13} | {candidate['pipeline']:<25} | "
            f"seeds {candidate['completed_seed_count']}/3 | mean F1 {percent(candidate['mean_macro_f1'])} | SD {std_text}"
        )
    lines.extend(
        [
            "",
            "REMAINING QUEUE",
            "-" * 94,
        ]
    )
    for index, item in enumerate(TARGETS):
        item_status = read_json(RUNS_ROOT / item["run_id"] / "status.json").get("status", "pending")
        marker = "CURRENT" if index == queue_index else ("DONE" if item_status == "completed" else "QUEUED")
        lines.append(
            f"{index + 1}. {item['model']:<12} seed {item['seed']:<4} | {item['pipeline']:<25} | {marker:<7} | {item_status}"
        )
    lines.extend(
        [
            "",
            "*Time estimates become more reliable after the first completed epoch.",
            "Keep the laptop plugged in and prevent sleep/shutdown while training.",
        ]
    )
    os.system("cls" if os.name == "nt" else "clear")
    print("\n".join(lines), flush=True)


def run_target(target: dict[str, Any], queue_index: int) -> bool:
    run_dir = RUNS_ROOT / target["run_id"]
    if read_json(run_dir / "status.json").get("status") == "completed":
        save_comparison()
        return True

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stdout_path = OUTPUT_ROOT / f"run_{queue_index + 1}_seed_{target['seed']}.stdout.log"
    stderr_path = OUTPUT_ROOT / f"run_{queue_index + 1}_seed_{target['seed']}.stderr.log"
    checkpoint = run_dir / "last_checkpoint.pt"
    saved_config = run_dir / "config.json"
    if checkpoint.exists() and saved_config.exists():
        command = [str(PYTHON_EXE), str(TRAIN_SCRIPT), "--config", str(saved_config), "--resume"]
    else:
        command = [
            str(PYTHON_EXE),
            str(TRAIN_SCRIPT),
            "--config",
            str(target["source_config"]),
            "--seed",
            str(target["seed"]),
            "--run-id",
            target["run_id"],
        ]

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
        atomic_write_json(
            PROGRESS_PATH,
            {
                "status": "running",
                "controller_pid": os.getpid(),
                "trainer_pid": process.pid,
                "queue_index": queue_index,
                "queue_total": len(TARGETS),
                "current_model": target["model"],
                "current_seed": target["seed"],
                "current_run_id": target["run_id"],
                "updated_at": now_iso(),
                "locked_test_accessed": False,
            },
        )
        while process.poll() is None:
            render(target, queue_index, f"RUNNING (trainer PID {process.pid})")
            time.sleep(2)
        return_code = process.returncode

    save_comparison()
    render(target, queue_index, f"EXITED (code {return_code})")
    return return_code == 0 and read_json(run_dir / "status.json").get("status") == "completed"


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    os.system("title Top Pneumonia Model Competitors - Multi-Seed Confirmation")
    for queue_index, target in enumerate(TARGETS):
        if not run_target(target, queue_index):
            atomic_write_json(
                PROGRESS_PATH,
                {
                    "status": "failed",
                    "controller_pid": os.getpid(),
                    "queue_index": queue_index,
                    "current_model": target["model"],
                    "current_seed": target["seed"],
                    "current_run_id": target["run_id"],
                    "updated_at": now_iso(),
                    "locked_test_accessed": False,
                },
            )
            print("\nQueue stopped because the current training run failed.", flush=True)
            return 1

    summary = save_comparison()
    atomic_write_json(
        PROGRESS_PATH,
        {
            "status": "completed",
            "controller_pid": os.getpid(),
            "queue_total": len(TARGETS),
            "completed_at": now_iso(),
            "locked_test_accessed": False,
        },
    )
    os.system("cls" if os.name == "nt" else "clear")
    print("TOP MODEL MULTI-SEED COMPARISON COMPLETE")
    print("=" * 82)
    for rank, candidate in enumerate(summary["candidates"], start=1):
        print(
            f"{rank}. {candidate['model']} + {candidate['pipeline']} | "
            f"seeds {candidate['completed_seed_count']}/3 | mean macro F1 {percent(candidate['mean_macro_f1'])} | "
            f"SD {percent(candidate['sample_std_macro_f1'])}"
        )
    print(f"\nSummary: {SUMMARY_JSON}")
    print("Locked test was not accessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
