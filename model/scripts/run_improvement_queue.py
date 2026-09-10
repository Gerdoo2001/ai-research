"""Persistent queue for ensemble confirmation followed by the PSPNet ROI audit."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
RUNS_ROOT = REPO_ROOT / "model" / "results" / "runs"
OUTPUT_ROOT = REPO_ROOT / "model" / "results" / "improvement_queue"
PROGRESS_PATH = OUTPUT_ROOT / "progress.json"
EVENTS_PATH = OUTPUT_ROOT / "events.jsonl"
LOG_ROOT = OUTPUT_ROOT / "logs"
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


NEW_RUNS = [
    {
        "item_id": "ENS-VGE-S123",
        "config_path": "model/configs/fine_tuning/ensemble_confirmation/densenet201_viridis_ge_s123.json",
        "run_id": "confirmation_densenet201_viridis_ge_s123",
        "seed": 123,
        "member": "VIRIDIS -> GE",
    },
    {
        "item_id": "ENS-GE-S123",
        "config_path": "model/configs/fine_tuning/ensemble_confirmation/densenet201_ge_s123.json",
        "run_id": "confirmation_densenet201_ge_s123",
        "seed": 123,
        "member": "GE",
    },
    {
        "item_id": "ENS-VGE-S2026",
        "config_path": "model/configs/fine_tuning/ensemble_confirmation/densenet201_viridis_ge_s2026.json",
        "run_id": "confirmation_densenet201_viridis_ge_s2026",
        "seed": 2026,
        "member": "VIRIDIS -> GE",
    },
    {
        "item_id": "ENS-GE-S2026",
        "config_path": "model/configs/fine_tuning/ensemble_confirmation/densenet201_ge_s2026.json",
        "run_id": "confirmation_densenet201_ge_s2026",
        "seed": 2026,
        "member": "GE",
    },
]

ENSEMBLE_MEMBERS = {
    "42": [
        "staged_densenet201_pseudo_color_then_histogram_equalization_then_clahe_s42",
        "staged_densenet201_pseudo_color_then_histogram_equalization_s42",
        "ablation_densenet201_histogram_equalization_s42",
    ],
    "123": [
        "robustness_densenet201_pseudo_color_then_histogram_equalization_then_clahe_s123",
        "confirmation_densenet201_viridis_ge_s123",
        "confirmation_densenet201_ge_s123",
    ],
    "2026": [
        "robustness_densenet201_pseudo_color_then_histogram_equalization_then_clahe_s2026",
        "confirmation_densenet201_viridis_ge_s2026",
        "confirmation_densenet201_ge_s2026",
    ],
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    for _ in range(5):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, PermissionError):
            time.sleep(0.1)
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


def event(state: dict, level: str, message: str, **details: object) -> None:
    record = {"timestamp": now(), "level": level, "message": message, **details}
    state.setdefault("events", []).append(record)
    state["events"] = state["events"][-40:]
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def last_history(run_dir: Path) -> dict:
    path = run_dir / "history.csv"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return rows[-1] if rows else {}
    except (PermissionError, OSError):
        return {}


def run_snapshot(run_id: str) -> dict:
    run_dir = RUNS_ROOT / run_id
    status = read_json(run_dir / "status.json")
    metrics = read_json(run_dir / "metrics.json")
    live = read_json(run_dir / "live_progress.json")
    return {
        "run_dir": str(run_dir),
        "status": status,
        "live_progress": live,
        "latest_completed_epoch": last_history(run_dir),
        "metrics": metrics,
        "artifacts": {
            "best_checkpoint": str(run_dir / "best_checkpoint.pt"),
            "last_checkpoint": str(run_dir / "last_checkpoint.pt"),
            "predictions": str(run_dir / "predictions.csv"),
            "history": str(run_dir / "history.csv"),
            "metrics": str(run_dir / "metrics.json"),
        },
    }


def config_record(item: dict) -> dict:
    path = REPO_ROOT / item["config_path"]
    config = read_json(path)
    return {
        **item,
        "config_path_absolute": str(path),
        "configuration": config,
        "estimated_seconds": 5700,
        "status": "pending",
        "snapshot": {},
        "stdout_log": str(LOG_ROOT / f"{item['run_id']}.stdout.log"),
        "stderr_log": str(LOG_ROOT / f"{item['run_id']}.stderr.log"),
    }


def refresh_runs(state: dict) -> None:
    completed = 0
    for item in state["stages"][0]["training_runs"]:
        snapshot = run_snapshot(item["run_id"])
        item["snapshot"] = snapshot
        run_status = snapshot.get("status", {}).get("status")
        if run_status == "completed":
            item["status"] = "completed"
            completed += 1
        elif state.get("current", {}).get("run_id") == item["run_id"]:
            item["status"] = "running"
        elif (RUNS_ROOT / item["run_id"] / "last_checkpoint.pt").exists():
            item["status"] = "pending_resume"
        else:
            item["status"] = "pending"
    state["stages"][0]["completed_training_runs"] = completed
    state["stages"][0]["total_training_runs"] = len(state["stages"][0]["training_runs"])


def publish(state: dict) -> None:
    refresh_runs(state)
    state["updated_at"] = now()
    completed = state["stages"][0]["completed_training_runs"]
    state["queue_summary"] = {
        "completed_training_runs": completed,
        "total_training_runs": len(NEW_RUNS),
        "pending_training_runs": len(NEW_RUNS) - completed,
        "locked_test_accessed": False,
        "controller_pid": os.getpid(),
    }
    atomic_json(PROGRESS_PATH, state)


def initial_state() -> dict:
    import numpy
    import torch
    import torchvision

    return {
        "schema_version": 1,
        "queue_id": "FT-004-ENSEMBLE-THEN-FT-002-ROI",
        "title": "Ensemble multi-seed confirmation followed by pediatric lung ROI audit",
        "status": "initializing",
        "started_at": now(),
        "updated_at": now(),
        "scientific_controls": {
            "manifest_version": "three_class_seed42_3b815cf769c8",
            "manifest_hash": "3b815cf769c85658a04748da5eb2e15b981b89d9c9b77a404e90fe23473d2774",
            "train_images": 4076,
            "validation_images": 874,
            "test_images": 874,
            "locked_test_accessed": False,
            "selection_metric": "validation macro F1",
            "class_order": ["Normal", "Viral", "Bacterial"],
            "one_gpu_run_at_a_time": True,
        },
        "environment": {
            "python": platform.python_version(),
            "executable": str(PYTHON),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": numpy.__version__,
            "torchxrayvision": version("torchxrayvision"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "stages": [
            {
                "stage_id": "FT-004-ENS-CONFIRMATION",
                "order": 1,
                "status": "pending",
                "objective": "Confirm equal-probability ensemble across seeds 42, 123, and 2026",
                "method": "Arithmetic mean of three class-probability vectors",
                "members": ["VIRIDIS -> GE -> CLAHE", "VIRIDIS -> GE", "GE"],
                "seeds": [42, 123, 2026],
                "existing_members": ENSEMBLE_MEMBERS,
                "training_runs": [config_record(item) for item in NEW_RUNS],
                "completed_training_runs": 0,
                "total_training_runs": len(NEW_RUNS),
                "evaluation_output": str(REPO_ROOT / "model/results/ensemble_confirmation/summary.json"),
                "confirmation_rule": "Report each seed plus mean and sample standard deviation; no test feedback",
            },
            {
                "stage_id": "FT-002-ROI-AUDIT",
                "order": 2,
                "status": "pending",
                "objective": "Audit pretrained lung segmentation before any ROI classifier is trained",
                "model": {
                    "package": "torchxrayvision 1.5.2",
                    "architecture": "ChestX-Det PSPNet",
                    "parameters": 68068316,
                    "native_input": "one-channel square CXR; internally resized to 512 x 512",
                    "input_normalization": "[-1024, 1024], then internal ImageNet normalization",
                    "output": "14 anatomical logits at 512 x 512",
                    "lung_channels": ["Left Lung", "Right Lung"],
                    "threshold": 0.5,
                    "crop_margin": "8% of predicted lung bounding box",
                },
                "audit": {
                    "source": "locked validation manifest only",
                    "selection_seed": 42,
                    "images_per_class": 30,
                    "total_images": 90,
                    "classes": ["Normal", "Viral", "Bacterial"],
                    "automated_checks": [
                        "bilateral mask non-empty",
                        "left and right lung area ranges",
                        "combined lung area range",
                        "bounding-box width and height ranges",
                        "raw fragmentation count",
                    ],
                    "automated_gate": "at least 95% pass and no empty bilateral masks",
                    "manual_gate": "mandatory review of every contact sheet before ROI classification",
                    "progress_path": str(REPO_ROOT / "model/results/roi_pspnet_audit/progress.json"),
                    "summary_path": str(REPO_ROOT / "model/results/roi_pspnet_audit/summary.json"),
                },
                "status_detail": {},
            },
        ],
        "current": {},
        "events": [],
    }


def launch_training(state: dict, item: dict) -> None:
    run_dir = RUNS_ROOT / item["run_id"]
    resume = (run_dir / "last_checkpoint.pt").exists() and read_json(run_dir / "status.json").get("status") != "completed"
    if resume:
        config_path = run_dir / "config.json"
        command = [str(PYTHON), "model/scripts/train_multiclass.py", "--config", str(config_path), "--resume"]
        action = "resuming"
    else:
        config_path = REPO_ROOT / item["config_path"]
        command = [str(PYTHON), "model/scripts/train_multiclass.py", "--config", str(config_path)]
        action = "starting"
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    event(state, "INFO", f"{action} ensemble confirmation run", item_id=item["item_id"], run_id=item["run_id"])
    state["status"] = "running"
    state["stages"][0]["status"] = "running"
    state["current"] = {
        "type": "training",
        "stage_id": "FT-004-ENS-CONFIRMATION",
        "item_id": item["item_id"],
        "run_id": item["run_id"],
        "command": command,
        "action": action,
        "started_at": now(),
        "configuration": item["configuration"],
    }
    publish(state)
    with Path(item["stdout_log"]).open("a", encoding="utf-8") as stdout, Path(item["stderr_log"]).open(
        "a", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=stdout,
            stderr=stderr,
            creationflags=CREATE_NO_WINDOW,
        )
        state["current"]["process_id"] = process.pid
        publish(state)
        while process.poll() is None:
            snapshot = run_snapshot(item["run_id"])
            state["current"]["snapshot"] = snapshot
            state["current"]["elapsed_seconds"] = (
                datetime.now().astimezone() - datetime.fromisoformat(state["current"]["started_at"])
            ).total_seconds()
            publish(state)
            time.sleep(5)
        return_code = process.returncode
    final = run_snapshot(item["run_id"])
    state["current"]["snapshot"] = final
    state["current"]["return_code"] = return_code
    if return_code != 0 or final.get("status", {}).get("status") != "completed":
        state["status"] = "failed"
        state["stages"][0]["status"] = "failed"
        event(
            state,
            "ERROR",
            "training run failed",
            run_id=item["run_id"],
            return_code=return_code,
            failure_reason=final.get("status", {}).get("failure_reason"),
        )
        publish(state)
        raise RuntimeError(f"training failed: {item['run_id']}")
    event(
        state,
        "INFO",
        "ensemble confirmation run completed",
        run_id=item["run_id"],
        best_epoch=final["status"].get("best_epoch"),
        macro_f1=final.get("metrics", {}).get("macro_f1"),
        accuracy=final.get("metrics", {}).get("accuracy"),
        viral_recall=final.get("metrics", {}).get("per_class", {}).get("Viral", {}).get("recall"),
    )
    state["current"] = {}
    publish(state)


def run_substage(state: dict, stage_index: int, substage_type: str, command: list[str], stdout_name: str) -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    state["status"] = "running"
    state["stages"][stage_index]["status"] = "running"
    state["current"] = {
        "type": substage_type,
        "stage_id": state["stages"][stage_index]["stage_id"],
        "command": command,
        "started_at": now(),
    }
    event(state, "INFO", f"starting {substage_type}")
    publish(state)
    with (LOG_ROOT / f"{stdout_name}.stdout.log").open("a", encoding="utf-8") as stdout, (
        LOG_ROOT / f"{stdout_name}.stderr.log"
    ).open("a", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=stdout,
            stderr=stderr,
            creationflags=CREATE_NO_WINDOW,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        state["current"]["process_id"] = process.pid
        publish(state)
        while process.poll() is None:
            if substage_type == "roi_mask_audit":
                roi_progress = read_json(REPO_ROOT / "model/results/roi_pspnet_audit/progress.json")
                state["current"]["roi_progress"] = roi_progress
                state["stages"][1]["status_detail"] = roi_progress
            state["current"]["elapsed_seconds"] = (
                datetime.now().astimezone() - datetime.fromisoformat(state["current"]["started_at"])
            ).total_seconds()
            publish(state)
            time.sleep(3)
        return_code = process.returncode
    state["current"]["return_code"] = return_code
    publish(state)
    return return_code


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    state = initial_state()
    event(state, "INFO", "two-stage improvement queue initialized")
    publish(state)
    try:
        for item in state["stages"][0]["training_runs"]:
            if run_snapshot(item["run_id"]).get("status", {}).get("status") == "completed":
                event(state, "INFO", "completed run reused", run_id=item["run_id"])
                publish(state)
                continue
            launch_training(state, item)

        ensemble_command = [str(PYTHON), "model/scripts/evaluate_validation_ensemble.py"]
        if run_substage(state, 0, "ensemble_evaluation", ensemble_command, "ensemble_evaluation") != 0:
            raise RuntimeError("ensemble evaluation failed")
        ensemble_summary = read_json(REPO_ROOT / "model/results/ensemble_confirmation/summary.json")
        state["stages"][0]["status"] = "completed"
        state["stages"][0]["result"] = ensemble_summary
        event(
            state,
            "INFO",
            "ensemble multi-seed confirmation completed",
            mean_macro_f1=ensemble_summary.get("aggregate", {}).get("macro_f1", {}).get("mean"),
        )
        state["current"] = {}
        publish(state)

        roi_command = [
            str(PYTHON),
            "model/scripts/audit_pspnet_lung_roi.py",
            "--count-per-class", "30",
            "--seed", "42",
            "--threshold", "0.5",
            "--margin-fraction", "0.08",
            "--device", "cuda",
        ]
        if run_substage(state, 1, "roi_mask_audit", roi_command, "roi_mask_audit") != 0:
            raise RuntimeError("ROI mask audit failed")
        roi_summary = read_json(REPO_ROOT / "model/results/roi_pspnet_audit/summary.json")
        state["stages"][1]["status"] = "awaiting_manual_review"
        state["stages"][1]["result"] = roi_summary
        state["status"] = "awaiting_roi_approval"
        state["current"] = {
            "type": "approval_gate",
            "stage_id": "FT-002-ROI-AUDIT",
            "message": "Review every ROI contact sheet before ROI classifier training is authorized.",
            "approval_file": str(REPO_ROOT / "model/results/roi_pspnet_audit/manual_approval.json"),
            "contact_sheets": roi_summary.get("contact_sheets", []),
        }
        event(
            state,
            "ACTION_REQUIRED",
            "ROI automated audit completed; manual pediatric mask review required",
            automated_pass_rate=roi_summary.get("automated_pass_rate"),
            automated_gate_passed=roi_summary.get("automated_gate_passed"),
        )
        publish(state)
        return 0
    except Exception as exc:
        state["status"] = "failed"
        event(state, "ERROR", "queue stopped on failure", error=f"{type(exc).__name__}: {exc}")
        publish(state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
