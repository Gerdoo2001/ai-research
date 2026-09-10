"""Run the validation-only staged CLAHE/GE/VIRIDIS order experiment.

The staged design is intentionally smaller than a 128-cell exhaustive matrix:

1. Complete the six-column comparison table for all eight architectures.
2. Evaluate every ordered subset of C/G/V on DenseNet-201 (16 conditions).
3. Evaluate the same 16 ordered subsets on ResNet-50.
4. Select by validation macro F1 and transfer the DenseNet-201 winner to the
   other models.

The locked test split is never opened by this script.  Every target config is
derived from its architecture's completed seed-42 baseline, so augmentation,
optimizer, learning rate, loss, epoch budget, batch size, and manifests remain
matched within architecture.  Completed hashes are reused and interrupted runs
resume from their last checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import _multiclass_bootstrap  # noqa: F401
from multiclass.config import ExperimentConfig, load_config
from multiclass.data import read_manifest, validate_manifest_bundle
from multiclass.models import build_model
from multiclass.preprocessing import PREPROCESSING_PIPELINES, ImageTransform
from multiclass.tracking import atomic_write_json, now_iso


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "model"
RUNS_ROOT = MODEL_ROOT / "results" / "runs"
OUTPUT_ROOT = MODEL_ROOT / "results" / "staged_preprocessing_experiment"
CONFIG_ROOT = MODEL_ROOT / "configs" / "staged_preprocessing"
TRAIN_SCRIPT = MODEL_ROOT / "scripts" / "train_multiclass.py"

ARCHITECTURES = (
    "densenet121",
    "densenet169",
    "densenet201",
    "resnet50",
    "resnet101",
    "resnet152",
    "yolov5_cls",
    "yolov8_cls",
)

BASELINE_RUN_IDS = {
    architecture: f"baseline_{architecture}_s42" for architecture in ARCHITECTURES
}

CURRENT_TABLE_PIPELINES = (
    "baseline",
    "clahe",
    "histogram_equalization",
    "pseudo_color",
    "histogram_equalization_then_clahe",
    "clahe_then_histogram_equalization",
)

ORDERED_PIPELINES = (
    "baseline",
    "clahe",
    "histogram_equalization",
    "pseudo_color",
    "histogram_equalization_then_clahe",
    "clahe_then_histogram_equalization",
    "clahe_then_pseudo_color",
    "pseudo_color_then_clahe",
    "histogram_equalization_then_pseudo_color",
    "pseudo_color_then_histogram_equalization",
    "clahe_then_histogram_equalization_then_pseudo_color",
    "clahe_then_pseudo_color_then_histogram_equalization",
    "histogram_equalization_then_clahe_then_pseudo_color",
    "histogram_equalization_then_pseudo_color_then_clahe",
    "pseudo_color_then_clahe_then_histogram_equalization",
    "pseudo_color_then_histogram_equalization_then_clahe",
)

NEW_PIPELINES = tuple(
    pipeline for pipeline in ORDERED_PIPELINES if pipeline not in CURRENT_TABLE_PIPELINES
)

DISPLAY_NAMES = {
    "densenet121": "DenseNet-121",
    "densenet169": "DenseNet-169",
    "densenet201": "DenseNet-201",
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    "resnet152": "ResNet-152",
    "yolov5_cls": "YOLOv5",
    "yolov8_cls": "YOLOv8",
}

PIPELINE_LABELS = {
    "baseline": "Base",
    "clahe": "CLAHE",
    "histogram_equalization": "GE",
    "pseudo_color": "VIRIDIS",
    "histogram_equalization_then_clahe": "GE -> CLAHE",
    "clahe_then_histogram_equalization": "CLAHE -> GE",
    "clahe_then_pseudo_color": "CLAHE -> VIRIDIS",
    "pseudo_color_then_clahe": "VIRIDIS -> CLAHE",
    "histogram_equalization_then_pseudo_color": "GE -> VIRIDIS",
    "pseudo_color_then_histogram_equalization": "VIRIDIS -> GE",
    "clahe_then_histogram_equalization_then_pseudo_color": "CLAHE -> GE -> VIRIDIS",
    "clahe_then_pseudo_color_then_histogram_equalization": "CLAHE -> VIRIDIS -> GE",
    "histogram_equalization_then_clahe_then_pseudo_color": "GE -> CLAHE -> VIRIDIS",
    "histogram_equalization_then_pseudo_color_then_clahe": "GE -> VIRIDIS -> CLAHE",
    "pseudo_color_then_clahe_then_histogram_equalization": "VIRIDIS -> CLAHE -> GE",
    "pseudo_color_then_histogram_equalization_then_clahe": "VIRIDIS -> GE -> CLAHE",
}


@dataclass(frozen=True)
class Target:
    phase: str
    architecture: str
    preprocessing: str
    run_id: str
    config_path: Path
    config_hash: str


@dataclass(frozen=True)
class CompletedRun:
    run_id: str
    config_hash: str
    run_dir: Path
    metrics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument(
        "--from-phase",
        choices=(
            "current_table",
            "densenet201_exhaustive",
            "resnet50_exhaustive",
            "winner_transfer",
        ),
        default="current_table",
    )
    parser.add_argument(
        "--stop-after-phase",
        choices=(
            "current_table",
            "densenet201_exhaustive",
            "resnet50_exhaustive",
            "winner_transfer",
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def baseline_payload(architecture: str) -> dict[str, Any]:
    path = RUNS_ROOT / BASELINE_RUN_IDS[architecture] / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"completed baseline config is absent: {path}")
    payload = read_json(path)
    status = read_json(path.parent / "status.json")
    if status.get("status") != "completed":
        raise RuntimeError(f"baseline is not completed: {path.parent.name}")
    return payload


def preferred_existing_run_id(architecture: str, preprocessing: str) -> str | None:
    if preprocessing == "baseline":
        return BASELINE_RUN_IDS[architecture]
    candidate = f"ablation_{architecture}_{preprocessing}_s42"
    status_path = RUNS_ROOT / candidate / "status.json"
    if status_path.exists() and read_json(status_path).get("status") == "completed":
        return candidate
    return None


def staged_run_id(architecture: str, preprocessing: str) -> str:
    existing = preferred_existing_run_id(architecture, preprocessing)
    return existing or f"staged_{architecture}_{preprocessing}_s42"


def materialize_target(phase: str, architecture: str, preprocessing: str) -> Target:
    if preprocessing not in PREPROCESSING_PIPELINES:
        raise ValueError(preprocessing)
    payload = baseline_payload(architecture)
    payload["architecture"] = architecture
    payload["preprocessing"] = preprocessing
    payload["run_id"] = staged_run_id(architecture, preprocessing)
    config = ExperimentConfig(**payload)
    config_path = CONFIG_ROOT / architecture / f"{preprocessing}.json"
    atomic_write_json(config_path, config.as_dict())
    return Target(
        phase=phase,
        architecture=architecture,
        preprocessing=preprocessing,
        run_id=config.effective_run_id(),
        config_path=config_path,
        config_hash=config.config_hash,
    )


def current_table_targets() -> list[Target]:
    return [
        materialize_target("current_table", architecture, preprocessing)
        for architecture in ARCHITECTURES
        for preprocessing in CURRENT_TABLE_PIPELINES
    ]


def densenet201_targets() -> list[Target]:
    return [
        materialize_target("densenet201_exhaustive", "densenet201", preprocessing)
        for preprocessing in ORDERED_PIPELINES
    ]


def resnet50_targets() -> list[Target]:
    return [
        materialize_target("resnet50_exhaustive", "resnet50", preprocessing)
        for preprocessing in ORDERED_PIPELINES
    ]


def completed_index() -> dict[str, list[CompletedRun]]:
    by_hash: dict[str, list[CompletedRun]] = {}
    if not RUNS_ROOT.exists():
        return by_hash
    for run_dir in sorted(path for path in RUNS_ROOT.iterdir() if path.is_dir()):
        status_path = run_dir / "status.json"
        metrics_path = run_dir / "metrics.json"
        if not status_path.exists() or not metrics_path.exists():
            continue
        try:
            status = read_json(status_path)
            metrics = read_json(metrics_path)
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("status") != "completed" or metrics.get("split") != "validation":
            continue
        config_hash = str(status.get("config_hash", ""))
        if not config_hash:
            continue
        by_hash.setdefault(config_hash, []).append(
            CompletedRun(run_dir.name, config_hash, run_dir, metrics)
        )
    return by_hash


def completed_for(target: Target, index: dict[str, list[CompletedRun]]) -> CompletedRun | None:
    matches = index.get(target.config_hash, [])
    if not matches:
        return None

    def preference(run: CompletedRun) -> tuple[int, str]:
        prefixes = ("baseline_", "ablation_", "staged_", "final_", "tuning_")
        rank = next((i for i, prefix in enumerate(prefixes) if run.run_id.startswith(prefix)), 99)
        return rank, run.run_id

    return sorted(matches, key=preference)[0]


def target_row(target: Target, index: dict[str, list[CompletedRun]]) -> dict[str, Any]:
    completed = completed_for(target, index)
    metrics = completed.metrics if completed else {}
    return {
        "phase": target.phase,
        "architecture": target.architecture,
        "model": DISPLAY_NAMES[target.architecture],
        "preprocessing": target.preprocessing,
        "display_pipeline": PIPELINE_LABELS[target.preprocessing],
        "operation_sequence": " -> ".join(PREPROCESSING_PIPELINES[target.preprocessing]) or "none",
        "operation_count": len(PREPROCESSING_PIPELINES[target.preprocessing]),
        "target_run_id": target.run_id,
        "source_run_id": completed.run_id if completed else "",
        "config_hash": target.config_hash,
        "config_path": str(target.config_path),
        "status": "completed_reused" if completed else "pending_training",
        "validation_macro_f1": metrics.get("macro_f1", ""),
        "validation_balanced_accuracy": metrics.get("balanced_accuracy", ""),
        "validation_accuracy": metrics.get("accuracy", ""),
    }


def write_plan(targets: list[Target], index: dict[str, list[CompletedRun]]) -> None:
    rows = [target_row(target, index) for target in targets]
    columns = list(rows[0])
    write_csv(OUTPUT_ROOT / "experiment_plan.csv", rows, columns)
    atomic_write_json(
        OUTPUT_ROOT / "experiment_plan.json",
        {
            "schema_version": 1,
            "generated_at": now_iso(),
            "selection_split": "validation only",
            "locked_test_accessed": False,
            "selection_rule": [
                "maximum validation macro F1",
                "maximum validation balanced accuracy",
                "maximum validation accuracy",
                "fewer enhancement operations",
                "lexical pipeline name for a deterministic final tie",
            ],
            "design": {
                "current_table_cells": len(ARCHITECTURES) * len(CURRENT_TABLE_PIPELINES),
                "densenet201_ordered_cells": len(ORDERED_PIPELINES),
                "resnet50_ordered_cells": len(ORDERED_PIPELINES),
                "winner_transfer_cells": len(ARCHITECTURES),
                "maximum_new_training_runs": 56,
            },
            "rows": rows,
        },
    )


def write_progress(
    *,
    state: str,
    phase: str,
    targets: Iterable[Target],
    index: dict[str, list[CompletedRun]],
    current_target: Target | None = None,
    message: str = "",
) -> None:
    target_list = list(targets)
    completed = sum(completed_for(target, index) is not None for target in target_list)
    atomic_write_json(
        OUTPUT_ROOT / "progress.json",
        {
            "schema_version": 1,
            "updated_at": now_iso(),
            "state": state,
            "phase": phase,
            "message": message,
            "current_target": (
                {
                    "run_id": current_target.run_id,
                    "architecture": current_target.architecture,
                    "preprocessing": current_target.preprocessing,
                    "config_path": str(current_target.config_path),
                }
                if current_target
                else None
            ),
            "phase_targets": len(target_list),
            "phase_completed": completed,
            "phase_remaining": len(target_list) - completed,
            "locked_test_accessed": False,
            "runner_pid": os.getpid(),
        },
    )


def run_target(target: Target, index: dict[str, list[CompletedRun]]) -> None:
    if completed_for(target, index):
        return
    run_dir = RUNS_ROOT / target.run_id
    status_path = run_dir / "status.json"
    resume = False
    if status_path.exists():
        status = read_json(status_path)
        existing_hash = status.get("config_hash")
        if existing_hash and existing_hash != target.config_hash:
            raise RuntimeError(
                f"refusing hash mismatch in {target.run_id}: {existing_hash} != {target.config_hash}"
            )
        resume = (run_dir / "last_checkpoint.pt").exists()

    command = [sys.executable, str(TRAIN_SCRIPT), "--config", str(target.config_path)]
    if resume:
        command.append("--resume")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{target.run_id} exited with code {completed.returncode}")


def run_targets(phase: str, targets: list[Target]) -> dict[str, list[CompletedRun]]:
    index = completed_index()
    write_progress(
        state="running",
        phase=phase,
        targets=targets,
        index=index,
        message=f"Starting or resuming {phase}.",
    )
    for target in targets:
        index = completed_index()
        if completed_for(target, index):
            continue
        write_progress(
            state="running",
            phase=phase,
            targets=targets,
            index=index,
            current_target=target,
            message="Training one matched seed-42 configuration on train; selecting on validation.",
        )
        run_target(target, index)
        index = completed_index()
        if not completed_for(target, index):
            raise RuntimeError(f"run returned without a completed validation artifact: {target.run_id}")
    write_progress(
        state="phase_completed",
        phase=phase,
        targets=targets,
        index=index,
        message=f"All {phase} targets are complete.",
    )
    return index


def metric_value(run: CompletedRun, key: str) -> float:
    return float(run.metrics[key])


def select_architecture_winner(
    architecture: str,
    targets: list[Target],
    index: dict[str, list[CompletedRun]],
) -> tuple[Target, CompletedRun]:
    candidates: list[tuple[Target, CompletedRun]] = []
    for target in targets:
        completed = completed_for(target, index)
        if not completed:
            raise RuntimeError(f"cannot select winner; target is incomplete: {target.preprocessing}")
        candidates.append((target, completed))
    candidates.sort(
        key=lambda item: (
            -metric_value(item[1], "macro_f1"),
            -metric_value(item[1], "balanced_accuracy"),
            -metric_value(item[1], "accuracy"),
            len(PREPROCESSING_PIPELINES[item[0].preprocessing]),
            item[0].preprocessing,
        )
    )
    winner = candidates[0]
    atomic_write_json(
        OUTPUT_ROOT / f"{architecture}_winner.json",
        {
            "selected_at": now_iso(),
            "selection_split": "validation",
            "locked_test_accessed": False,
            "architecture": winner[0].architecture,
            "preprocessing": winner[0].preprocessing,
            "display_pipeline": PIPELINE_LABELS[winner[0].preprocessing],
            "source_run_id": winner[1].run_id,
            "validation_macro_f1": winner[1].metrics["macro_f1"],
            "validation_balanced_accuracy": winner[1].metrics["balanced_accuracy"],
            "validation_accuracy": winner[1].metrics["accuracy"],
            "tie_breakers": [
                "balanced accuracy",
                "accuracy",
                "fewer enhancement operations",
                "pipeline name",
            ],
        },
    )
    return winner


def winner_transfer_targets(preprocessing: str) -> list[Target]:
    return [
        materialize_target("winner_transfer", architecture, preprocessing)
        for architecture in ARCHITECTURES
    ]


def format_percent(value: Any) -> str:
    return "" if value in (None, "") else f"{100.0 * float(value):.3f}%"


def write_result_tables(
    table_targets: list[Target],
    densenet_targets: list[Target],
    resnet50_order_targets: list[Target],
    winner_targets: list[Target] | None = None,
) -> None:
    index = completed_index()
    wide_rows: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        row: dict[str, Any] = {"Model": DISPLAY_NAMES[architecture]}
        for pipeline in CURRENT_TABLE_PIPELINES:
            target = next(
                item
                for item in table_targets
                if item.architecture == architecture and item.preprocessing == pipeline
            )
            completed = completed_for(target, index)
            row[PIPELINE_LABELS[pipeline]] = (
                format_percent(completed.metrics["macro_f1"]) if completed else ""
            )
        wide_rows.append(row)
    write_csv(
        OUTPUT_ROOT / "current_six_column_table.csv",
        wide_rows,
        ["Model", *[PIPELINE_LABELS[pipeline] for pipeline in CURRENT_TABLE_PIPELINES]],
    )

    def write_all_orders_table(filename: str, targets: list[Target]) -> None:
        rows: list[dict[str, Any]] = []
        for target in targets:
            completed = completed_for(target, index)
            metrics = completed.metrics if completed else {}
            rows.append(
                {
                    "Order": PIPELINE_LABELS[target.preprocessing],
                    "Pipeline ID": target.preprocessing,
                    "Operations": len(PREPROCESSING_PIPELINES[target.preprocessing]),
                    "Validation Macro F1": format_percent(metrics.get("macro_f1")),
                    "Validation Balanced Accuracy": format_percent(
                        metrics.get("balanced_accuracy")
                    ),
                    "Validation Accuracy": format_percent(metrics.get("accuracy")),
                    "Run ID": completed.run_id if completed else "",
                    "Status": "Completed" if completed else "Pending",
                }
            )
        write_csv(OUTPUT_ROOT / filename, rows, list(rows[0]))

    write_all_orders_table("densenet201_all_orders.csv", densenet_targets)
    write_all_orders_table("resnet50_all_orders.csv", resnet50_order_targets)

    if winner_targets:
        winner_rows: list[dict[str, Any]] = []
        for target in winner_targets:
            completed = completed_for(target, index)
            metrics = completed.metrics if completed else {}
            winner_rows.append(
                {
                    "Model": DISPLAY_NAMES[target.architecture],
                    "Winning Pipeline": PIPELINE_LABELS[target.preprocessing],
                    "Validation Macro F1": format_percent(metrics.get("macro_f1")),
                    "Validation Balanced Accuracy": format_percent(metrics.get("balanced_accuracy")),
                    "Validation Accuracy": format_percent(metrics.get("accuracy")),
                    "Run ID": completed.run_id if completed else "",
                    "Status": "Completed" if completed else "Pending",
                }
            )
        write_csv(
            OUTPUT_ROOT / "winner_cross_model_table.csv",
            winner_rows,
            list(winner_rows[0]),
        )


def gpu_smoke_new_pipelines() -> None:
    import torch
    import torch.nn.functional as functional
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the staged GPU smoke validation")
    summary = validate_manifest_bundle(MODEL_ROOT / "data_manifests")
    rows = read_manifest(MODEL_ROOT / "data_manifests" / "train_manifest.csv")
    source_path = Path(rows[0]["filepath"])
    with Image.open(source_path) as opened:
        source_image = opened.copy()

    payload = baseline_payload("densenet201")
    payload.update(weights="none", tensorboard=False, run_id="smoke_staged_order_contract")
    config = ExperimentConfig(**payload)
    model, metadata = build_model(config)
    device = torch.device("cuda")
    model.to(device).train()
    records: list[dict[str, Any]] = []
    for preprocessing in NEW_PIPELINES:
        transform = ImageTransform(preprocessing, config.input_size, training=False)
        inputs = transform(source_image).unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            logits = model(inputs)
            loss = functional.cross_entropy(logits, torch.tensor([0], device=device))
        loss.backward()
        if tuple(logits.shape) != (1, 3) or not bool(torch.isfinite(loss)):
            raise RuntimeError(f"GPU smoke contract failed for {preprocessing}")
        records.append(
            {
                "preprocessing": preprocessing,
                "display_pipeline": PIPELINE_LABELS[preprocessing],
                "input_shape": list(inputs.shape),
                "logits_shape": list(logits.shape),
                "loss": float(loss.detach().cpu()),
                "finite": True,
            }
        )
    atomic_write_json(
        OUTPUT_ROOT / "gpu_smoke.json",
        {
            "completed_at": now_iso(),
            "gpu": torch.cuda.get_device_name(0),
            "architecture": metadata.architecture,
            "weights": "none (contract test only)",
            "source_image": str(source_path),
            "manifest_version": summary["manifest_version"],
            "locked_test_accessed": False,
            "pipelines_tested": len(records),
            "records": records,
        },
    )
    del model, inputs, logits, loss
    torch.cuda.empty_cache()


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


def acquire_lock() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / "runner.lock.json"
    if path.exists():
        try:
            prior = read_json(path)
            prior_pid = int(prior.get("pid", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            prior_pid = 0
        if pid_is_running(prior_pid):
            raise RuntimeError(f"staged runner is already active with PID {prior_pid}")
        path.unlink()
    atomic_write_json(path, {"pid": os.getpid(), "started_at": now_iso()})
    return path


def main() -> int:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.prepare_only or args.smoke_only:
        table_targets = current_table_targets()
        dense_targets = densenet201_targets()
        resnet50_order_targets = resnet50_targets()
        index = completed_index()
        write_plan(table_targets + dense_targets + resnet50_order_targets, index)
        write_result_tables(table_targets, dense_targets, resnet50_order_targets)
        if args.prepare_only:
            print(f"Prepared staged experiment in {OUTPUT_ROOT}", flush=True)
            return 0
        gpu_smoke_new_pipelines()
        print(f"GPU-smoked {len(NEW_PIPELINES)} new ordered pipelines.", flush=True)
        return 0

    # Acquire the singleton lock before materializing configs or scanning the
    # matrix.  This prevents the monitor from starting overlapping controllers
    # while a slow Windows disk is rebuilding the experiment index.
    lock_path = acquire_lock()
    all_phases = (
        "current_table",
        "densenet201_exhaustive",
        "resnet50_exhaustive",
        "winner_transfer",
    )
    start_index = all_phases.index(args.from_phase)
    try:
        table_targets = current_table_targets()
        dense_targets = densenet201_targets()
        resnet50_order_targets = resnet50_targets()
        index = completed_index()
        write_plan(table_targets + dense_targets + resnet50_order_targets, index)
        write_result_tables(table_targets, dense_targets, resnet50_order_targets)

        if not args.skip_smoke:
            gpu_smoke_new_pipelines()

        if start_index <= 0:
            index = run_targets("current_table", table_targets)
            write_result_tables(table_targets, dense_targets, resnet50_order_targets)
            if args.stop_after_phase == "current_table":
                return 0

        if start_index <= 1:
            index = run_targets("densenet201_exhaustive", dense_targets)
            write_result_tables(table_targets, dense_targets, resnet50_order_targets)
            if args.stop_after_phase == "densenet201_exhaustive":
                return 0
        else:
            index = completed_index()

        if start_index <= 2:
            index = run_targets("resnet50_exhaustive", resnet50_order_targets)
            write_result_tables(table_targets, dense_targets, resnet50_order_targets)
            if args.stop_after_phase == "resnet50_exhaustive":
                return 0
        else:
            index = completed_index()

        winner_target, winner_run = select_architecture_winner(
            "densenet201", dense_targets, index
        )
        resnet50_winner_target, resnet50_winner_run = select_architecture_winner(
            "resnet50", resnet50_order_targets, index
        )
        transfer_targets = winner_transfer_targets(winner_target.preprocessing)
        write_result_tables(
            table_targets, dense_targets, resnet50_order_targets, transfer_targets
        )
        if start_index <= 3:
            index = run_targets("winner_transfer", transfer_targets)
        write_result_tables(
            table_targets, dense_targets, resnet50_order_targets, transfer_targets
        )
        write_progress(
            state="completed",
            phase="winner_transfer",
            targets=transfer_targets,
            index=index,
            message=(
                f"Staged experiment completed. DenseNet-201 selected "
                f"{PIPELINE_LABELS[winner_target.preprocessing]} from validation macro F1 "
                f"{float(winner_run.metrics['macro_f1']):.6f}. ResNet-50 selected "
                f"{PIPELINE_LABELS[resnet50_winner_target.preprocessing]} from "
                f"validation macro F1 "
                f"{float(resnet50_winner_run.metrics['macro_f1']):.6f}."
            ),
        )
        return 0
    except BaseException as error:
        index = completed_index()
        atomic_write_json(
            OUTPUT_ROOT / "runner_error.json",
            {
                "failed_at": now_iso(),
                "error_type": type(error).__name__,
                "error": str(error),
                "resume_command": (
                    ".venv\\Scripts\\python.exe "
                    "model\\scripts\\run_staged_preprocessing_experiment.py --skip-smoke"
                ),
                "locked_test_accessed": False,
            },
        )
        raise
    finally:
        if lock_path.exists():
            lock_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
