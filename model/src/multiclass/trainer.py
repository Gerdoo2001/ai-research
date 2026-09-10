"""Resumable mixed-precision training loop for manifest-driven experiments."""

from __future__ import annotations

import json
import math
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .data import build_dataloaders, remap_class_index
from .evaluation import evaluate_model, measure_inference, plot_training_history, save_evaluation_artifacts
from .models import assert_output_contract, build_model
from .tracking import (
    append_failure,
    append_journal,
    atomic_write_json,
    now_iso,
    update_ledger,
    update_current_status,
    write_history,
)


def set_deterministic_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(config):
    import torch

    requested = config.device.lower()
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if not config.allow_cpu_fallback:
            raise RuntimeError("CUDA was requested but is unavailable; CPU fallback is disabled")
    return torch.device("cpu")


class FocalLoss:
    def __init__(self, weight=None, gamma: float = 2.0, label_smoothing: float = 0.0) -> None:
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def __call__(self, logits, targets):
        import torch
        import torch.nn.functional as functional

        log_probabilities = functional.log_softmax(logits, dim=1)
        target_probability = log_probabilities.gather(1, targets[:, None]).squeeze(1).exp()
        ce = functional.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        focal = ((1 - target_probability) ** self.gamma) * ce
        if self.weight is not None:
            focal = focal * self.weight[targets]
        return focal.mean()


def build_loss(config, train_rows, device):
    import torch
    import torch.nn as nn

    weight = None
    if config.class_weighting == "balanced":
        counts = np.zeros(config.num_classes, dtype=np.float64)
        for row in train_rows:
            label = remap_class_index(int(row["class_index"]), config.task_mode)
            counts[label] += 1
        weights = counts.sum() / (len(counts) * counts)
        weight = torch.tensor(weights, dtype=torch.float32, device=device)
    elif config.class_weighting != "none":
        raise ValueError("class_weighting must be 'balanced' or 'none'")
    if config.loss == "focal":
        return FocalLoss(
            weight=weight,
            gamma=config.effective_focal_gamma,
            label_smoothing=config.label_smoothing,
        )
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=config.label_smoothing)


def build_optimizer(config, model):
    import torch

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if config.optimizer == "adam":
        return torch.optim.Adam(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            parameters, lr=config.learning_rate, momentum=0.9, nesterov=True, weight_decay=config.weight_decay
        )
    return torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)


def build_scheduler(config, optimizer):
    import torch

    if config.scheduler == "none":
        return None
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )


def _status_template(config, run_dir: Path, manifest_hash: str) -> dict[str, Any]:
    run_id = config.effective_run_id()
    created = now_iso()
    return {
        "run_id": run_id,
        "config_hash": config.config_hash,
        "manifest_hash": manifest_hash,
        "status": "queued",
        "created_at": created,
        "started_at": None,
        "updated_at": created,
        "finished_at": None,
        "current_epoch": 0,
        "total_epochs": config.epochs,
        "best_epoch": None,
        "best_validation_macro_f1": None,
        "last_checkpoint": str(run_dir / "last_checkpoint.pt"),
        "best_checkpoint": str(run_dir / "best_checkpoint.pt"),
        "failure_reason": None,
        "resume_command": (
            f'.venv\\Scripts\\python.exe model\\scripts\\train_multiclass.py '
            f'--config "{run_dir / "config.json"}" --resume'
        ),
    }


def _ledger_record(config, status: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    return {
        "run_id": status["run_id"],
        "config_hash": config.config_hash,
        "architecture": config.architecture,
        "preprocessing": config.preprocessing,
        "task_mode": config.task_mode,
        "seed": config.seed,
        "manifest_hash": status["manifest_hash"],
        "status": status["status"],
        "created_at": status["created_at"],
        "started_at": status["started_at"],
        "updated_at": status["updated_at"],
        "finished_at": status["finished_at"],
        "best_epoch": status["best_epoch"],
        "best_validation_macro_f1": status["best_validation_macro_f1"],
        "run_dir": str(run_dir),
        "failure_reason": status["failure_reason"],
        "resume_command": status["resume_command"],
    }


def _save_checkpoint(path, *, epoch, model, optimizer, scheduler, scaler, config, history, best_epoch, best_f1):
    import torch

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict(),
        "config": config.as_dict(),
        "config_hash": config.config_hash,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def _load_checkpoint(path, model, optimizer, scheduler, scaler, config, device):
    import torch

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["config_hash"] != config.config_hash:
        raise RuntimeError("resume checkpoint configuration hash does not match")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
    return checkpoint


def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, *, amp: bool, progress_callback=None
):
    import torch

    model.train()
    total_loss = 0.0
    seen = 0
    total_batches = len(loader)
    for batch_index, (inputs, targets, _) in enumerate(loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(inputs)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu()) * inputs.shape[0]
        seen += inputs.shape[0]
        if progress_callback is not None:
            progress_callback(batch_index, total_batches, total_loss / max(1, seen))
    return total_loss / max(1, seen)


def run_training(config, *, resume: bool = False) -> dict[str, Any]:
    import torch

    set_deterministic_seed(config.seed)
    run_id = config.effective_run_id()
    run_dir = config.results_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    live_progress_path = run_dir / "live_progress.json"
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed" and existing.get("config_hash") == config.config_hash:
            return existing

    train_loader = val_loader = None
    writer = None
    status: dict[str, Any] | None = None
    command = " ".join(sys.argv)
    try:
        device = resolve_device(config)
        train_loader, val_loader, summary = build_dataloaders(config)
        manifest_hash = summary["manifest_version_hash"]
        status = _status_template(config, run_dir, manifest_hash)
        if status_path.exists():
            prior = json.loads(status_path.read_text(encoding="utf-8"))
            status["created_at"] = prior.get("created_at", status["created_at"])
        atomic_write_json(run_dir / "config.json", config.as_dict())
        status.update(status="running", started_at=now_iso(), updated_at=now_iso())
        atomic_write_json(status_path, status)
        atomic_write_json(
            live_progress_path,
            {
                "run_id": run_id,
                "stage": "initializing",
                "epoch": 0,
                "total_epochs": config.epochs,
                "batch": 0,
                "total_batches": len(train_loader),
                "fraction": 0.0,
                "elapsed_seconds": 0.0,
                "estimated_remaining_seconds": None,
                "updated_at": now_iso(),
            },
        )
        update_ledger(_ledger_record(config, status, run_dir))
        update_current_status(run_id, f"Started run {run_id} after CUDA, manifest, and configuration validation.")
        append_journal(
            f"Run {run_id} started",
            {
                "Run ID": run_id,
                "Configuration hash": config.config_hash,
                "Architecture": config.architecture,
                "Preprocessing": config.preprocessing,
                "Dataset manifest": summary["manifest_version"],
                "Seed": config.seed,
                "Device/GPU": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
                "Epoch": f"0/{config.epochs}",
                "Status": "running",
            },
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model, model_metadata = build_model(config)
        model.to(device)
        output_shape = assert_output_contract(model, config.input_size, config.num_classes, device)
        atomic_write_json(run_dir / "model_metadata.json", {**model_metadata.__dict__, "output_shape_smoke": output_shape})
        criterion = build_loss(config, train_loader.dataset.rows, device)
        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(config, optimizer)
        scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
        if config.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
            except ImportError:
                writer = None

        start_epoch = 1
        history: list[dict[str, Any]] = []
        best_epoch = 0
        best_f1 = -math.inf
        if resume:
            checkpoint_path = run_dir / "last_checkpoint.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"resume requested but checkpoint is absent: {checkpoint_path}")
            checkpoint = _load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, config, device)
            start_epoch = int(checkpoint["epoch"]) + 1
            history = list(checkpoint.get("history", []))
            best_epoch = int(checkpoint.get("best_epoch", 0))
            best_f1 = float(checkpoint.get("best_validation_macro_f1", -math.inf))
            status.update(
                updated_at=now_iso(),
                current_epoch=start_epoch - 1,
                best_epoch=best_epoch,
                best_validation_macro_f1=best_f1,
            )
            atomic_write_json(status_path, status)
            update_ledger(_ledger_record(config, status, run_dir))
            append_journal(
                f"Run {run_id} resumed",
                {"Run ID": run_id, "Epoch": f"{start_epoch - 1}/{config.epochs}", "Checkpoint": checkpoint_path},
            )

        run_start = time.perf_counter()
        epochs_without_improvement = max(0, start_epoch - 1 - best_epoch)
        for epoch in range(start_epoch, config.epochs + 1):
            epoch_start = time.perf_counter()
            last_live_update = 0.0

            def report_batch(batch_index, total_batches, running_loss):
                nonlocal last_live_update
                now = time.perf_counter()
                if (
                    batch_index != 1
                    and batch_index != total_batches
                    and now - last_live_update < 2.0
                ):
                    return
                elapsed_live = now - epoch_start
                batches_per_second = batch_index / max(elapsed_live, 1e-9)
                remaining_live = (total_batches - batch_index) / max(batches_per_second, 1e-9)
                atomic_write_json(
                    live_progress_path,
                    {
                        "run_id": run_id,
                        "stage": "training",
                        "epoch": epoch,
                        "total_epochs": config.epochs,
                        "batch": batch_index,
                        "total_batches": total_batches,
                        "fraction": batch_index / max(1, total_batches),
                        "running_train_loss": running_loss,
                        "elapsed_seconds": elapsed_live,
                        "estimated_remaining_seconds": remaining_live,
                        "updated_at": now_iso(),
                    },
                )
                last_live_update = now

            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                amp=config.amp,
                progress_callback=report_batch,
            )
            atomic_write_json(
                live_progress_path,
                {
                    "run_id": run_id,
                    "stage": "validation",
                    "epoch": epoch,
                    "total_epochs": config.epochs,
                    "batch": len(train_loader),
                    "total_batches": len(train_loader),
                    "fraction": 1.0,
                    "running_train_loss": train_loss,
                    "elapsed_seconds": time.perf_counter() - epoch_start,
                    "estimated_remaining_seconds": None,
                    "updated_at": now_iso(),
                },
            )
            validation, _, _, _ = evaluate_model(
                model, val_loader, device, config.resolved_class_names, amp=config.amp
            )
            current_f1 = float(validation["macro_f1"])
            improved = current_f1 > best_f1 + config.minimum_delta
            if improved:
                best_f1 = current_f1
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if scheduler:
                if config.scheduler == "reduce_on_plateau":
                    scheduler.step(current_f1)
                else:
                    scheduler.step()
            elapsed = time.perf_counter() - epoch_start
            mean_epoch = (time.perf_counter() - run_start) / (epoch - start_epoch + 1)
            remaining = mean_epoch * (config.epochs - epoch)
            peak_memory = (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            )
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": validation["loss"],
                "val_accuracy": validation["accuracy"],
                "val_balanced_accuracy": validation["balanced_accuracy"],
                "val_macro_f1": current_f1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": remaining,
                "peak_gpu_memory_bytes": peak_memory,
                "best_epoch": best_epoch,
                "best_validation_macro_f1": best_f1,
                "train_loss_definition": "class-weighted training objective",
                "val_loss_definition": "unweighted cross entropy",
            }
            for class_name in config.resolved_class_names:
                key = "recall_" + class_name.lower().replace(" ", "_")
                row[key] = validation["per_class"][class_name]["recall"]
            history.append(row)
            if writer:
                writer.add_scalar("loss/train_weighted", train_loss, epoch)
                writer.add_scalar("loss/validation_unweighted", validation["loss"], epoch)
                writer.add_scalar("metrics/validation_accuracy", validation["accuracy"], epoch)
                writer.add_scalar("metrics/validation_balanced_accuracy", validation["balanced_accuracy"], epoch)
                writer.add_scalar("metrics/validation_macro_f1", current_f1, epoch)
                for class_name in config.resolved_class_names:
                    writer.add_scalar(
                        f"recall/{class_name.lower()}",
                        validation["per_class"][class_name]["recall"],
                        epoch,
                    )
                writer.add_scalar("optimization/learning_rate", optimizer.param_groups[0]["lr"], epoch)
                writer.add_scalar("resources/peak_gpu_memory_bytes", peak_memory, epoch)
                writer.add_scalar("timing/epoch_seconds", elapsed, epoch)
                writer.flush()
            write_history(run_dir / "history.csv", history)
            _save_checkpoint(
                run_dir / "last_checkpoint.pt", epoch=epoch, model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, config=config, history=history,
                best_epoch=best_epoch, best_f1=best_f1,
            )
            if improved:
                _save_checkpoint(
                    run_dir / "best_checkpoint.pt", epoch=epoch, model=model, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler, config=config, history=history,
                    best_epoch=best_epoch, best_f1=best_f1,
                )
                append_journal(
                    f"Run {run_id} new best checkpoint",
                    {"Run ID": run_id, "Epoch": f"{epoch}/{config.epochs}", "Best validation macro F1": best_f1,
                     "Best checkpoint": run_dir / "best_checkpoint.pt"},
                )
            if epoch % config.documentation_interval == 0:
                append_journal(
                    f"Run {run_id} epoch {epoch}",
                    {
                        "Run ID": run_id, "Configuration hash": config.config_hash,
                        "Architecture": config.architecture, "Preprocessing": config.preprocessing,
                        "Dataset manifest": summary["manifest_version"], "Seed": config.seed,
                        "Device/GPU": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
                        "Epoch": f"{epoch}/{config.epochs}", "Training loss": train_loss,
                        "Validation loss": validation["loss"], "Validation accuracy": validation["accuracy"],
                        "Validation balanced accuracy": validation["balanced_accuracy"],
                        "Validation macro F1": current_f1,
                        "Per-class recall": {name: validation["per_class"][name]["recall"] for name in config.resolved_class_names},
                        "Learning rate": optimizer.param_groups[0]["lr"], "Elapsed seconds": elapsed,
                        "Estimated remaining seconds": remaining, "Peak GPU memory bytes": peak_memory,
                        "Best epoch so far": best_epoch, "Best validation macro F1 so far": best_f1,
                        "Last checkpoint": run_dir / "last_checkpoint.pt", "Best checkpoint": run_dir / "best_checkpoint.pt",
                        "Status": "running",
                    },
                )
            status.update(
                updated_at=now_iso(), current_epoch=epoch, best_epoch=best_epoch,
                best_validation_macro_f1=best_f1,
            )
            atomic_write_json(status_path, status)
            atomic_write_json(
                live_progress_path,
                {
                    "run_id": run_id,
                    "stage": "epoch_completed",
                    "epoch": epoch,
                    "total_epochs": config.epochs,
                    "batch": len(train_loader),
                    "total_batches": len(train_loader),
                    "fraction": 1.0,
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": 0.0,
                    "validation_macro_f1": current_f1,
                    "updated_at": now_iso(),
                },
            )
            update_ledger(_ledger_record(config, status, run_dir))
            if improved:
                update_current_status(
                    run_id,
                    f"Run {run_id} reached a new best validation macro F1 of {best_f1:.6f} at epoch {epoch}.",
                )
            if config.controlled_stop_after_epoch == epoch:
                raise KeyboardInterrupt(f"controlled interruption after epoch {epoch}")
            if epochs_without_improvement >= config.early_stopping_patience:
                append_journal(
                    f"Run {run_id} early stopping",
                    {"Run ID": run_id, "Epoch": f"{epoch}/{config.epochs}", "Best epoch": best_epoch,
                     "Best validation macro F1": best_f1, "Status": "early_stopping"},
                )
                break

        best_checkpoint = torch.load(run_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        final_metrics, labels, probabilities, paths = evaluate_model(
            model, val_loader, device, config.resolved_class_names, amp=config.amp
        )
        inference = measure_inference(model, device, config.input_size, batch_size=1, warmup=5, runs=20)
        final_metrics.update(
            {
                "run_id": run_id, "configuration_hash": config.config_hash,
                "manifest_version": summary["manifest_version"], "best_epoch": best_epoch,
                "training_duration_seconds": time.perf_counter() - run_start,
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                "parameter_count": model_metadata.parameter_count,
                "trainable_parameter_count": model_metadata.trainable_parameter_count,
                "model_file_size_bytes": (run_dir / "best_checkpoint.pt").stat().st_size,
                "inference": inference,
            }
        )
        save_evaluation_artifacts(
            run_dir, final_metrics, labels, probabilities, paths,
            config.resolved_class_names, split="validation",
        )
        plot_training_history(history, run_dir / "training_curves.png")
        status.update(
            status="completed", updated_at=now_iso(), finished_at=now_iso(),
            current_epoch=history[-1]["epoch"], best_epoch=best_epoch,
            best_validation_macro_f1=best_f1,
        )
        atomic_write_json(status_path, status)
        atomic_write_json(
            live_progress_path,
            {
                "run_id": run_id,
                "stage": "completed",
                "epoch": status["current_epoch"],
                "total_epochs": config.epochs,
                "batch": len(train_loader),
                "total_batches": len(train_loader),
                "fraction": 1.0,
                "elapsed_seconds": 0.0,
                "estimated_remaining_seconds": 0.0,
                "updated_at": now_iso(),
            },
        )
        update_ledger(_ledger_record(config, status, run_dir))
        update_current_status(run_id, f"Completed run {run_id}; validation artifacts and both checkpoints were saved.")
        append_journal(
            f"Run {run_id} finished",
            {"Run ID": run_id, "Epoch": f"{status['current_epoch']}/{config.epochs}",
             "Validation macro F1": final_metrics["macro_f1"],
             "Validation balanced accuracy": final_metrics["balanced_accuracy"],
             "Per-class recall": {name: final_metrics["per_class"][name]["recall"] for name in config.resolved_class_names},
             "Best checkpoint": run_dir / "best_checkpoint.pt", "Status": "completed"},
        )
        if writer:
            writer.close()
            writer = None
        return status
    except KeyboardInterrupt as error:
        if status is not None:
            status.update(status="interrupted", updated_at=now_iso(), failure_reason="KeyboardInterrupt")
            atomic_write_json(status_path, status)
            atomic_write_json(
                live_progress_path,
                {
                    "run_id": run_id,
                    "stage": "interrupted",
                    "epoch": status.get("current_epoch", 0),
                    "total_epochs": config.epochs,
                    "fraction": 0.0,
                    "updated_at": now_iso(),
                },
            )
            update_ledger(_ledger_record(config, status, run_dir))
            update_current_status(run_id, f"Run {run_id} was interrupted after its latest persisted checkpoint.")
        append_failure(run_id, run_dir, "training", command, error)
        append_journal(f"Run {run_id} interrupted", {"Run ID": run_id, "Status": "interrupted", "Error": error})
        raise
    except BaseException as error:
        error_log = run_dir / "error.log"
        error_log.write_text(traceback.format_exc(), encoding="utf-8")
        if status is None:
            status = _status_template(config, run_dir, "unavailable")
        status.update(status="failed", updated_at=now_iso(), finished_at=now_iso(), failure_reason=str(error))
        atomic_write_json(status_path, status)
        atomic_write_json(
            live_progress_path,
            {
                "run_id": run_id,
                "stage": "failed",
                "epoch": status.get("current_epoch", 0),
                "total_epochs": config.epochs,
                "fraction": 0.0,
                "error": str(error),
                "updated_at": now_iso(),
            },
        )
        update_ledger(_ledger_record(config, status, run_dir))
        update_current_status(run_id, f"Run {run_id} failed; its configuration and traceback were preserved.")
        append_failure(run_id, run_dir, "training", command, error)
        append_journal(
            f"Run {run_id} failed",
            {"Run ID": run_id, "Configuration hash": config.config_hash, "Architecture": config.architecture,
             "Preprocessing": config.preprocessing, "Status": "failed", "Error": str(error), "Error log": error_log},
        )
        raise
    finally:
        if writer:
            writer.close()
