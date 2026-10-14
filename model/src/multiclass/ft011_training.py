"""FT-011: historical training semantics with two rates and batch-level recovery."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

from .data import build_dataloaders
from .evaluation import measure_inference, plot_training_history, save_evaluation_artifacts
from .metrics import calculate_metrics
from .models import assert_output_contract, build_model
from .tracking import atomic_write_json, now_iso, write_history
from .trainer import build_loss, build_scheduler, resolve_device, set_deterministic_seed


class ProtectiveStop(RuntimeError):
    pass


class ReplayBatchSampler:
    """Retain the original RandomSampler permutation and the next unconsumed batch.

    Materialization uses the original sampler's generator, after DataLoader has
    consumed its base seed. No new RNG is introduced. num_workers=0 is required.
    """

    def __init__(self, original):
        self.original = original
        self.order = None
        self.next_batch = 0

    def __iter__(self):
        if self.order is None:
            self.order = list(self.original.sampler)
        size = self.original.batch_size
        for batch in range(self.next_batch, self.total_batches):
            selected = self.order[batch * size:(batch + 1) * size]
            if len(selected) == size or not self.original.drop_last:
                yield selected

    @property
    def total_batches(self):
        count = len(self.original.sampler)
        size = self.original.batch_size
        return count // size if self.original.drop_last else math.ceil(count / size)

    def __len__(self):
        return max(0, self.total_batches - self.next_batch)

    def reset(self):
        self.order = None
        self.next_batch = 0

    def state_dict(self):
        return {"order": self.order, "next_batch": self.next_batch}

    def load_state_dict(self, value):
        self.order = value["order"]
        self.next_batch = value["next_batch"]
        if self.order is not None and sorted(self.order) != list(range(len(self.original.sampler))):
            raise RuntimeError("resumable sample order is not a complete permutation")


def differential_optimizer(model, backbone_rate=1e-5, classifier_rate=1e-4):
    import torch

    named = dict(model.named_parameters())
    classifier = [(name, p) for name, p in named.items() if name.startswith("classifier.")]
    backbone = [(name, p) for name, p in named.items() if not name.startswith("classifier.")]
    if not classifier or not backbone or not all(p.requires_grad for p in named.values()):
        raise RuntimeError("FT-011 requires a fully trainable DenseNet with a classifier")
    head_ids = {id(p) for _, p in classifier}
    backbone_ids = {id(p) for _, p in backbone}
    if head_ids & backbone_ids or head_ids | backbone_ids != {id(p) for p in named.values()}:
        raise RuntimeError("optimizer groups are not disjoint and exhaustive")
    groups = [
        {"params": [p for _, p in backbone], "lr": backbone_rate, "name": "backbone"},
        {"params": [p for _, p in classifier], "lr": classifier_rate, "name": "classifier"},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    audit = {
        "disjoint": True, "exhaustive": True,
        "backbone_parameter_names": [n for n, _ in backbone],
        "classifier_parameter_names": [n for n, _ in classifier],
        "backbone_parameters": sum(p.numel() for _, p in backbone),
        "classifier_parameters": sum(p.numel() for _, p in classifier),
        "initial_learning_rates": [backbone_rate, classifier_rate],
    }
    return optimizer, audit


def tensor_state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def capture_rng(generator):
    import torch

    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
        "data_generator": generator.get_state(),
    }


def restore_rng(value, generator):
    import torch

    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in value["cuda"]])
    generator.set_state(value["data_generator"].cpu())


def atomic_torch_save(payload, path):
    import torch

    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(10):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            time.sleep(0.2 * (attempt + 1))
    raise RuntimeError(f"atomic checkpoint replacement failed: {path}")


def run_ft011(config, run_dir, watchdog, *, protocol_hash, resume=False, smoke=False):
    import torch
    import torch.nn.functional as functional

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    if status_path.exists() and not resume:
        raise RuntimeError("existing FT-011 run requires verified completion reuse or explicit resume")
    if config.num_workers != 0 or config.freeze_backbone or config.batch_size != 8:
        raise RuntimeError("FT-011 runtime/configuration does not match FT-010")
    set_deterministic_seed(config.seed)
    device = resolve_device(config)
    train_loader, val_loader, manifest = build_dataloaders(config)
    for loader, expected in [(train_loader, "train"), (val_loader, "val")]:
        if any(row["assigned_split"] != expected for row in loader.dataset.rows):
            raise RuntimeError("loader includes an incorrect assigned_split")
        sampler = ReplayBatchSampler(loader.batch_sampler)
        object.__setattr__(loader, "batch_sampler", sampler)
    generator = train_loader.generator
    if generator is not val_loader.generator:
        raise RuntimeError("historical shared data-loader generator is missing")
    model, metadata = build_model(config)
    initial = {
        "model_state_sha256": tensor_state_hash(model.state_dict()),
        "classifier_state_sha256": tensor_state_hash(model.classifier.state_dict()),
        "fresh_imagenet_initialization": True, "seed": config.seed,
    }
    model.to(device)
    assert_output_contract(model, config.input_size, config.num_classes, device)
    criterion = build_loss(config, train_loader.dataset.rows, device)
    atomic_write_json(run_dir / "class_weights.json", {
        "class_order": list(config.resolved_class_names),
        "balanced_class_weights": criterion.weight.detach().cpu().tolist(),
        "label_smoothing": config.label_smoothing,
        "definition": "N / (K * per-class training count)",
    })
    optimizer, groups = differential_optimizer(model)
    scheduler = build_scheduler(config, optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
    writer = None
    if config.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
        except ImportError:
            writer = None
    atomic_write_json(run_dir / "initialization.json", initial)
    atomic_write_json(run_dir / "optimizer_groups.json", groups)
    atomic_write_json(run_dir / "scheduler_initial.json", scheduler.state_dict())
    atomic_write_json(run_dir / "config.json", config.as_dict())
    atomic_write_json(run_dir / "model_metadata.json", metadata.__dict__)
    state = {
        "epoch": 1, "phase": "train", "history": [], "best_epoch": 0,
        "best_f1": -math.inf, "epochs_without_improvement": 0,
        "total_loss": 0.0, "seen": 0, "pending_batch": None,
        "validation": {"loss_sum": 0.0, "labels": [], "probabilities": [], "paths": []},
        "active_seconds": 0.0, "epoch_seconds": 0.0, "interruptions": [],
    }
    checkpoint_path = run_dir / "last_checkpoint.pt"
    wall_start = time.time()
    active_start = time.perf_counter()
    epoch_segment_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    def persist_status(stage):
        sampler = train_loader.batch_sampler if state["phase"] == "train" else val_loader.batch_sampler
        watchdog.update_progress(seed=config.seed, epoch=state["epoch"], batch=sampler.next_batch,
            total_batches=sampler.total_batches, phase=stage + ":" + state["phase"],
            elapsed_seconds=state["active_seconds"] + time.perf_counter() - active_start,
            backbone_lr=optimizer.param_groups[0]["lr"], classifier_lr=optimizer.param_groups[1]["lr"])
        atomic_write_json(status_path, {
            "status": stage, "run_id": run_dir.name, "seed": config.seed,
            "epoch": state["epoch"], "phase": state["phase"],
            "batch": sampler.next_batch, "total_batches": sampler.total_batches,
            "updated_at": now_iso(), "best_epoch": state["best_epoch"],
            "best_validation_macro_f1": state["best_f1"] if math.isfinite(state["best_f1"]) else None,
            "elapsed_seconds": state["active_seconds"] + time.perf_counter() - active_start,
            "learning_rates": {g["name"]: g["lr"] for g in optimizer.param_groups},
            "checkpoint": str(checkpoint_path), "protocol_hash": protocol_hash,
            "smoke_execution_only": smoke,
        })

    def save(path=checkpoint_path):
        snapshot = copy.deepcopy(state)
        snapshot["active_seconds"] += time.perf_counter() - active_start
        payload = {
            "schema_version": 1, "protocol_hash": protocol_hash,
            "config": config.as_dict(), "config_hash": config.config_hash,
            "manifest_hash": manifest["manifest_version_hash"],
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
            "rng_state": capture_rng(generator), "state": snapshot,
            "train_sampler": train_loader.batch_sampler.state_dict(),
            "val_sampler": val_loader.batch_sampler.state_dict(),
        }
        atomic_torch_save(payload, path)

    def load(path=checkpoint_path):
        nonlocal state, active_start
        cp = torch.load(path, map_location="cpu", weights_only=False)
        required = {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict",
                    "rng_state", "state", "train_sampler", "val_sampler", "protocol_hash", "manifest_hash"}
        if not required <= cp.keys() or cp["protocol_hash"] != protocol_hash:
            raise RuntimeError("checkpoint is incomplete or belongs to another FT-011 protocol")
        if cp["config_hash"] != config.config_hash or cp["manifest_hash"] != manifest["manifest_version_hash"]:
            raise RuntimeError("checkpoint configuration/manifest mismatch")
        model.load_state_dict(cp["model_state_dict"])
        optimizer.load_state_dict(cp["optimizer_state_dict"])
        scheduler.load_state_dict(cp["scheduler_state_dict"])
        scaler.load_state_dict(cp["scaler_state_dict"])
        train_loader.batch_sampler.load_state_dict(cp["train_sampler"])
        val_loader.batch_sampler.load_state_dict(cp["val_sampler"])
        state = cp["state"]
        restore_rng(cp["rng_state"], generator)
        active_start = time.perf_counter()

    def guard():
        nonlocal active_start, epoch_segment_started
        decision = watchdog.poll()
        if decision["action"] == "continue":
            return
        event = {"at": now_iso(), "phase": state["phase"], "epoch": state["epoch"],
                 "decision": decision, "hardware": watchdog.status}
        state["interruptions"].append(event)
        state["epoch_seconds"] += time.perf_counter() - epoch_segment_started
        save()
        persist_status("protective_pause")
        if decision["action"] == "stop":
            raise ProtectiveStop(str(decision))
        watchdog.begin_cooling(checkpoint_path)
        # Restore every serialized component before permitting a cooling resume.
        load()
        while True:
            decision = watchdog.poll_resume(checkpoint_verified=True)
            if decision["action"] == "resume":
                active_start = time.perf_counter()
                epoch_segment_started = time.perf_counter()
                persist_status("running")
                return
            if decision["action"] == "stop":
                raise ProtectiveStop(str(decision))
            persist_status("cooling")
            time.sleep(2)

    def iterator(loader, resumed=False):
        saved = generator.get_state() if resumed else None
        result = iter(loader)
        if resumed:
            # Reconstructing an iterator must not consume an extra base seed.
            generator.set_state(saved)
        return result

    if resume:
        load()
    persist_status("running")
    try:
        while state["epoch"] <= config.epochs:
            epoch_started = time.perf_counter()
            epoch_segment_started = epoch_started
            if state["phase"] == "train":
                model.train()
                it = iterator(train_loader, resumed=bool(train_loader.batch_sampler.order))
                used_rates = [g["lr"] for g in optimizer.param_groups]
                while train_loader.batch_sampler.next_batch < train_loader.batch_sampler.total_batches:
                    guard()
                    if state["pending_batch"] is None:
                        state["pending_batch"] = next(it)
                    guard()
                    inputs, targets, _ = state["pending_batch"]
                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=config.amp):
                        loss = criterion(model(inputs), targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    state["total_loss"] += float(loss.detach().cpu()) * len(inputs)
                    state["seen"] += len(inputs)
                    train_loader.batch_sampler.next_batch += 1
                    state["pending_batch"] = None
                    del loss, inputs, targets
                    persist_status("running")
                state["used_rates"] = used_rates
                state["phase"] = "validation"
                val_loader.batch_sampler.reset()
            if state["phase"] == "validation":
                model.eval()
                it = iterator(val_loader, resumed=bool(val_loader.batch_sampler.order))
                with torch.inference_mode():
                    while val_loader.batch_sampler.next_batch < val_loader.batch_sampler.total_batches:
                        guard()
                        if state["pending_batch"] is None:
                            state["pending_batch"] = next(it)
                        guard()
                        inputs, targets, paths = state["pending_batch"]
                        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                        with torch.amp.autocast("cuda", enabled=config.amp):
                            logits = model(inputs)
                            loss = functional.cross_entropy(logits, targets)
                        v = state["validation"]
                        v["loss_sum"] += float(loss.detach().cpu()) * len(inputs)
                        v["labels"].extend(targets.detach().cpu().tolist())
                        v["probabilities"].extend(torch.softmax(logits, dim=1).detach().cpu().numpy())
                        v["paths"].extend(paths)
                        val_loader.batch_sampler.next_batch += 1
                        state["pending_batch"] = None
                        del logits, loss, inputs, targets
                        persist_status("running")
                v = state["validation"]
                validation = calculate_metrics(np.asarray(v["labels"]), np.asarray(v["probabilities"], dtype=np.float64), config.resolved_class_names)
                validation["loss"] = v["loss_sum"] / len(v["labels"])
                score = validation["macro_f1"]
                improved = score > state["best_f1"] + config.minimum_delta
                if improved:
                    state["best_f1"], state["best_epoch"] = score, state["epoch"]
                    state["epochs_without_improvement"] = 0
                else:
                    state["epochs_without_improvement"] += 1
                scheduler.step(score)
                row = {
                    "epoch": state["epoch"], "train_loss": state["total_loss"] / state["seen"],
                    "val_loss": validation["loss"], "val_accuracy": validation["accuracy"],
                    "val_balanced_accuracy": validation["balanced_accuracy"], "val_macro_f1": score,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "backbone_lr_used": state["used_rates"][0], "classifier_lr_used": state["used_rates"][1],
                    "backbone_lr_next": optimizer.param_groups[0]["lr"], "classifier_lr_next": optimizer.param_groups[1]["lr"],
                    "elapsed_seconds": state["epoch_seconds"] + time.perf_counter() - epoch_segment_started,
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
                    "best_epoch": state["best_epoch"], "best_validation_macro_f1": state["best_f1"],
                }
                row.update({"recall_" + n.lower(): validation["per_class"][n]["recall"] for n in config.resolved_class_names})
                state["history"].append(row)
                if writer:
                    for key in ("train_loss", "val_loss", "val_accuracy", "val_balanced_accuracy", "val_macro_f1",
                                "backbone_lr_used", "classifier_lr_used"):
                        writer.add_scalar(key, row[key], state["epoch"])
                    writer.flush()
                write_history(run_dir / "history.csv", state["history"])
                state["phase"] = "epoch_completed"
                save()
                if improved:
                    save(run_dir / "best_checkpoint.pt")
                    # These predictions are already evaluated with the selected checkpoint.
                    save_evaluation_artifacts(run_dir, validation, np.asarray(v["labels"]),
                        np.asarray(v["probabilities"], dtype=np.float64), v["paths"], config.resolved_class_names, split="validation")
            if state["epochs_without_improvement"] >= config.early_stopping_patience or state["epoch"] >= config.epochs:
                break
            state.update(epoch=state["epoch"] + 1, phase="train", total_loss=0.0, seen=0, epoch_seconds=0.0,
                         validation={"loss_sum": 0.0, "labels": [], "probabilities": [], "paths": []})
            train_loader.batch_sampler.reset()
            val_loader.batch_sampler.reset()
        state["phase"] = "inference"
        guard()
        best = torch.load(run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(best["model_state_dict"])
        model.eval()
        # Preserve the historical full-float, batch-one inference measurement.
        timing = []
        inputs = torch.zeros(1, 3, config.input_size, config.input_size, device=device)
        with torch.inference_mode():
            for index in range(25):
                guard()
                torch.cuda.synchronize()
                started = time.perf_counter()
                model(inputs)
                torch.cuda.synchronize()
                if index >= 5:
                    timing.append(time.perf_counter() - started)
        metrics_path = run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.update({"run_id": run_dir.name, "configuration_hash": config.config_hash,
            "protocol_hash": protocol_hash, "best_epoch": state["best_epoch"],
            "training_duration_seconds": state["active_seconds"] + time.perf_counter() - active_start,
            "wall_duration_seconds": time.time() - wall_start,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
            "parameter_count": metadata.parameter_count, "trainable_parameter_count": metadata.trainable_parameter_count,
            "model_file_size_bytes": (run_dir / "best_checkpoint.pt").stat().st_size,
            "inference": {"latency_ms_per_image": float(np.mean(timing) * 1000),
                "throughput_images_per_second": float(1 / np.mean(timing)), "batch_size": 1,
                "preprocessing_included": False, "warmup_runs": 5, "measured_runs": 20},
            "interruptions": state["interruptions"], "smoke_execution_only": smoke})
        atomic_write_json(metrics_path, metrics)
        plot_training_history(state["history"], run_dir / "training_curves.png")
        state["phase"] = "completed"
        persist_status("completed")
        return metrics
    except BaseException as exc:
        # Preserve the exact pending transformed batch as well as all RNG/sampler state.
        save()
        persist_status("protective_stopped" if isinstance(exc, ProtectiveStop) else "interrupted")
        atomic_write_json(run_dir / "interruption.json", {"at": now_iso(), "type": type(exc).__name__, "reason": str(exc)})
        raise
    finally:
        if writer:
            writer.close()
