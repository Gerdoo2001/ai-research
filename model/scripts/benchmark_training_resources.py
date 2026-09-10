"""Measure CUDA training throughput/memory and estimate baseline resources."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import _multiclass_bootstrap
from multiclass.config import load_config
from multiclass.models import build_model
from multiclass.tracking import atomic_write_json, now_iso


BATCH_SIZES = {
    "densenet121": 16,
    "densenet169": 12,
    "densenet201": 8,
    "resnet50": 16,
    "resnet101": 8,
    "resnet152": 6,
    "yolov5_cls": 32,
    "yolov8_cls": 32,
}
TRAIN_IMAGES = 4076
VALIDATION_IMAGES = 874
BASELINE_EPOCHS = 12


def main() -> int:
    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for resource calibration")
    repo = _multiclass_bootstrap.REPO_ROOT
    config_path = repo / "model/configs/multiclass_baseline.json"
    device = torch.device("cuda")
    results = []
    for architecture, batch_size in BATCH_SIZES.items():
        model = optimizer = scaler = inputs = targets = logits = loss = None
        record = {"architecture": architecture, "batch_size": batch_size, "status": "failed"}
        try:
            config = load_config(
                config_path,
                {"architecture": architecture, "weights": "none", "batch_size": batch_size, "device": "cuda"},
            )
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.cuda.empty_cache()
            model, metadata = build_model(config)
            model.to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scaler = torch.amp.GradScaler("cuda")
            inputs = torch.randn(batch_size, 3, 224, 224, device=device)
            targets = torch.arange(batch_size, device=device) % 3

            def step() -> None:
                nonlocal logits, loss
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    logits = model(inputs)
                    loss = functional.cross_entropy(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            for _ in range(3):
                step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            measured_steps = 8
            started = time.perf_counter()
            for _ in range(measured_steps):
                step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            images_per_second = measured_steps * batch_size / elapsed
            train_epoch = TRAIN_IMAGES / images_per_second
            validation_epoch = VALIDATION_IMAGES / (images_per_second * 2.0)
            checkpoint_estimate = int(metadata.parameter_count * 12.2)
            record.update(
                {
                    "status": "passed",
                    "parameter_count": metadata.parameter_count,
                    "measured_steps": measured_steps,
                    "measured_seconds": elapsed,
                    "training_images_per_second_compute_only": images_per_second,
                    "peak_gpu_memory_bytes_measured_steps": int(torch.cuda.max_memory_allocated(device)),
                    "estimated_train_epoch_compute_seconds": train_epoch,
                    "estimated_validation_epoch_compute_seconds": validation_epoch,
                    "estimated_12_epoch_compute_seconds": (train_epoch + validation_epoch) * BASELINE_EPOCHS,
                    "estimated_checkpoint_bytes_each": checkpoint_estimate,
                    "estimated_run_storage_bytes": checkpoint_estimate * 2 + 5_000_000,
                }
            )
        except BaseException as error:
            record.update({"error_type": type(error).__name__, "error_message": str(error)})
        finally:
            results.append(record)
            del model, optimizer, scaler, inputs, targets, logits, loss
            gc.collect()
            torch.cuda.empty_cache()
    completed = [item for item in results if item["status"] == "passed"]
    payload = {
        "generated_at": now_iso(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "input_size": 224,
        "amp": True,
        "optimizer": "AdamW",
        "warmup_steps": 3,
        "measured_steps": 8,
        "preprocessing_and_disk_io_included": False,
        "estimate_warning": (
            "Compute-only synthetic measurements are lower bounds. End-to-end duration also includes "
            "image decoding, preprocessing, checkpointing, metrics, plotting, and Windows overhead."
        ),
        "train_images": TRAIN_IMAGES,
        "validation_images": VALIDATION_IMAGES,
        "baseline_epochs": BASELINE_EPOCHS,
        "results": results,
        "estimated_baseline_matrix_compute_seconds": sum(
            item["estimated_12_epoch_compute_seconds"] for item in completed
        ),
        "estimated_baseline_matrix_storage_bytes": sum(
            item["estimated_run_storage_bytes"] for item in completed
        ),
    }
    output = repo / "model/results/training_resource_estimate.json"
    atomic_write_json(output, payload)
    print(json.dumps(payload, indent=2))
    return 1 if len(completed) != len(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
