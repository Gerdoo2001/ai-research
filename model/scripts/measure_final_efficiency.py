"""Measure post-freeze end-to-end latency without changing final predictions."""

from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

import _multiclass_bootstrap
from multiclass.config import ExperimentConfig
from multiclass.models import build_model
from multiclass.preprocessing import build_transform


def main() -> int:
    import numpy as np
    import torch
    from PIL import Image

    repo = _multiclass_bootstrap.REPO_ROOT
    results = repo / "model/results"
    selection = json.loads((results / "final_selection.json").read_text(encoding="utf-8"))
    if selection.get("status") != "locked_test_evaluated":
        raise RuntimeError("efficiency measurement is post-freeze and requires a completed locked-test evaluation")
    source_run = Path(selection["selected_run_dir"])
    config = ExperimentConfig(**json.loads((source_run / "config.json").read_text(encoding="utf-8")))
    checkpoint_path = Path(selection["selected_checkpoint"])
    device = torch.device("cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, metadata = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    transform = build_transform(config, training=False)
    source_root = Path(
        json.loads((repo / "model/data_manifests/dataset_source_snapshot.json").read_text(encoding="utf-8"))["source_root"]
    )
    with (results / "runs/final_locked_test/predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected_rows = rows[: min(200, len(rows))]
    if not selected_rows:
        raise RuntimeError("locked-test predictions are empty")

    first_image = Image.open(source_root / Path(selected_rows[0]["relative_path"])).convert("RGB")
    first_tensor = transform(first_image).unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(10):
            model(first_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    latencies_ms = []
    with torch.inference_mode():
        for row in selected_rows:
            started = time.perf_counter()
            with Image.open(source_root / Path(row["relative_path"])) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
            probabilities = torch.softmax(model(tensor), dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            _ = int(probabilities.argmax(dim=1).item())
            latencies_ms.append((time.perf_counter() - started) * 1000.0)

    parent_metrics = json.loads((source_run / "metrics.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "selection_status": selection["status"],
        "source_run_id": selection["selected_run_id"],
        "architecture": config.architecture,
        "preprocessing": config.preprocessing,
        "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "device": str(device),
        "measured_images": len(latencies_ms),
        "warmup_runs": 10,
        "end_to_end_scope": "disk image open/decode, deterministic preprocessing, host-to-device transfer, forward pass, softmax, synchronization, and argmax; batch size 1",
        "end_to_end_latency_ms": {
            "mean": statistics.mean(latencies_ms),
            "median": statistics.median(latencies_ms),
            "p95": float(np.quantile(latencies_ms, 0.95)),
            "minimum": min(latencies_ms),
            "maximum": max(latencies_ms),
        },
        "end_to_end_throughput_images_per_second_from_mean": 1000.0 / statistics.mean(latencies_ms),
        "previous_compute_only_measurement": parent_metrics.get("inference"),
        "parameter_count": metadata.parameter_count,
        "trainable_parameter_count": metadata.trainable_parameter_count,
        "checkpoint_file_size_bytes": checkpoint_path.stat().st_size,
        "peak_training_gpu_memory_bytes": parent_metrics.get("peak_gpu_memory_bytes"),
        "interpretation_boundary": "This is a local single-device engineering measurement, not a clinical deployment benchmark.",
    }
    output = results / "efficiency"
    output.mkdir(parents=True, exist_ok=True)
    (output / "final_efficiency.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (output / "end_to_end_latency_samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "latency_ms"])
        writer.writeheader()
        writer.writerows({"index": index, "latency_ms": value} for index, value in enumerate(latencies_ms, 1))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
