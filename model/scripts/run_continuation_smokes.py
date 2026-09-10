"""Run tiny representative smokes for the audited continuation changes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import _multiclass_bootstrap
from multiclass.config import load_config
from multiclass.trainer import run_training


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    repo = _multiclass_bootstrap.REPO_ROOT
    manifests = repo / "model" / "data_manifests"
    baseline = repo / "model" / "configs" / "multiclass_baseline.json"
    before = {path.name: sha256(path) for path in sorted(manifests.glob("*manifest.csv"))}
    smoke_specs = [
        ("smoke_continuation_d169_clahe", "densenet169", "clahe"),
        ("smoke_continuation_d201_histogram", "densenet201", "histogram_equalization"),
        ("smoke_continuation_r50_viridis", "resnet50", "pseudo_color"),
    ]
    records = []
    for run_id, architecture, preprocessing in smoke_specs:
        config = load_config(
            baseline,
            {
                "run_id": run_id,
                "architecture": architecture,
                "preprocessing": preprocessing,
                "weights": "none",
                "batch_size": 3,
                "epochs": 1,
                "max_train_samples": 12,
                "max_val_samples": 12,
                "tensorboard": True,
                "smoke_test": True,
                "early_stopping_patience": 1,
            },
        )
        status = run_training(config, resume=False)
        run_dir = config.results_path / run_id
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        event_files = sorted((run_dir / "tensorboard").glob("events.out.tfevents.*"))
        if not event_files:
            raise RuntimeError(f"{run_id} did not create a TensorBoard event file")
        accumulator = EventAccumulator(str(run_dir / "tensorboard"))
        accumulator.Reload()
        scalar_tags = sorted(accumulator.Tags().get("scalars", []))
        required = {
            "loss/train_weighted",
            "loss/validation_unweighted",
            "metrics/validation_macro_f1",
            "recall/normal",
            "recall/viral",
            "recall/bacterial",
            "optimization/learning_rate",
        }
        missing = sorted(required - set(scalar_tags))
        if missing:
            raise RuntimeError(f"{run_id} is missing TensorBoard scalars: {missing}")
        records.append(
            {
                "run_id": run_id,
                "architecture": architecture,
                "preprocessing": preprocessing,
                "status": status["status"],
                "prediction_rows": sum(1 for _ in (run_dir / "predictions.csv").open(encoding="utf-8")) - 1,
                "tensorboard_scalar_tags": scalar_tags,
            }
        )
    after = {path.name: sha256(path) for path in sorted(manifests.glob("*manifest.csv"))}
    if before != after:
        raise RuntimeError("locked manifest hashes changed during continuation smokes")
    output = repo / "model" / "results" / "audits" / "continuation_smoke_audit.json"
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "locked_manifest_hashes_unchanged": True,
                "test_manifest_loaded": False,
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
