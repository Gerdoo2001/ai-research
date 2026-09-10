"""Evaluate a completed checkpoint on validation or explicitly unlocked test data."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import _multiclass_bootstrap  # noqa: F401
from multiclass.config import ExperimentConfig, REPO_ROOT
from multiclass.data import build_evaluation_loader
from multiclass.evaluation import evaluate_model, save_evaluation_artifacts
from multiclass.models import build_model
from multiclass.tracking import append_journal, atomic_write_json, now_iso, update_current_status, update_ledger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--allow-locked-test", action="store_true")
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument(
        "--final-selection",
        type=Path,
        help="Frozen validation-only selection JSON required for the one final locked-test evaluation",
    )
    return parser.parse_args()


def main() -> int:
    import torch

    args = parse_args()
    run_dir = args.run_dir.resolve()
    selection = None
    if args.split == "test":
        if not args.allow_locked_test or args.final_selection is None:
            raise RuntimeError(
                "locked-test evaluation requires both --allow-locked-test and --final-selection"
            )
        selection_path = args.final_selection.resolve()
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("status") != "selection_frozen_before_test":
            raise RuntimeError("final selection is not frozen before test")
        if Path(selection["selected_run_dir"]).resolve() != run_dir:
            raise RuntimeError("--run-dir does not match the frozen representative run")
    config = ExperimentConfig(**json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    device = torch.device("cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint_path = run_dir / f"{args.checkpoint}_checkpoint.pt"
    if selection is not None:
        if checkpoint_path.resolve() != Path(selection["selected_checkpoint"]).resolve():
            raise RuntimeError("checkpoint does not match the frozen final selection")
        if config.config_hash != selection["configuration_hash"]:
            raise RuntimeError("configuration hash does not match the frozen final selection")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["config_hash"] != config.config_hash:
        raise RuntimeError("checkpoint/config hash mismatch")
    model, _ = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    loader = build_evaluation_loader(config, args.split, allow_locked_test=args.allow_locked_test)
    metrics, labels, probabilities, paths = evaluate_model(
        model, loader, device, config.resolved_class_names, amp=config.amp
    )
    output_dir = run_dir if args.split == "val" else REPO_ROOT / "model/results/runs/final_locked_test"
    if args.split == "test":
        parent_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        metrics.update(
            {
                "run_id": "final_locked_test",
                "source_run_id": config.effective_run_id(),
                "configuration_hash": config.config_hash,
                "manifest_hash": selection["manifest_hash"],
                "best_epoch": checkpoint["epoch"],
                "training_duration_seconds": parent_metrics.get("training_duration_seconds"),
                "parameter_count": parent_metrics.get("parameter_count"),
                "trainable_parameter_count": parent_metrics.get("trainable_parameter_count"),
                "model_file_size_bytes": parent_metrics.get("model_file_size_bytes"),
                "peak_gpu_memory_bytes": parent_metrics.get("peak_gpu_memory_bytes"),
                "inference": parent_metrics.get("inference"),
            }
        )
    save_evaluation_artifacts(
        output_dir, metrics, labels, probabilities, paths,
        config.resolved_class_names, split="validation" if args.split == "val" else "locked_test",
    )
    if args.split == "test":
        evaluation_config = config.as_dict()
        evaluation_config["run_id"] = "final_locked_test"
        atomic_write_json(output_dir / "config.json", evaluation_config)
        for name in ("model_metadata.json", "training_curves.png"):
            source = run_dir / name
            if source.exists():
                shutil.copy2(source, output_dir / name)
        timestamp = now_iso()
        status = {
            "run_id": "final_locked_test",
            "status": "completed",
            "created_at": timestamp,
            "started_at": timestamp,
            "updated_at": timestamp,
            "finished_at": timestamp,
            "current_epoch": checkpoint["epoch"],
            "total_epochs": checkpoint["epoch"],
            "best_epoch": checkpoint["epoch"],
            "best_validation_macro_f1": None,
            "config_hash": config.config_hash,
            "manifest_hash": selection["manifest_hash"],
            "source_run_id": config.effective_run_id(),
            "best_checkpoint": str(checkpoint_path),
            "last_checkpoint": str(run_dir / "last_checkpoint.pt"),
            "failure_reason": None,
            "resume_command": "not applicable; locked test evaluation completed once",
        }
        atomic_write_json(output_dir / "status.json", status)
        selection.update(status="locked_test_evaluated", tested_at=timestamp, test_metrics=metrics)
        atomic_write_json(args.final_selection.resolve(), selection)
        update_ledger(
            {
                "run_id": "final_locked_test",
                "config_hash": config.config_hash,
                "architecture": config.architecture,
                "preprocessing": config.preprocessing,
                "task_mode": config.task_mode,
                "seed": config.seed,
                "manifest_hash": selection["manifest_hash"],
                "status": "completed",
                "created_at": timestamp,
                "started_at": timestamp,
                "updated_at": timestamp,
                "finished_at": timestamp,
                "best_epoch": checkpoint["epoch"],
                "best_validation_macro_f1": "",
                "run_dir": str(output_dir),
                "failure_reason": "",
                "resume_command": "not applicable; locked test evaluation completed once",
            }
        )
        update_current_status("final_locked_test", "Completed the single frozen locked-test evaluation.")
    append_journal(
        f"Run {'final_locked_test' if args.split == 'test' else config.effective_run_id()} "
        f"{'locked test' if args.split == 'test' else 'validation'} evaluation",
        {
            "Run ID": "final_locked_test" if args.split == "test" else config.effective_run_id(),
            "Source run ID": config.effective_run_id() if args.split == "test" else "same run",
            "Configuration hash": config.config_hash,
            "Epoch": checkpoint["epoch"],
            "Validation macro F1" if args.split == "val" else "Locked test macro F1": metrics["macro_f1"],
            "Balanced accuracy": metrics["balanced_accuracy"],
            "Per-class recall": {name: metrics["per_class"][name]["recall"] for name in config.resolved_class_names},
            "Checkpoint": checkpoint_path,
            "Status": "evaluation_complete",
        },
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
