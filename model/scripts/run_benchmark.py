"""Run planned independent experiments sequentially, preserving failures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _multiclass_bootstrap  # noqa: F401
from multiclass.config import load_config
from multiclass.tracking import append_journal
from multiclass.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=Path("model/results/run_matrix.csv"))
    parser.add_argument("--config", type=Path, default=Path("model/configs/multiclass_baseline.json"))
    parser.add_argument("--stage", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage == "tuning":
        raise RuntimeError(
            "tuning rows must be run with model/scripts/run_tuning.py using tuning_plan.json"
        )
    if args.stage == "locked_test":
        raise RuntimeError(
            "locked test must be run with evaluate_multiclass.py and a frozen final_selection.json"
        )
    if args.stage == "robustness":
        expected = Path("model/configs/final_selected.json").resolve()
        if args.config.resolve() != expected:
            raise RuntimeError(
                "robustness requires --config model/configs/final_selected.json from the tuning resolver"
            )
    with args.matrix.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    failures = 0
    for row in rows:
        if args.stage and row["stage"] != args.stage:
            continue
        if row["stage"] in {"tuning", "locked_test"}:
            continue
        if row["status"] in {"unsupported", "skipped"}:
            continue
        overrides = {
            "run_id": row["run_id"],
            "architecture": row["architecture"],
            "preprocessing": row["preprocessing"],
            "seed": int(row["seed"]),
            "input_size": int(row["input_size"]),
            "batch_size": int(row["batch_size"]),
        }
        if row["stage"] == "smoke":
            overrides.update(
                {"smoke_test": True, "weights": "none", "epochs": 1,
                 "max_train_samples": 12, "max_val_samples": 12, "tensorboard": False}
            )
        config = load_config(args.config, overrides)
        append_journal(f"Run {row['run_id']} queued", {"Run ID": row["run_id"], "Status": "queued"})
        try:
            checkpoint = config.results_path / config.effective_run_id() / "last_checkpoint.pt"
            run_training(config, resume=args.resume and checkpoint.exists())
        except BaseException as error:
            failures += 1
            print(f"FAILED {row['run_id']}: {error}")
            continue
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
