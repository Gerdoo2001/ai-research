"""Train or resume one locked-manifest classification experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _multiclass_bootstrap  # noqa: F401
from multiclass.config import load_config
from multiclass.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("model/configs/multiclass_baseline.json"))
    parser.add_argument("--architecture")
    parser.add_argument("--preprocessing")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--weights", choices=["imagenet", "none"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--controlled-stop-after-epoch", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overrides = {
        "architecture": args.architecture,
        "preprocessing": args.preprocessing,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "input_size": args.input_size,
        "seed": args.seed,
        "device": args.device,
        "num_workers": args.num_workers,
        "run_id": args.run_id,
        "weights": args.weights,
        "controlled_stop_after_epoch": args.controlled_stop_after_epoch,
    }
    if args.allow_cpu_fallback:
        overrides["allow_cpu_fallback"] = True
    if args.smoke_test:
        architecture = args.architecture or "densenet121"
        preprocessing = args.preprocessing or "baseline"
        overrides.update(
            {
                "smoke_test": True,
                "weights": args.weights or "none",
                "epochs": args.epochs or 1,
                "batch_size": args.batch_size or 3,
                "max_train_samples": 12,
                "max_val_samples": 12,
                "tensorboard": False,
                "early_stopping_patience": 2,
                "run_id": args.run_id or f"smoke_{architecture}_{preprocessing}_s{args.seed or 42}",
            }
        )
    config_path = args.config.resolve()
    config = load_config(config_path, overrides)
    status = run_training(config, resume=args.resume)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
