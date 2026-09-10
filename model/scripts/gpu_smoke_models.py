"""Run isolated CUDA forward/backward/output-contract smokes for model factories."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import _multiclass_bootstrap  # noqa: F401
from multiclass.config import load_config
from multiclass.models import SUPPORTED_ARCHITECTURES, build_model
from multiclass.tracking import atomic_write_json, now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("model/configs/multiclass_baseline.json"))
    parser.add_argument("--architectures", nargs="*", default=sorted(SUPPORTED_ARCHITECTURES))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--output", type=Path, default=Path("model/results/gpu_model_smokes.json"))
    return parser.parse_args()


def main() -> int:
    import torch
    import torch.nn.functional as functional

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke suite")
    device = torch.device("cuda")
    results = []
    for architecture in args.architectures:
        started = time.perf_counter()
        result = {"architecture": architecture, "status": "failed"}
        model = optimizer = scaler = inputs = targets = logits = loss = None
        try:
            config = load_config(
                args.config,
                {
                    "architecture": architecture,
                    "weights": "none",
                    "input_size": args.input_size,
                    "batch_size": args.batch_size,
                    "device": "cuda",
                    "amp": True,
                },
            )
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model, metadata = build_model(config)
            model.to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler("cuda")
            inputs = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device)
            targets = torch.arange(args.batch_size, device=device) % config.num_classes
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(inputs)
                if tuple(logits.shape) != (args.batch_size, 3):
                    raise RuntimeError(f"unexpected output shape {tuple(logits.shape)}")
                loss = functional.cross_entropy(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            result.update(
                {
                    "status": "passed",
                    "output_shape": list(logits.shape),
                    "loss": float(loss.detach().cpu()),
                    "parameter_count": metadata.parameter_count,
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                }
            )
        except BaseException as error:
            result.update({"error_type": type(error).__name__, "error_message": str(error)})
        finally:
            result["elapsed_seconds"] = time.perf_counter() - started
            results.append(result)
            del model, optimizer, scaler, inputs, targets, logits, loss
            gc.collect()
            torch.cuda.empty_cache()
    payload = {
        "generated_at": now_iso(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "input_size": args.input_size,
        "batch_size": args.batch_size,
        "results": results,
    }
    output = args.output if args.output.is_absolute() else _multiclass_bootstrap.REPO_ROOT / args.output
    atomic_write_json(output, payload)
    print(json.dumps(payload, indent=2))
    return 1 if any(item["status"] != "passed" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
