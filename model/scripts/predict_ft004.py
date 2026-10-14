"""Run the frozen FT-004 research ensemble on one supplied or synthetic image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _multiclass_bootstrap  # noqa: F401
from multiclass.config import REPO_ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=REPO_ROOT / "model/artifacts/ft004/selection.json")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path, help="path to your own chest X-ray image")
    inputs.add_argument("--synthetic", action="store_true", help="execution check using a generated gray image; no medical meaning")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        raise ValueError("cpu thread count must be positive")
    import torch
    from PIL import Image
    from multiclass.inference import FT004EnsemblePredictor

    torch.set_num_threads(args.cpu_threads)
    predictor = FT004EnsemblePredictor(args.selection, device=args.device)
    if args.synthetic:
        image = Image.new("RGB", (256, 320), (128, 128, 128))
        output = predictor.predict(image)
    else:
        with Image.open(args.image) as original:
            output = predictor.predict(original.copy())
    output.update({"execution_check_only": args.synthetic,
                   "input": "generated uniform gray image" if args.synthetic else str(args.image.resolve()),
                   "device": str(predictor.members[0][1].device),
                   "clinical_interpretation": "research prototype output; not a medical diagnosis"})
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
