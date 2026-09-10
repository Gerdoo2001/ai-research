"""Generate post-freeze Grad-CAM examples from the single locked-test evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import _multiclass_bootstrap
from multiclass.config import ExperimentConfig
from multiclass.models import build_model
from multiclass.preprocessing import build_transform, visual_preprocess


CLASS_NAMES = ("Normal", "Viral", "Bacterial")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("model/results/final_selection.json"))
    return parser.parse_args()


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def target_layer(model, architecture: str):
    if architecture.startswith("densenet"):
        return model.features.norm5
    if architecture.startswith("resnet"):
        return model.layer4[-1]
    if architecture in {"yolov5_cls", "yolov8_cls"}:
        return model.wrapped.model[-2]
    raise ValueError(f"Grad-CAM target layer is not defined for {architecture}")


def select_examples(rows):
    selected = []
    for class_name in CLASS_NAMES:
        correct = [row for row in rows if row["true_class"] == class_name and row["predicted_class"] == class_name]
        incorrect = [row for row in rows if row["true_class"] == class_name and row["predicted_class"] != class_name]
        if correct:
            selected.append(("correct", max(correct, key=lambda row: float(row["confidence"]))))
        if incorrect:
            selected.append(("error", max(incorrect, key=lambda row: float(row["confidence"]))))
    return selected


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from PIL import Image

    args = parse_args()
    repo = _multiclass_bootstrap.REPO_ROOT
    selection_path = args.selection if args.selection.is_absolute() else repo / args.selection
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "locked_test_evaluated":
        raise RuntimeError("Grad-CAM is post-freeze only and requires a completed locked-test evaluation")
    source_run = Path(selection["selected_run_dir"])
    config = ExperimentConfig(**json.loads((source_run / "config.json").read_text(encoding="utf-8")))
    checkpoint_path = Path(selection["selected_checkpoint"])
    device = torch.device("cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["config_hash"] != selection["configuration_hash"]:
        raise RuntimeError("frozen selection/checkpoint configuration mismatch")
    model, _ = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    layer = target_layer(model, config.architecture)
    predictions = read_csv(repo / "model/results/runs/final_locked_test/predictions.csv")
    examples = select_examples(predictions)
    source_root = Path(
        json.loads((repo / "model/data_manifests/dataset_source_snapshot.json").read_text(encoding="utf-8"))["source_root"]
    )
    transform = build_transform(config, training=False)
    output_dir = repo / "model/results/gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for index, (case_type, row) in enumerate(examples, 1):
        image_path = source_root / Path(row["relative_path"])
        original = Image.open(image_path).convert("RGB")
        tensor = transform(original).unsqueeze(0).to(device)
        captured = {}

        def hook(_module, _inputs, output):
            captured["activation"] = output.detach()
            output.register_hook(lambda gradient: captured.__setitem__("gradient", gradient.detach()))

        handle = layer.register_forward_hook(hook)
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        target_index = int(row["predicted_index"])
        logits[0, target_index].backward()
        handle.remove()
        activation = captured["activation"]
        gradient = captured["gradient"]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = functional.interpolate(cam, size=(config.input_size, config.input_size), mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)
        cam_np = cam.detach().cpu().numpy()
        visual = np.asarray(visual_preprocess(original, config.preprocessing, config.input_size), dtype=np.float32) / 255.0
        gray = visual.mean(axis=2)
        overlay = np.clip(0.62 * gray + 0.38 * cam_np, 0.0, 1.0)
        slug = re.sub(r"[^a-z0-9]+", "-", row["relative_path"].lower()).strip("-")[-70:]
        output_path = output_dir / f"{index:02d}_{case_type}_{row['true_class'].lower()}_{slug}.png"
        figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
        axes[0].imshow(gray, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Model input (pre-normalization)", color="black")
        axes[1].imshow(cam_np, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM intensity", color="black")
        axes[2].imshow(overlay, cmap="gray", vmin=0, vmax=1)
        axes[2].contour(cam_np, levels=[0.5, 0.75], colors=["black", "white"], linewidths=[0.8, 1.0])
        axes[2].set_title("Grayscale overlay and contours", color="black")
        for axis in axes:
            axis.axis("off")
        figure.suptitle(
            f"{case_type.upper()} | truth {row['true_class']} | predicted {row['predicted_class']} | confidence {float(row['confidence']):.3f}",
            color="black",
            fontsize=11,
        )
        figure.tight_layout()
        figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        records.append(
            {
                "case_type": case_type,
                "sample_id": row["sample_id"],
                "relative_path": row["relative_path"],
                "true_class": row["true_class"],
                "predicted_class": row["predicted_class"],
                "confidence": float(row["confidence"]),
                "target_class_for_gradcam": CLASS_NAMES[target_index],
                "target_layer": layer.__class__.__name__,
                "output_path": str(output_path),
            }
        )

    payload = {
        "schema_version": 1,
        "selection_status": selection["status"],
        "source_run_id": selection["selected_run_id"],
        "checkpoint": str(checkpoint_path),
        "architecture": config.architecture,
        "preprocessing": config.preprocessing,
        "method": "Grad-CAM of the predicted class at the final convolutional feature stage",
        "display": "grayscale input, grayscale activation, and grayscale overlay with black/white contours",
        "interpretation_boundary": "Qualitative localization only; not segmentation ground truth, causal proof, or clinical validation.",
        "records": records,
    }
    (output_dir / "gradcam_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
