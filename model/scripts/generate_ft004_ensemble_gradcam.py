"""Generate validation-only Grad-CAM evidence for the frozen FT-004 ensemble."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import _multiclass_bootstrap
from multiclass.config import ExperimentConfig
from multiclass.models import build_model
from multiclass.preprocessing import build_transform, visual_preprocess


CLASS_NAMES = ("Normal", "Viral", "Bacterial")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def target_layer(model, architecture: str):
    if architecture.startswith("densenet"):
        return model.features.norm5
    if architecture.startswith("resnet"):
        return model.layer4[-1]
    raise ValueError(f"Grad-CAM target layer is not defined for {architecture}")


def select_examples(rows: list[dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
    selected: list[tuple[str, dict[str, str]]] = []
    for class_name in CLASS_NAMES:
        correct = [
            row
            for row in rows
            if row["true_class"] == class_name and row["predicted_class"] == class_name
        ]
        incorrect = [
            row
            for row in rows
            if row["true_class"] == class_name and row["predicted_class"] != class_name
        ]
        if correct:
            selected.append(
                ("correct", max(correct, key=lambda row: max_probability(row)))
            )
        if incorrect:
            selected.append(
                ("error", max(incorrect, key=lambda row: max_probability(row)))
            )
    return selected


def max_probability(row: dict[str, str]) -> float:
    return max(
        float(row["probability_normal"]),
        float(row["probability_viral"]),
        float(row["probability_bacterial"]),
    )


def load_member(member: dict[str, Any], device):
    import torch

    config = ExperimentConfig(
        **json.loads(Path(member["configuration_path"]).read_text(encoding="utf-8"))
    )
    checkpoint_path = Path(member["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["config_hash"] != member["configuration_hash_recorded"]:
        raise RuntimeError(f"checkpoint/config mismatch: {member['run_id']}")
    model, _ = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return config, model


def compute_cam(model, layer, tensor, target_index: int):
    import torch
    import torch.nn.functional as functional

    captured: dict[str, Any] = {}

    def hook(_module, _inputs, output):
        captured["activation"] = output.detach()
        output.register_hook(
            lambda gradient: captured.__setitem__("gradient", gradient.detach())
        )

    handle = layer.register_forward_hook(hook)
    model.zero_grad(set_to_none=True)
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    logits[0, target_index].backward()
    handle.remove()
    activation = captured["activation"]
    gradient = captured["gradient"]
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = functional.interpolate(
        cam, size=(tensor.shape[-2], tensor.shape[-1]), mode="bilinear", align_corners=False
    )[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)
    return cam.detach().cpu().numpy(), probabilities


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("model/results/final_ft004_ensemble/frozen_selection.json"),
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image

    repo = _multiclass_bootstrap.REPO_ROOT
    selection_path = args.selection if args.selection.is_absolute() else repo / args.selection
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "frozen_validation_selected_research_model":
        raise RuntimeError("FT-004 Grad-CAM requires the frozen validation selection")
    if selection.get("locked_test_accessed_for_this_selection") is not False:
        raise RuntimeError("this Grad-CAM workflow is validation-only")

    predictions_path = (
        repo / "model" / "results" / "ensemble_confirmation" / "ensemble_predictions_s42.csv"
    )
    rows = read_csv(predictions_path)
    examples = select_examples(rows)
    source_root = Path(
        json.loads(
            (repo / "model/data_manifests/dataset_source_snapshot.json").read_text(
                encoding="utf-8"
            )
        )["source_root"]
    )
    output_dir = repo / "model" / "results" / "final_ft004_ensemble" / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    example_data: list[dict[str, Any]] = []
    for case_type, row in examples:
        image_path = source_root / Path(row["relative_path"])
        original = Image.open(image_path).convert("RGB")
        example_data.append(
            {
                "case_type": case_type,
                "row": row,
                "image_path": image_path,
                "original": original,
                "member_cams": [],
                "member_probabilities": [],
                "member_visuals": [],
            }
        )

    members = selection["members"]
    for member in members:
        config, model = load_member(member, device)
        layer = target_layer(model, config.architecture)
        transform = build_transform(config, training=False)
        for example in example_data:
            tensor = transform(example["original"]).unsqueeze(0).to(device)
            target_index = int(example["row"]["predicted_index"])
            cam, probabilities = compute_cam(model, layer, tensor, target_index)
            visual = np.asarray(
                visual_preprocess(example["original"], config.preprocessing, config.input_size),
                dtype=np.float32,
            ) / 255.0
            example["member_cams"].append(cam)
            example["member_probabilities"].append(probabilities)
            example["member_visuals"].append(visual)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    records = []
    for index, example in enumerate(example_data, 1):
        row = example["row"]
        cams = np.stack(example["member_cams"], axis=0)
        combined_cam = cams.mean(axis=0)
        combined_cam = (combined_cam - combined_cam.min()) / (
            combined_cam.max() - combined_cam.min() + 1e-12
        )
        original_gray = np.asarray(example["original"].convert("L").resize((224, 224)), dtype=np.float32) / 255.0
        slug = re.sub(r"[^a-z0-9]+", "-", row["relative_path"].lower()).strip("-")[-60:]
        output_path = output_dir / (
            f"{index:02d}_{example['case_type']}_{row['true_class'].lower()}_{slug}.png"
        )
        figure, axes = plt.subplots(1, 5, figsize=(17.0, 3.7))
        axes[0].imshow(original_gray, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Original X-ray", fontsize=9)
        short_names = ("VIRIDIS -> GE -> CLAHE", "VIRIDIS -> GE", "GE")
        for member_index in range(3):
            cam = cams[member_index]
            overlay = np.clip(0.70 * original_gray + 0.30 * cam, 0.0, 1.0)
            axes[member_index + 1].imshow(overlay, cmap="gray", vmin=0, vmax=1)
            axes[member_index + 1].contour(
                cam, levels=[0.5, 0.75], colors=["black", "white"], linewidths=[0.7, 0.9]
            )
            axes[member_index + 1].set_title(f"Member {member_index + 1}\n{short_names[member_index]}", fontsize=8)
        combined_overlay = np.clip(0.70 * original_gray + 0.30 * combined_cam, 0.0, 1.0)
        axes[4].imshow(combined_overlay, cmap="gray", vmin=0, vmax=1)
        axes[4].contour(
            combined_cam, levels=[0.5, 0.75], colors=["black", "white"], linewidths=[0.8, 1.0]
        )
        axes[4].set_title("Mean member attention", fontsize=9)
        for axis in axes:
            axis.axis("off")
        confidence = max_probability(row)
        figure.suptitle(
            f"{example['case_type'].upper()} | truth {row['true_class']} | "
            f"ensemble prediction {row['predicted_class']} | confidence {confidence:.3f}",
            fontsize=11,
            color="black",
        )
        figure.tight_layout(rect=(0, 0, 1, 0.90))
        figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(figure)

        records.append(
            {
                "case_type": example["case_type"],
                "sample_id": row["sample_id"],
                "relative_path": row["relative_path"],
                "true_class": row["true_class"],
                "ensemble_predicted_class": row["predicted_class"],
                "ensemble_confidence": confidence,
                "target_class_for_gradcam": row["predicted_class"],
                "member_predictions": [
                    {
                        "run_id": member["run_id"],
                        "preprocessing": member["preprocessing"],
                        "probabilities": {
                            class_name: float(probability)
                            for class_name, probability in zip(
                                CLASS_NAMES, example["member_probabilities"][member_index]
                            )
                        },
                    }
                    for member_index, member in enumerate(members)
                ],
                "output_path": str(output_path),
            }
        )

    payload = {
        "schema_version": 1,
        "selection_id": selection["selection_id"],
        "split": "validation",
        "validation_seed_view": 42,
        "locked_test_accessed": False,
        "method": (
            "Grad-CAM for the ensemble-predicted class at DenseNet-201 features.norm5; "
            "member maps normalized independently and their mean displayed descriptively"
        ),
        "examples": "highest-confidence correct and highest-confidence error for each true class",
        "interpretation_boundary": (
            "Qualitative attention evidence only. Grad-CAM is not a lung mask, causal explanation, "
            "localization ground truth, or clinical validation."
        ),
        "records": records,
    }
    (output_dir / "gradcam_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
