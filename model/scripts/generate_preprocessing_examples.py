"""Generate validation-only visual examples for each implemented preprocessing path."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import _multiclass_bootstrap
from multiclass.preprocessing import preprocessing_metadata, visual_preprocess
from multiclass.tracking import atomic_write_json, now_iso


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    repo = _multiclass_bootstrap.REPO_ROOT
    manifest = repo / "model/data_manifests/val_manifest.csv"
    output = repo / "model/results/preprocessing_examples"
    output.mkdir(parents=True, exist_ok=True)
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = {}
    for row in rows:
        selected.setdefault(row["class_name"], row)
    methods = ["baseline", "clahe", "histogram_equalization", "pseudo_color"]
    classes = ["Normal", "Viral", "Bacterial"]
    figure, axes = plt.subplots(len(classes), len(methods), figsize=(12, 8.3))
    artifacts = []
    for row_index, class_name in enumerate(classes):
        row = selected[class_name]
        with Image.open(row["filepath"]) as source:
            source = source.copy()
        for column_index, method in enumerate(methods):
            transformed = visual_preprocess(source, method, 224)
            filename = f"{class_name.lower()}_{method}.png"
            transformed.save(output / filename)
            artifacts.append(
                {
                    "class_name": class_name,
                    "relative_source_path": row["relative_path"],
                    "preprocessing": method,
                    "output": filename,
                    "parameters": preprocessing_metadata(method).parameters,
                }
            )
            axis = axes[row_index, column_index]
            axis.imshow(transformed)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(method.replace("_", " ").title())
            if column_index == 0:
                axis.text(
                    -0.10, 0.5, class_name, transform=axis.transAxes, rotation=90,
                    va="center", ha="right", fontsize=11, fontweight="bold",
                )
    figure.suptitle("Validation examples before ImageNet normalization", fontsize=14)
    figure.tight_layout(rect=(0.02, 0.0, 1.0, 0.96))
    figure.savefig(output / "preprocessing_comparison.png", dpi=180)
    plt.close(figure)
    payload = {
        "generated_at": now_iso(),
        "manifest": "val_manifest.csv",
        "manifest_version": "three_class_seed42_3b815cf769c8",
        "input_size": 224,
        "training_data_used_for_fitting": False,
        "validation_data_used_for_fitting": False,
        "note": "All current preprocessing paths are deterministic fixed transforms; no learned fit occurs.",
        "comparison_figure": "preprocessing_comparison.png",
        "artifacts": artifacts,
    }
    atomic_write_json(output / "preprocessing_examples.json", payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
