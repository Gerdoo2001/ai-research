"""Generate dataset-audit figures from locked manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_DIR = REPO_ROOT / "model" / "data_manifests"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "model" / "results" / "dataset_audit"
CLASS_ORDER = ["Normal", "Viral", "Bacterial"]
SPLITS = ["train", "val", "test"]
COLORS = {"Normal": "#4C78A8", "Viral": "#F2CF5B", "Bacterial": "#E45756"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def plot_class_counts(rows: list[dict[str, str]], output: Path) -> None:
    counts = {
        split_name: {
            class_name: sum(
                row["assigned_split"] == split_name and row["class_name"] == class_name for row in rows
            )
            for class_name in CLASS_ORDER
        }
        for split_name in SPLITS
    }
    x = range(len(SPLITS))
    width = 0.24
    figure, axis = plt.subplots(figsize=(9, 5.4))
    for index, class_name in enumerate(CLASS_ORDER):
        offsets = [value + (index - 1) * width for value in x]
        values = [counts[split_name][class_name] for split_name in SPLITS]
        bars = axis.bar(offsets, values, width, label=class_name, color=COLORS[class_name])
        axis.bar_label(bars, padding=3, fontsize=9)
    axis.set_xticks(list(x), [name.title() for name in SPLITS])
    axis.set_ylabel("Images")
    axis.set_title("Locked Three-Class Manifest Counts")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_image_sizes(rows: list[dict[str, str]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for class_name in CLASS_ORDER:
        class_rows = [row for row in rows if row["class_name"] == class_name]
        widths = [int(row["image_width"]) for row in class_rows]
        heights = [int(row["image_height"]) for row in class_rows]
        axes[0].hist(widths, bins=35, alpha=0.45, label=class_name, color=COLORS[class_name])
        axes[1].hist(heights, bins=35, alpha=0.45, label=class_name, color=COLORS[class_name])
    axes[0].set_title("Image Width Distribution")
    axes[0].set_xlabel("Pixels")
    axes[1].set_title("Image Height Distribution")
    axes[1].set_xlabel("Pixels")
    for axis in axes:
        axis.set_ylabel("Images")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_dir = args.manifest_dir if args.manifest_dir.is_absolute() else REPO_ROOT / args.manifest_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(manifest_dir / "all_images_manifest.csv")
    plot_class_counts(rows, output_dir / "class_counts.png")
    plot_image_sizes(rows, output_dir / "image_size_distributions.png")
    summary = json.loads((manifest_dir / "split_summary.json").read_text(encoding="utf-8"))
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": summary["manifest_version"],
                "figures": ["class_counts.png", "image_size_distributions.png"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote dataset audit figures to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
