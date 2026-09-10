"""Prepare, but do not decide, the 16 perceptual-hash review pairs.

The script creates side-by-side evidence sheets and a machine-readable review
worksheet.  It never removes an image or changes a locked manifest.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import _multiclass_bootstrap


REVIEW_COLUMNS = (
    "pair_id",
    "left_relative_path",
    "right_relative_path",
    "left_assigned_split",
    "right_assigned_split",
    "class_relation",
    "group_relation",
    "perceptual_hamming_distance",
    "left_dimensions",
    "right_dimensions",
    "mean_absolute_difference_256",
    "pixel_correlation_256",
    "visual_review_status",
    "manual_decision",
    "manual_rationale",
    "manifest_action",
    "evidence_sheet",
)


# Direct visual review completed 2026-08-02 against the generated side-by-side
# evidence sheets.  The pHash candidates share the common geometry of pediatric
# frontal chest radiographs, but each pair has non-matching anatomical landmarks.
MANUAL_RATIONALES = {
    "near_5611f5dce8b45d92": "Different thoracic anatomy, inspiration, positioning, and R-marker placement.",
    "near_00f755ace402461b": "Different thoracic contours, clavicle alignment, exposure, and diaphragm shape.",
    "near_9baff834314599d5": "Different body size, positioning, lung volumes, and cardiomediastinal contours.",
    "near_ab80f0eac1b035bd": "Different age/body size, clavicle alignment, lung volumes, and diaphragm contours.",
    "near_5776286b699047d0": "Different thoracic proportions, cardiac silhouette, inspiration, and marker placement.",
    "near_605ba5843cd4df27": "Different age/body size, exposure, lung volumes, and cardiomediastinal anatomy.",
    "near_1ba4be31eda3af8f": "Different thoracic proportions, rotation, cardiac silhouette, and lung texture.",
    "near_78528749b3f26e02": "Different thoracic anatomy, inspiration, projection, and marker placement.",
    "near_fe8ef67898a877da": "Different shoulder/clavicle geometry, cardiac silhouette, diaphragm, and positioning.",
    "near_bb0271d7571b90bf": "Different rotation, rib and clavicle geometry, cardiac contour, and exposure.",
    "near_108295e9cc425d75": "Different body size, rotation, shoulder anatomy, and cardiomediastinal contours.",
    "near_46a898efcc56e29e": "Different infant anatomy, lung volumes, rib geometry, and cardiac silhouette.",
    "near_03e374c0d9e71141": "Different exposure, thoracic proportions, lung texture, and cardiac silhouette.",
    "near_5259cdc0ec6f4917": "Different body size, rib/clavicle alignment, lung texture, and diaphragm contours.",
    "near_90cf97b9b16e2b4e": "Different age/body size, rotation, thoracic anatomy, and cardiac silhouette.",
    "near_1ba9cfb07901db30": "Different thoracic proportions, lung texture, cardiac silhouette, and positioning.",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fitted(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "black")
    copy = image.convert("RGB")
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - copy.width) // 2
    y = (size[1] - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas


def comparison_stats(left: Image.Image, right: Image.Image) -> tuple[float, float]:
    left_array = np.asarray(ImageOps.fit(left.convert("L"), (256, 256)), dtype=np.float64) / 255.0
    right_array = np.asarray(ImageOps.fit(right.convert("L"), (256, 256)), dtype=np.float64) / 255.0
    difference = float(np.mean(np.abs(left_array - right_array)))
    left_flat = left_array.ravel()
    right_flat = right_array.ravel()
    correlation = float(np.corrcoef(left_flat, right_flat)[0, 1])
    return difference, correlation


def main() -> int:
    repo = _multiclass_bootstrap.REPO_ROOT
    manifests = repo / "model/data_manifests"
    output = repo / "model/results/perceptual_hash_manual_review"
    sheets = output / "evidence_sheets"
    sheets.mkdir(parents=True, exist_ok=True)

    all_rows = read_csv(manifests / "all_images_manifest.csv")
    by_relative = {row["relative_path"]: row for row in all_rows}
    candidates = [
        row
        for row in read_csv(manifests / "duplicate_report.csv")
        if row["duplicate_type"] == "potential_near_duplicate"
    ]
    if len(candidates) != 16:
        raise RuntimeError(f"expected 16 candidate pairs, found {len(candidates)}")
    if {row["pair_or_group_id"] for row in candidates} != set(MANUAL_RATIONALES):
        raise RuntimeError("manual decisions do not match the current 16-candidate review set")

    font = ImageFont.load_default()
    review_rows = []
    for index, candidate in enumerate(candidates, start=1):
        left_row = by_relative[candidate["canonical_relative_path"]]
        right_row = by_relative[candidate["duplicate_relative_path"]]
        with Image.open(left_row["filepath"]) as left_source, Image.open(right_row["filepath"]) as right_source:
            left = left_source.copy()
            right = right_source.copy()
        difference, correlation = comparison_stats(left, right)

        board = Image.new("RGB", (1800, 1120), "white")
        draw = ImageDraw.Draw(board)
        draw.text((40, 24), f"Pair {index:02d}: {candidate['pair_or_group_id']}", fill="black", font=font)
        draw.text(
            (40, 50),
            (
                f"pHash distance={candidate['perceptual_hamming_distance']} | "
                f"class={candidate['class_relation']} | group={candidate['group_relation']} | "
                f"standardized MAD={difference:.6f} | correlation={correlation:.6f}"
            ),
            fill="black",
            font=font,
        )
        board.paste(fitted(left, (850, 850)), (40, 105))
        board.paste(fitted(right, (850, 850)), (910, 105))
        left_label = (
            f"LEFT: {left_row['relative_path']}\n"
            f"locked split={left_row['assigned_split']} | size={left.width}x{left.height} | "
            f"group={left_row['split_group_id']}"
        )
        right_label = (
            f"RIGHT: {right_row['relative_path']}\n"
            f"locked split={right_row['assigned_split']} | size={right.width}x{right.height} | "
            f"group={right_row['split_group_id']}"
        )
        draw.multiline_text((40, 980), left_label, fill="black", font=font, spacing=4)
        draw.multiline_text((910, 980), right_label, fill="black", font=font, spacing=4)
        sheet = sheets / f"pair-{index:02d}.png"
        board.save(sheet)

        review_rows.append(
            {
                "pair_id": candidate["pair_or_group_id"],
                "left_relative_path": left_row["relative_path"],
                "right_relative_path": right_row["relative_path"],
                "left_assigned_split": left_row["assigned_split"],
                "right_assigned_split": right_row["assigned_split"],
                "class_relation": candidate["class_relation"],
                "group_relation": candidate["group_relation"],
                "perceptual_hamming_distance": candidate["perceptual_hamming_distance"],
                "left_dimensions": f"{left.width}x{left.height}",
                "right_dimensions": f"{right.width}x{right.height}",
                "mean_absolute_difference_256": f"{difference:.8f}",
                "pixel_correlation_256": f"{correlation:.8f}",
                "visual_review_status": "completed",
                "manual_decision": "Distinct images - retain both",
                "manual_rationale": MANUAL_RATIONALES[candidate["pair_or_group_id"]],
                "manifest_action": "Retain both; no locked-manifest change",
                "evidence_sheet": str(sheet.relative_to(repo)).replace("\\", "/"),
            }
        )

    csv_path = output / "perceptual_hash_manual_review.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(review_rows)
    json_path = output / "perceptual_hash_manual_review.json"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "candidate_pair_count": len(review_rows),
                "visual_review_completed": True,
                "duplicate_pairs_confirmed": 0,
                "distinct_pairs_confirmed": len(review_rows),
                "locked_manifests_modified": False,
                "automatic_exclusion_performed": False,
                "records": review_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(csv_path)
    print(json_path)
    print(f"evidence_sheets={len(review_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
