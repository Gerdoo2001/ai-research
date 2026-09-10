"""Audit the immutable chest X-ray source tree and build canonical image records.

The script never writes under the source directory. It inventories every physical
file, validates the canonical top-level train/val/test images, derives the three
requested labels from folder and filename evidence, computes SHA-256 and a 64-bit
perceptual hash, and records physical/exact/near duplicates and original-split
group leakage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "model" / "datasets" / "chest_xray"
DEFAULT_OUTPUT = REPO_ROOT / "model" / "data_manifests"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_TO_INDEX = {"Normal": 0, "Viral": 1, "Bacterial": 2}
CLASS_ORDER = ["Normal", "Viral", "Bacterial"]
CANONICAL_SPLITS = {"train", "val", "test"}

MANIFEST_FIELDS = [
    "filepath",
    "relative_path",
    "original_split",
    "assigned_split",
    "class_name",
    "class_index",
    "patient_or_group_id",
    "split_group_id",
    "filename_label_evidence",
    "image_width",
    "image_height",
    "image_mode",
    "file_size",
    "cryptographic_hash",
    "perceptual_hash",
    "validation_status",
    "exclusion_reason",
    "physical_role",
]

DUPLICATE_FIELDS = [
    "duplicate_type",
    "scope",
    "pair_or_group_id",
    "canonical_relative_path",
    "duplicate_relative_path",
    "cryptographic_hash",
    "perceptual_hamming_distance",
    "original_split_relation",
    "class_relation",
    "group_relation",
    "action",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physical_role(relative_path: Path) -> str:
    parts = relative_path.parts
    if not parts:
        return "unrecognized"
    if parts[0] in CANONICAL_SPLITS and len(parts) >= 3:
        return "canonical"
    if parts[0] == "chest_xray" and len(parts) >= 4 and parts[1] in CANONICAL_SPLITS:
        return "nested_copy"
    if parts[0] == "__MACOSX":
        return "macos_metadata"
    return "unrecognized"


def split_and_folder(relative_path: Path, role: str) -> tuple[str, str]:
    parts = relative_path.parts
    if role == "canonical" and len(parts) >= 3:
        return parts[0], parts[1]
    if role == "nested_copy" and len(parts) >= 4:
        return parts[1], parts[2]
    return "", ""


def derive_label(folder: str, filename: str) -> tuple[str, str, str]:
    lowered = filename.lower()
    if folder.upper() == "NORMAL":
        if "_virus_" in lowered or "_bacteria_" in lowered:
            return "", "", "normal_folder_conflicts_with_pneumonia_filename_token"
        return "Normal", "folder=NORMAL", ""
    if folder.upper() == "PNEUMONIA":
        has_virus = "_virus_" in lowered
        has_bacteria = "_bacteria_" in lowered
        if has_virus and not has_bacteria:
            return "Viral", "folder=PNEUMONIA; filename_token=_virus_", ""
        if has_bacteria and not has_virus:
            return "Bacterial", "folder=PNEUMONIA; filename_token=_bacteria_", ""
        if has_virus and has_bacteria:
            return "", "folder=PNEUMONIA", "conflicting_viral_and_bacterial_filename_tokens"
        return "", "folder=PNEUMONIA", "missing_viral_or_bacterial_filename_token"
    return "", f"folder={folder or '<missing>'}", "unsupported_class_folder"


def derive_group_id(class_name: str, filename: str) -> tuple[str, str]:
    if class_name in {"Viral", "Bacterial"}:
        match = re.match(r"^person(\d+)_", filename, flags=re.IGNORECASE)
        if match:
            return f"pneumonia_person_{int(match.group(1))}", ""
        return "", "unparseable_pneumonia_group_id"
    if class_name == "Normal":
        match = re.match(r"^(?:NORMAL2-)?IM-(\d+)-", filename, flags=re.IGNORECASE)
        if match:
            return f"normal_im_{int(match.group(1))}", ""
        return "", "unparseable_normal_group_id"
    return "", "missing_class_for_group_derivation"


def perceptual_hash(image: Image.Image) -> str:
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    array = np.asarray(gray, dtype=np.float32)
    coefficients = cv2.dct(array)[:8, :8]
    median = float(np.median(coefficients.flatten()[1:]))
    bits = (coefficients > median).flatten()
    bits[0] = False
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return f"{value:016x}"


def base_row(path: Path, source: Path, file_hash: str) -> dict[str, Any]:
    relative = path.relative_to(source)
    role = physical_role(relative)
    original_split, folder = split_and_folder(relative, role)
    return {
        "filepath": str(path.resolve()),
        "relative_path": relative.as_posix(),
        "original_split": original_split,
        "assigned_split": "",
        "class_name": "",
        "class_index": "",
        "patient_or_group_id": "",
        "split_group_id": "",
        "filename_label_evidence": "",
        "image_width": "",
        "image_height": "",
        "image_mode": "",
        "file_size": path.stat().st_size,
        "cryptographic_hash": file_hash,
        "perceptual_hash": "",
        "validation_status": "excluded",
        "exclusion_reason": "",
        "physical_role": role,
        "_folder": folder,
    }


def inspect_canonical(row: dict[str, Any], path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        row["exclusion_reason"] = "unsupported_file_format"
        return row
    class_name, evidence, label_error = derive_label(row["_folder"], path.name)
    row["filename_label_evidence"] = evidence
    if label_error:
        row["exclusion_reason"] = label_error
        return row
    group_id, group_error = derive_group_id(class_name, path.name)
    if group_error:
        row["exclusion_reason"] = group_error
        return row

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            row["image_width"], row["image_height"] = image.size
            row["image_mode"] = image.mode
            row["perceptual_hash"] = perceptual_hash(image)
    except Exception as exc:
        row["exclusion_reason"] = f"unreadable_or_corrupt_image:{type(exc).__name__}"
        return row

    row["class_name"] = class_name
    row["class_index"] = CLASS_TO_INDEX[class_name]
    row["patient_or_group_id"] = group_id
    row["split_group_id"] = group_id
    row["validation_status"] = "valid"
    row["exclusion_reason"] = ""
    return row


def inspect_path(path: Path, source: Path, file_hash: str) -> dict[str, Any]:
    row = base_row(path, source, file_hash)
    role = row["physical_role"]
    if role == "canonical":
        row = inspect_canonical(row, path)
    elif role == "nested_copy":
        row["exclusion_reason"] = "nested_copy_pending_hash_verification"
    elif role == "macos_metadata":
        row["exclusion_reason"] = "macos_resource_fork_or_metadata"
    else:
        row["exclusion_reason"] = "unrecognized_source_layout_or_metadata"
    row.pop("_folder", None)
    return row


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def inventory_fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["relative_path"]):
        digest.update(row["relative_path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["file_size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(row["cryptographic_hash"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def relation(left: str, right: str) -> str:
    return "same" if left == right else "different"


def canonical_duplicate_rows(valid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        by_hash[row["cryptographic_hash"]].append(row)
    for digest, group in sorted(by_hash.items()):
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda item: item["relative_path"])
        canonical = ordered[0]
        pair_id = f"exact_{digest[:16]}"
        for duplicate in ordered[1:]:
            reports.append(
                {
                    "duplicate_type": "exact_duplicate",
                    "scope": "canonical_source",
                    "pair_or_group_id": pair_id,
                    "canonical_relative_path": canonical["relative_path"],
                    "duplicate_relative_path": duplicate["relative_path"],
                    "cryptographic_hash": digest,
                    "perceptual_hamming_distance": 0,
                    "original_split_relation": relation(canonical["original_split"], duplicate["original_split"]),
                    "class_relation": relation(canonical["class_name"], duplicate["class_name"]),
                    "group_relation": relation(canonical["patient_or_group_id"], duplicate["patient_or_group_id"]),
                    "action": "retain_first_path_as_representative; exclude_redundant_canonical_copy",
                }
            )
    return reports


def nested_copy_rows(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    by_relative = {row["relative_path"]: row for row in all_rows}
    for row in all_rows:
        if row["physical_role"] != "nested_copy":
            continue
        relative = Path(row["relative_path"])
        canonical_relative = Path(*relative.parts[1:]).as_posix()
        canonical = by_relative.get(canonical_relative)
        if canonical and canonical["cryptographic_hash"] == row["cryptographic_hash"]:
            row["exclusion_reason"] = "verified_exact_nested_copy"
            duplicate_type = "exact_duplicate"
            action = "exclude_nested_physical_copy"
        else:
            row["exclusion_reason"] = "nested_copy_missing_or_hash_mismatch"
            duplicate_type = "nested_copy_mismatch"
            action = "exclude_and_review"
        reports.append(
            {
                "duplicate_type": duplicate_type,
                "scope": "physical_source_tree",
                "pair_or_group_id": f"physical_{row['cryptographic_hash'][:16]}",
                "canonical_relative_path": canonical_relative if canonical else "",
                "duplicate_relative_path": row["relative_path"],
                "cryptographic_hash": row["cryptographic_hash"],
                "perceptual_hamming_distance": 0 if canonical and canonical["cryptographic_hash"] == row["cryptographic_hash"] else "",
                "original_split_relation": "same" if canonical else "unknown",
                "class_relation": "same" if canonical else "unknown",
                "group_relation": "same_physical_image" if canonical else "unknown",
                "action": action,
            }
        )
    return reports


def original_group_leakage_rows(valid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        by_group[row["patient_or_group_id"]].append(row)
    for group_id, group in sorted(by_group.items()):
        splits = sorted({row["original_split"] for row in group})
        if len(splits) < 2:
            continue
        classes = sorted({row["class_name"] for row in group})
        ordered = sorted(group, key=lambda item: item["relative_path"])
        reports.append(
            {
                "duplicate_type": "group_leakage_original_split",
                "scope": "canonical_source",
                "pair_or_group_id": group_id,
                "canonical_relative_path": ordered[0]["relative_path"],
                "duplicate_relative_path": "|".join(row["relative_path"] for row in ordered[1:]),
                "cryptographic_hash": "",
                "perceptual_hamming_distance": "",
                "original_split_relation": "|".join(splits),
                "class_relation": "|".join(classes),
                "group_relation": "same_derived_group",
                "action": "rebuild_group_aware_split",
            }
        )
    return reports


def near_duplicate_rows(valid_rows: list[dict[str, Any]], threshold: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    hashes = [int(row["perceptual_hash"], 16) for row in valid_rows]
    for index, value in enumerate(hashes):
        for segment in range(8):
            buckets[(segment, (value >> (segment * 8)) & 0xFF)].append(index)

    candidates: set[tuple[int, int]] = set()
    for indices in buckets.values():
        if len(indices) < 2:
            continue
        for left, right in combinations(indices, 2):
            candidates.add((left, right) if left < right else (right, left))

    reports: list[dict[str, Any]] = []
    for left_index, right_index in sorted(candidates):
        left = valid_rows[left_index]
        right = valid_rows[right_index]
        if left["cryptographic_hash"] == right["cryptographic_hash"]:
            continue
        distance = (hashes[left_index] ^ hashes[right_index]).bit_count()
        if distance > threshold:
            continue
        same_group = left["patient_or_group_id"] == right["patient_or_group_id"]
        same_dimensions = (
            left["image_width"] == right["image_width"]
            and left["image_height"] == right["image_height"]
        )
        high_confidence = distance <= 2 or same_group or (same_dimensions and distance <= 3)
        if not high_confidence:
            continue
        pair_digest = hashlib.sha256(
            f"{left['relative_path']}\0{right['relative_path']}".encode("utf-8")
        ).hexdigest()[:16]
        reports.append(
            {
                "duplicate_type": "potential_near_duplicate",
                "scope": "canonical_source",
                "pair_or_group_id": f"near_{pair_digest}",
                "canonical_relative_path": left["relative_path"],
                "duplicate_relative_path": right["relative_path"],
                "cryptographic_hash": "",
                "perceptual_hamming_distance": distance,
                "original_split_relation": relation(left["original_split"], right["original_split"]),
                "class_relation": relation(left["class_name"], right["class_name"]),
                "group_relation": relation(left["patient_or_group_id"], right["patient_or_group_id"]),
                "action": "review_only; group-aware split contains same derived group",
            }
        )
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--near-duplicate-threshold", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = (args.source if args.source.is_absolute() else REPO_ROOT / args.source).resolve()
    output_dir = (args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")
    if output_dir == source or source in output_dir.parents:
        raise ValueError("Output directory must be outside the immutable source dataset")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    paths = sorted(path for path in source.rglob("*") if path.is_file())
    print(f"Auditing {len(paths)} physical files under {source}")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        hashes = list(executor.map(sha256_file, paths))

    rows: list[dict[str, Any]] = []
    for index, (path, file_hash) in enumerate(zip(paths, hashes), start=1):
        rows.append(inspect_path(path, source, file_hash))
        if index % 500 == 0 or index == len(paths):
            print(f"Inspected {index}/{len(paths)} files")

    nested_reports = nested_copy_rows(rows)
    valid_candidates = sorted(
        (row for row in rows if row["physical_role"] == "canonical" and row["validation_status"] == "valid"),
        key=lambda item: item["relative_path"],
    )
    canonical_exact_reports = canonical_duplicate_rows(valid_candidates)
    redundant_exact_paths = {
        report["duplicate_relative_path"] for report in canonical_exact_reports
    }
    for row in rows:
        if row["relative_path"] in redundant_exact_paths:
            row["validation_status"] = "excluded"
            row["exclusion_reason"] = "redundant_exact_duplicate_canonical"
    valid_rows = sorted(
        (
            row
            for row in rows
            if row["physical_role"] == "canonical" and row["validation_status"] == "valid"
        ),
        key=lambda item: item["relative_path"],
    )
    duplicate_reports = nested_reports
    duplicate_reports.extend(canonical_exact_reports)
    duplicate_reports.extend(original_group_leakage_rows(valid_rows))
    duplicate_reports.extend(near_duplicate_rows(valid_rows, args.near_duplicate_threshold))
    duplicate_reports.sort(
        key=lambda item: (
            item["duplicate_type"],
            item["canonical_relative_path"],
            item["duplicate_relative_path"],
        )
    )

    rows.sort(key=lambda item: item["relative_path"])
    excluded_rows = [row for row in rows if row["validation_status"] != "valid"]
    write_csv(output_dir / "dataset_audit.csv", rows, MANIFEST_FIELDS)
    write_csv(output_dir / "all_images_manifest.csv", valid_rows, MANIFEST_FIELDS)
    write_csv(output_dir / "excluded_images.csv", excluded_rows, MANIFEST_FIELDS)
    write_csv(output_dir / "duplicate_report.csv", duplicate_reports, DUPLICATE_FIELDS)

    class_mapping = {
        "schema_version": 1,
        "task_mode": "multiclass",
        "num_classes": 3,
        "class_order": CLASS_ORDER,
        "name_to_index": CLASS_TO_INDEX,
        "index_to_name": {str(value): key for key, value in CLASS_TO_INDEX.items()},
        "label_derivation": {
            "Normal": "canonical folder NORMAL",
            "Viral": "canonical folder PNEUMONIA and filename token _virus_",
            "Bacterial": "canonical folder PNEUMONIA and filename token _bacteria_",
        },
    }
    (output_dir / "class_mapping.json").write_text(
        json.dumps(class_mapping, indent=2, sort_keys=True), encoding="utf-8"
    )

    role_counts = Counter(row["physical_role"] for row in rows)
    exclusion_counts = Counter(row["exclusion_reason"] for row in excluded_rows)
    original_physical_counts = Counter(
        (row["original_split"], row["class_name"]) for row in valid_candidates
    )
    retained_original_counts = Counter(
        (row["original_split"], row["class_name"]) for row in valid_rows
    )
    group_classes: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in valid_rows:
        group_classes[row["patient_or_group_id"]].add(row["class_name"])
        group_splits[row["patient_or_group_id"]].add(row["original_split"])
    snapshot = {
        "schema_version": 1,
        "generated_at": now_iso(),
        "source_root": str(source),
        "source_file_count": len(rows),
        "source_bytes": sum(int(row["file_size"]) for row in rows),
        "source_content_inventory_hash": inventory_fingerprint(rows),
        "physical_role_counts": dict(sorted(role_counts.items())),
        "canonical_physical_valid_image_count_before_deduplication": len(valid_candidates),
        "canonical_valid_unique_image_count": len(valid_rows),
        "canonical_excluded_count": sum(
            1 for row in rows if row["physical_role"] == "canonical" and row["validation_status"] != "valid"
        ),
        "excluded_physical_file_count": len(excluded_rows),
        "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        "original_split_class_counts_before_deduplication": {
            f"{split_name}/{class_name}": count
            for (split_name, class_name), count in sorted(original_physical_counts.items())
        },
        "retained_original_split_class_counts": {
            f"{split_name}/{class_name}": count
            for (split_name, class_name), count in sorted(retained_original_counts.items())
        },
        "derived_group_count": len(group_classes),
        "groups_with_multiple_class_labels": sum(len(classes) > 1 for classes in group_classes.values()),
        "groups_crossing_original_splits": sum(len(splits) > 1 for splits in group_splits.values()),
        "duplicate_report_counts": dict(
            sorted(Counter(row["duplicate_type"] for row in duplicate_reports).items())
        ),
        "perceptual_hash": {
            "algorithm": "64-bit DCT perceptual hash; 32x32 grayscale, top-left 8x8 coefficients",
            "near_duplicate_threshold": args.near_duplicate_threshold,
            "near_duplicate_reporting_filter": "distance<=2 OR same derived group OR same dimensions and distance<=3",
        },
        "source_modified": False,
    }
    (output_dir / "dataset_source_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"Valid canonical images: {len(valid_rows)}")
    print(f"Excluded physical files: {len(excluded_rows)}")
    print(f"Duplicate/leakage report rows: {len(duplicate_reports)}")
    print(f"Source inventory hash: {snapshot['source_content_inventory_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
