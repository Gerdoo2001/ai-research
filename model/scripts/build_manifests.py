"""Build deterministic, group-aware locked train/validation/test manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_DIR = REPO_ROOT / "model" / "data_manifests"
CLASS_ORDER = ["Normal", "Viral", "Bacterial"]
SPLITS = ["train", "val", "test"]
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


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        canonical, other = sorted((left_root, right_root))
        self.parent[other] = canonical


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def objective(
    counts: dict[str, Counter[str]],
    totals: Counter[str],
    targets: dict[str, dict[str, float]],
    total_targets: dict[str, float],
) -> float:
    score = 0.0
    for split_name in SPLITS:
        for class_name in CLASS_ORDER:
            target = max(targets[split_name][class_name], 1.0)
            score += ((counts[split_name][class_name] - target) / target) ** 2
        total_target = max(total_targets[split_name], 1.0)
        score += 0.35 * ((totals[split_name] - total_target) / total_target) ** 2
    return score


def assign_groups(
    groups: dict[str, list[dict[str, str]]],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    global_class_counts = Counter(row["class_name"] for rows in groups.values() for row in rows)
    global_total = sum(global_class_counts.values())
    targets = {
        split_name: {
            class_name: global_class_counts[class_name] * ratios[split_name]
            for class_name in CLASS_ORDER
        }
        for split_name in SPLITS
    }
    total_targets = {split_name: global_total * ratios[split_name] for split_name in SPLITS}

    rng = random.Random(seed)
    tie_break = {group_id: rng.random() for group_id in groups}
    group_vectors = {
        group_id: Counter(row["class_name"] for row in rows) for group_id, rows in groups.items()
    }
    ordered_groups = sorted(
        groups,
        key=lambda group_id: (
            -sum(group_vectors[group_id].values()),
            -len(group_vectors[group_id]),
            -max(group_vectors[group_id].values()),
            tie_break[group_id],
            group_id,
        ),
    )

    counts = {split_name: Counter() for split_name in SPLITS}
    totals: Counter[str] = Counter()
    assignment: dict[str, str] = {}

    for group_id in ordered_groups:
        vector = group_vectors[group_id]
        best_split = ""
        best_score = float("inf")
        for split_name in SPLITS:
            for class_name, value in vector.items():
                counts[split_name][class_name] += value
            totals[split_name] += sum(vector.values())
            score = objective(counts, totals, targets, total_targets)
            for class_name, value in vector.items():
                counts[split_name][class_name] -= value
            totals[split_name] -= sum(vector.values())
            if score < best_score - 1e-12:
                best_score = score
                best_split = split_name
        assignment[group_id] = best_split
        for class_name, value in vector.items():
            counts[best_split][class_name] += value
        totals[best_split] += sum(vector.values())

    for _ in range(8):
        improved = False
        for group_id in sorted(groups, key=lambda item: tie_break[item]):
            source_split = assignment[group_id]
            vector = group_vectors[group_id]
            current_score = objective(counts, totals, targets, total_targets)
            best_split = source_split
            best_score = current_score
            for target_split in SPLITS:
                if target_split == source_split:
                    continue
                if any(
                    counts[source_split][class_name] - vector[class_name] <= 0
                    for class_name in vector
                    if vector[class_name] > 0
                ):
                    continue
                for class_name, value in vector.items():
                    counts[source_split][class_name] -= value
                    counts[target_split][class_name] += value
                total_value = sum(vector.values())
                totals[source_split] -= total_value
                totals[target_split] += total_value
                score = objective(counts, totals, targets, total_targets)
                for class_name, value in vector.items():
                    counts[source_split][class_name] += value
                    counts[target_split][class_name] -= value
                totals[source_split] += total_value
                totals[target_split] -= total_value
                if score < best_score - 1e-12:
                    best_score = score
                    best_split = target_split
            if best_split != source_split:
                for class_name, value in vector.items():
                    counts[source_split][class_name] -= value
                    counts[best_split][class_name] += value
                total_value = sum(vector.values())
                totals[source_split] -= total_value
                totals[best_split] += total_value
                assignment[group_id] = best_split
                improved = True
        if not improved:
            break
    return assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_dir = (
        args.manifest_dir if args.manifest_dir.is_absolute() else REPO_ROOT / args.manifest_dir
    ).resolve()
    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    if abs(sum(ratios.values()) - 1.0) > 1e-9 or any(value <= 0 for value in ratios.values()):
        raise ValueError("Split ratios must be positive and sum to 1")

    all_path = manifest_dir / "all_images_manifest.csv"
    audit_path = manifest_dir / "dataset_audit.csv"
    snapshot_path = manifest_dir / "dataset_source_snapshot.json"
    if not all_path.exists() or not audit_path.exists() or not snapshot_path.exists():
        raise FileNotFoundError("Run audit_dataset.py before build_manifests.py")

    rows = read_csv(all_path)
    if not rows:
        raise RuntimeError("Canonical image manifest is empty")
    if any(row["validation_status"] != "valid" for row in rows):
        raise RuntimeError("all_images_manifest.csv contains invalid rows")

    source_groups = sorted({row["patient_or_group_id"] for row in rows})
    union_find = UnionFind(source_groups)
    by_hash: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_hash[row["cryptographic_hash"]].append(row["patient_or_group_id"])
    for group_ids in by_hash.values():
        unique_group_ids = sorted(set(group_ids))
        for group_id in unique_group_ids[1:]:
            union_find.union(unique_group_ids[0], group_id)

    for row in rows:
        row["split_group_id"] = union_find.find(row["patient_or_group_id"])

    original_group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        original_group_splits[row["patient_or_group_id"]].add(row["original_split"])
    original_cross_split_groups = {
        group_id: sorted(split_names)
        for group_id, split_names in original_group_splits.items()
        if len(split_names) > 1
    }

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["split_group_id"]].append(row)
    assignment = assign_groups(groups, ratios, args.seed)
    for row in rows:
        row["assigned_split"] = assignment[row["split_group_id"]]
    rows.sort(key=lambda item: (item["assigned_split"], item["class_index"], item["relative_path"]))

    split_rows = {split_name: [row for row in rows if row["assigned_split"] == split_name] for split_name in SPLITS}
    for split_name, values in split_rows.items():
        present = {row["class_name"] for row in values}
        if present != set(CLASS_ORDER):
            raise RuntimeError(f"Assigned split {split_name} does not contain all classes: {sorted(present)}")

    split_group_sets = {
        split_name: {row["split_group_id"] for row in values}
        for split_name, values in split_rows.items()
    }
    group_leakage_pairs = [
        f"{left}/{right}"
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
        if split_group_sets[left] & split_group_sets[right]
    ]
    if group_leakage_pairs:
        raise RuntimeError(f"Group leakage remains after assignment: {group_leakage_pairs}")

    hash_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        hash_splits[row["cryptographic_hash"]].add(row["assigned_split"])
    exact_hash_leakage = {digest: values for digest, values in hash_splits.items() if len(values) > 1}
    if exact_hash_leakage:
        raise RuntimeError("Exact duplicate hashes cross assigned splits")

    write_csv(all_path, rows, MANIFEST_FIELDS)
    for split_name, values in split_rows.items():
        write_csv(manifest_dir / f"{split_name}_manifest.csv", values, MANIFEST_FIELDS)

    audit_rows = read_csv(audit_path)
    by_relative = {row["relative_path"]: row for row in rows}
    for audit_row in audit_rows:
        canonical = by_relative.get(audit_row["relative_path"])
        if canonical:
            audit_row["assigned_split"] = canonical["assigned_split"]
            audit_row["split_group_id"] = canonical["split_group_id"]
    write_csv(audit_path, audit_rows, MANIFEST_FIELDS)

    manifest_hashes = {
        name: sha256_file(manifest_dir / name)
        for name in (
            "all_images_manifest.csv",
            "train_manifest.csv",
            "val_manifest.csv",
            "test_manifest.csv",
            "class_mapping.json",
        )
    }
    combined = hashlib.sha256()
    for name, digest in sorted(manifest_hashes.items()):
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
    manifest_version_hash = combined.hexdigest()

    source_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    split_counts = {
        split_name: {
            class_name: sum(row["class_name"] == class_name for row in values)
            for class_name in CLASS_ORDER
        }
        for split_name, values in split_rows.items()
    }
    split_group_counts = {
        split_name: len(split_group_sets[split_name]) for split_name in SPLITS
    }
    summary = {
        "schema_version": 1,
        "manifest_version": f"three_class_seed{args.seed}_{manifest_version_hash[:12]}",
        "manifest_version_hash": manifest_version_hash,
        "manifest_hashes": manifest_hashes,
        "seed": args.seed,
        "class_order": CLASS_ORDER,
        "split_ratios": ratios,
        "split_algorithm": (
            "deterministic greedy multi-label group assignment with eight local-improvement passes; "
            "groups are derived conservatively from filename patient/image identifiers and exact-hash groups are unioned"
        ),
        "official_test_preserved": False,
        "official_test_decision_reason": (
            f"Not preserved because {len(original_cross_split_groups)} derived groups cross the original folder splits; "
            "all canonical images were reassigned without consulting image content or test metrics."
        ),
        "original_cross_split_group_count": len(original_cross_split_groups),
        "canonical_image_count": len(rows),
        "split_counts": split_counts,
        "split_total_counts": {split_name: len(values) for split_name, values in split_rows.items()},
        "split_group_counts": split_group_counts,
        "assigned_group_leakage_count": 0,
        "assigned_exact_hash_leakage_count": 0,
        "source_content_inventory_hash": source_snapshot["source_content_inventory_hash"],
        "source_root": source_snapshot["source_root"],
        "source_modified": False,
        "selection_policy": {
            "training": "fit model parameters and learned preprocessing only",
            "validation": "architecture, preprocessing, threshold, and hyperparameter selection",
            "test": "locked until final selected configuration",
        },
    }
    (manifest_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"Manifest version: {summary['manifest_version']}")
    for split_name in SPLITS:
        print(f"{split_name}: {len(split_rows[split_name])} images, {split_counts[split_name]}")
    print("Assigned group leakage: 0")
    print("Assigned exact-hash leakage: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
