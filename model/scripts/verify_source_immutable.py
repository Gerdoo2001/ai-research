"""Verify every source-dataset file against the Phase 2 SHA-256 inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT = REPO_ROOT / "model" / "data_manifests" / "dataset_audit.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = args.audit if args.audit.is_absolute() else REPO_ROOT / args.audit
    with audit.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    paths = [Path(row["filepath"]) for row in rows]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        print(f"Missing files: {len(missing)}")
        for value in missing[:20]:
            print(value)
        return 1
    size_mismatches = [
        row["relative_path"]
        for row, path in zip(rows, paths)
        if path.stat().st_size != int(row["file_size"])
    ]
    if size_mismatches:
        print(f"Size mismatches: {len(size_mismatches)}")
        for value in size_mismatches[:20]:
            print(value)
        return 1
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        hashes = list(executor.map(sha256_file, paths))
    hash_mismatches = [
        row["relative_path"]
        for row, current_hash in zip(rows, hashes)
        if current_hash != row["cryptographic_hash"]
    ]
    if hash_mismatches:
        print(f"SHA-256 mismatches: {len(hash_mismatches)}")
        for value in hash_mismatches[:20]:
            print(value)
        return 1
    print(f"Verified immutable source inventory: {len(rows)} files, 0 mismatches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
