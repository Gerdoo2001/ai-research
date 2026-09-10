"""Build the Phase 1 machine-readable legacy code and artifact inventory."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "model"
OUTPUT_PATH = MODEL_ROOT / "docs" / "00_code_inventory.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".csv", ".bat"}
GENERATED_ROOTS = {
    MODEL_ROOT / "docs",
    MODEL_ROOT / "results",
    MODEL_ROOT / "data_manifests",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def file_record(path: Path, include_hash: bool = True) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {
        "path": relative(path),
        "bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
        "suffix": path.suffix.lower(),
    }
    if include_hash:
        record["sha256"] = sha256(path)
    if path.suffix.lower() in TEXT_SUFFIXES:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace")
        record["line_count"] = len(text.splitlines())
    return record


def hdf5_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"path": relative(path)}
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            summary["root_groups"] = sorted(handle.keys())
            summary["root_attributes"] = sorted(handle.attrs.keys())
            model_config_raw = handle.attrs.get("model_config")
            training_config_raw = handle.attrs.get("training_config")
            if model_config_raw:
                if isinstance(model_config_raw, bytes):
                    model_config_raw = model_config_raw.decode("utf-8")
                model_config = json.loads(model_config_raw)
                config = model_config.get("config", {})
                layers = config.get("layers", [])
                summary["model_class"] = model_config.get("class_name")
                summary["model_name"] = config.get("name")
                summary["layer_count"] = len(layers)
                if layers:
                    final_layer = layers[-1]
                    summary["final_layer"] = {
                        "class_name": final_layer.get("class_name"),
                        "name": final_layer.get("config", {}).get("name"),
                        "units": final_layer.get("config", {}).get("units"),
                        "filters": final_layer.get("config", {}).get("filters"),
                        "activation": final_layer.get("config", {}).get("activation"),
                    }
            if training_config_raw:
                if isinstance(training_config_raw, bytes):
                    training_config_raw = training_config_raw.decode("utf-8")
                training_config = json.loads(training_config_raw)
                summary["training_config"] = {
                    "loss": training_config.get("loss"),
                    "optimizer_class": training_config.get("optimizer_config", {}).get("class_name"),
                    "metrics": training_config.get("metrics"),
                }
    except Exception as exc:
        summary["inspection_error"] = f"{type(exc).__name__}: {exc}"
    return summary


def directory_summary(path: Path) -> dict[str, Any]:
    files = [item for item in path.rglob("*") if item.is_file()] if path.exists() else []
    suffix_counts = Counter(item.suffix.lower() or "<none>" for item in files)
    return {
        "path": relative(path),
        "file_count": len(files),
        "bytes": sum(item.stat().st_size for item in files),
        "suffix_counts": dict(sorted(suffix_counts.items())),
    }


def is_generated(path: Path) -> bool:
    return any(path == root or root in path.parents for root in GENERATED_ROOTS)


def legacy_term_hits() -> dict[str, list[dict[str, Any]]]:
    terms = {
        "film": "film",
        "sas_tuning": "sas tuning",
        "automated_grading": "automated grading",
        "grading_of_chest": "grading of chest",
        "clahe": "clahe",
        "pca": "pca",
        "histogram_equalization": "histogram equalization",
        "pseudo_coloring": "pseudo-color",
        "yolo": "yolo",
        "resnet": "resnet",
    }
    candidates: list[Path] = []
    for root in (MODEL_ROOT / "src", REPO_ROOT / "web" / "backend"):
        if root.exists():
            candidates.extend(
                path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
            )
    candidates.extend(
        path
        for path in (MODEL_ROOT / "requirements.txt", REPO_ROOT / "requirements.working.txt", REPO_ROOT / "PneumoNet.bat")
        if path.exists()
    )
    hits: dict[str, list[dict[str, Any]]] = {key: [] for key in terms}
    for path in sorted(set(candidates)):
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()
            for key, term in terms.items():
                if term in lowered:
                    hits[key].append(
                        {"path": relative(path), "line": line_number, "text": line.strip()[:300]}
                    )
    return hits


def main() -> int:
    source_files = sorted(path for path in (MODEL_ROOT / "src").rglob("*") if path.is_file())
    legacy_outputs = sorted(path for path in (MODEL_ROOT / "outputs").rglob("*") if path.is_file())
    dependency_files = [
        path
        for path in (MODEL_ROOT / "requirements.txt", REPO_ROOT / "requirements.working.txt")
        if path.exists()
    ]
    scripts = sorted(
        path
        for path in (MODEL_ROOT / "scripts").rglob("*")
        if path.is_file() and not is_generated(path)
    )

    inventory: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "repository_root": str(REPO_ROOT),
        "model_root": str(MODEL_ROOT),
        "scope_note": (
            "Source images are summarized here and are inventoried per image in Phase 2 manifests. "
            "Generated documentation/results are excluded to avoid self-referential hashes."
        ),
        "directory_summaries": [
            directory_summary(MODEL_ROOT / name)
            for name in ("src", "datasets", "augmented_data", "outputs", "scripts")
        ],
        "legacy_source_files": [file_record(path) for path in source_files],
        "audit_scripts": [file_record(path) for path in scripts],
        "dependency_files": [file_record(path) for path in dependency_files],
        "legacy_output_artifacts": [file_record(path) for path in legacy_outputs],
        "legacy_hdf5_summaries": [
            hdf5_summary(path) for path in legacy_outputs if path.suffix.lower() in {".h5", ".hdf5"}
        ],
        "legacy_term_search": legacy_term_hits(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote code inventory: {OUTPUT_PATH}")
    print(f"Legacy source files: {len(source_files)}")
    print(f"Legacy output artifacts: {len(legacy_outputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
