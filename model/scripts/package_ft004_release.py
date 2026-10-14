"""Publish byte-identical FT-004 members and a portable selection descriptor.

No training, inference or dataset-image access occurs. The historical frozen
selection remains untouched; the derived package records its SHA-256.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import _multiclass_bootstrap  # noqa: F401
from multiclass.config import ExperimentConfig, REPO_ROOT

NAMES = (
    "member_1_viridis_ge_clahe_s42",
    "member_2_viridis_ge_s42",
    "member_3_ge_s42",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def copy_identical(source: Path, target: Path, expected: str) -> None:
    if not source.is_file() or sha256(source) != expected:
        raise RuntimeError(f"source artifact SHA-256 mismatch: {source}")
    if target.exists():
        if sha256(target) != expected:
            raise RuntimeError(f"existing publication artifact differs; preserve and investigate: {target}")
        return
    temporary = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, temporary)
    if sha256(temporary) != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"copied artifact SHA-256 mismatch: {target}")
    os.replace(temporary, target)


def verify_package(output: Path | str) -> dict:
    output = Path(output).resolve()
    manifest = json.loads((output / "package_manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    if manifest.get("system") != "FT-004" or selection.get("deployable_member_count") != 3:
        raise RuntimeError("publication package system/member mismatch")
    if selection.get("class_order") != ["Normal", "Viral", "Bacterial"] or selection.get("deployable_seed") != 42:
        raise RuntimeError("publication package class/seed mismatch")
    if len(selection.get("members", [])) != 3 or sorted(member["member_order"] for member in selection["members"]) != [1, 2, 3]:
        raise RuntimeError("publication package requires exactly the ordered three members")
    required = {"selection.json", "source_frozen_selection.json"}
    required.update(member[key] for member in selection["members"] for key in ("checkpoint_path", "configuration_path"))
    if not required <= set(manifest["files"]):
        raise RuntimeError("publication manifest omits a required artifact")
    for name, record in manifest["files"].items():
        path = output / name
        if path.parent != output or not path.is_file():
            raise RuntimeError(f"missing/invalid publication artifact path: {name}")
        if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"publication artifact changed: {name}")
    original = json.loads((output / "source_frozen_selection.json").read_text(encoding="utf-8"))
    if sha256(output / "source_frozen_selection.json") != manifest["source_frozen_selection_sha256"]:
        raise RuntimeError("source frozen selection checksum mismatch")
    for member, source in zip(sorted(selection["members"], key=lambda item: item["member_order"]), sorted(original["members"], key=lambda item: item["member_order"])):
        if member["checkpoint_sha256"] != source["checkpoint_sha256"] or member["run_id"] != source["run_id"]:
            raise RuntimeError("published member does not match frozen source")
        if manifest["files"][member["checkpoint_path"]]["sha256"] != member["checkpoint_sha256"]:
            raise RuntimeError("publication checkpoint manifest/selection mismatch")
        if manifest["files"][member["configuration_path"]]["sha256"] != source["configuration_sha256"]:
            raise RuntimeError("publication configuration bytes differ from frozen source")
        if Path(member["checkpoint_path"]).name != member["checkpoint_path"]:
            raise RuntimeError("published checkpoint path must be a portable filename")
        config = ExperimentConfig(**json.loads((output / member["configuration_path"]).read_text(encoding="utf-8")))
        payload = config.hash_payload()
        paths = member["configuration_hash_paths"]
        if set(paths) != {"manifest_dir", "results_dir"}:
            raise RuntimeError("original hash path context missing")
        payload.update(paths)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if digest != source["configuration_hash_recorded"] or member["configuration_hash_recorded"] != digest:
            raise RuntimeError("published configuration differs from historical configuration")
    return {"status": "verified", "system": "FT-004", "directory": str(output),
            "members": 3, "deployable_seed": 42, "files_verified": len(manifest["files"]),
            "checkpoint_bytes": sum(member["checkpoint_size_bytes"] for member in selection["members"]),
            "source_frozen_selection_sha256": manifest["source_frozen_selection_sha256"],
            "selection_sha256": sha256(output / "selection.json"), "research_use_only": True}


def package_release(source: Path | str, output: Path | str) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    original_hash = sha256(source)
    frozen = json.loads(source.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_validation_selected_research_model" or frozen.get("deployable_member_count") != 3 or frozen.get("deployable_seed") != 42:
        raise RuntimeError("source is not the original frozen seed-42 three-member FT-004")
    if frozen.get("class_order") != ["Normal", "Viral", "Bacterial"]:
        raise RuntimeError("source class order mismatch")
    if (output / "package_manifest.json").exists():
        existing = verify_package(output)
        if existing["source_frozen_selection_sha256"] != original_hash:
            raise RuntimeError("existing publication was derived from a different frozen record")
        return existing
    output.mkdir(parents=True, exist_ok=True)
    selection = copy.deepcopy(frozen)
    members = sorted(selection["members"], key=lambda item: item["member_order"])
    if [member["member_order"] for member in members] != [1, 2, 3]:
        raise RuntimeError("source member order mismatch")
    published = []
    for stem, member in zip(NAMES, members):
        checkpoint = Path(member["checkpoint_path"])
        configuration = Path(member["configuration_path"])
        config = ExperimentConfig(**json.loads(configuration.read_text(encoding="utf-8")))
        if config.config_hash != member["configuration_hash_recorded"]:
            raise RuntimeError("source config hash changed; generate the release in the original research checkout")
        if member["seed"] != 42 or config.seed != 42 or config.architecture != "densenet201":
            raise RuntimeError("source architecture/seed mismatch")
        checkpoint_name, config_name = stem + ".pt", stem + "_config.json"
        copy_identical(checkpoint, output / checkpoint_name, member["checkpoint_sha256"])
        copy_identical(configuration, output / config_name, member["configuration_sha256"])
        paths = config.hash_payload()
        member["source_checkpoint_path"] = member["checkpoint_path"]
        member["source_configuration_path"] = member["configuration_path"]
        member["checkpoint_path"] = checkpoint_name
        member["configuration_path"] = config_name
        member["configuration_hash_paths"] = {name: paths[name] for name in ("manifest_dir", "results_dir")}
        published.extend((checkpoint_name, config_name))
    selection["members"] = members
    selection["publication"] = {"created_at": datetime.now(timezone.utc).isoformat(),
        "source_frozen_selection": "source_frozen_selection.json",
        "source_frozen_selection_sha256": original_hash,
        "checkpoint_paths_relative_to": "this selection file's directory",
        "configuration_hash_path_context": "original absolute strings are retained only to verify historical config hashes; inference does not read research manifests",
        "checkpoints_byte_identical": True, "embedded_checkpoint_configurations_changed": False,
        "research_use_only": True, "independent_external_evaluation_completed": False}
    copy_identical(source, output / "source_frozen_selection.json", original_hash)
    atomic_json(output / "selection.json", selection)
    published.extend(("source_frozen_selection.json", "selection.json"))
    files = {name: {"sha256": sha256(output / name), "size_bytes": (output / name).stat().st_size} for name in published}
    atomic_json(output / "package_manifest.json", {"schema_version": 1, "system": "FT-004",
        "source_frozen_selection_sha256": original_hash, "files": files})
    if sha256(source) != original_hash:
        raise RuntimeError("original frozen source changed during packaging")
    return verify_package(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "model/results/final_ft004_ensemble/frozen_selection.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "model/artifacts/ft004")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = verify_package(args.output) if args.verify_only else package_release(args.source, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
