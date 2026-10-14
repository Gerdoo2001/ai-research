"""Bounded FT-011 preparation, protected smoke and conditional seed queue."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import _multiclass_bootstrap  # noqa: F401

from multiclass.config import REPO_ROOT, load_config
from multiclass.data import sha256_file, validate_manifest_bundle
from multiclass.ft011_safety import SafetyPolicy, SafetyWatchdog, HardwareSampler, query_gpu
from multiclass.tracking import atomic_write_json, now_iso


ROOT = REPO_ROOT / "model/results/ft011"
CONTROL_CONFIG = REPO_ROOT / "model/configs/fine_tuning/densenet201_viridis_ge_clahe_label_smoothing005_s42.json"
SEEDS = (42, 123, 2026)
SOURCES = (
    "model/src/multiclass/trainer.py", "model/src/multiclass/data.py",
    "model/src/multiclass/models.py", "model/src/multiclass/preprocessing.py",
    "model/src/multiclass/metrics.py", "model/src/multiclass/evaluation.py",
    "model/src/multiclass/config.py", "model/src/multiclass/ft011_training.py",
    "model/src/multiclass/ft011_safety.py", "model/scripts/run_ft011.py",
    "model/scripts/analyze_ft011.py",
)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare():
    from analyze_ft011 import freeze_controls

    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "protocol.json"
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        verify(frozen)
        return frozen
    control = load_config(CONTROL_CONFIG)
    manifests = validate_manifest_bundle(control.manifest_path)
    expected = {"architecture": "densenet201", "weights": "imagenet", "batch_size": 8,
        "epochs": 12, "learning_rate": 1e-4, "weight_decay": 1e-4,
        "label_smoothing": 0.05, "minimum_delta": 1e-4, "num_workers": 0,
        "amp": True, "freeze_backbone": False, "input_size": 224,
        "preprocessing": "pseudo_color_then_histogram_equalization_then_clahe",
        "loss": "cross_entropy", "class_weighting": "balanced",
        "scheduler": "reduce_on_plateau", "scheduler_factor": 0.5,
        "scheduler_patience": 2, "early_stopping_patience": 4}
    for key, value in expected.items():
        if getattr(control, key) != value:
            raise RuntimeError(f"FT-010 mismatch: {key}")
    packages = {p: importlib.metadata.version(p) for p in (
        "torch", "torchvision", "numpy", "Pillow", "opencv-python", "scikit-learn", "scipy", "pandas")}
    historical = json.loads((REPO_ROOT / "model/results/environment.json").read_text(encoding="utf-8"))["multiclass_environment"]
    for key, value in packages.items():
        if historical["packages"].get(key) != value:
            raise RuntimeError(f"Runtime mismatch: {key}: {value} vs {historical['packages'].get(key)}")
    if platform.python_version() != historical["python_version"]:
        raise RuntimeError("Historical Python version mismatch")
    gpu = query_gpu()
    if "RTX 2050" not in gpu["gpu_name"] or gpu["gpu_vram_total_bytes"] != 4096 * 1024**2:
        raise RuntimeError("Original RTX 2050 4GB hardware mismatch")
    policy = SafetyPolicy.from_hardware(gpu)
    checkpoint_bytes = 256 * 1024**2  # measured historical full checkpoints ~219MB; includes pending batch/order margin
    hardware = HardwareSampler(REPO_ROOT)()
    weights = Path(os.environ["USERPROFILE"]) / ".cache/torch/hub/checkpoints/densenet201-c1103571.pth"
    weight_hash = sha256_file(weights)
    if weight_hash != "c1103571aad4f45d0b5b15556d92944db94c33bee68cd58babba3d76225a6401":
        raise RuntimeError("Original ImageNet weight artifact mismatch")
    controls = freeze_controls(REPO_ROOT)
    record = {
        "experiment": "FT-011", "frozen_at": now_iso(), "status": "prepared",
        "protocol_sources": [
            "https://docs.google.com/document/d/15m0nbeDkg9mOR14nv6e6ugqWiczRaeiC5C2ijUdpRz0/edit",
            "https://docs.google.com/document/d/1mdLbxZMtv2icHKNs-VSjpZiWb5_2lG-S/edit",
            "https://docs.google.com/document/d/1I9-egb-SZRzbGQ-PXxM8UjvGmc_CCcsZ/edit"],
        "control_configuration": control.as_dict(), "control_configuration_hash": control.config_hash,
        "learning_rate_policy": {"backbone": 1e-5, "classifier": 1e-4},
        "control_learning_rate_policy": {"backbone": 1e-4, "classifier": 1e-4},
        "scheduler_defaults": {"mode": "max", "factor": 0.5, "patience": 2,
            "threshold": 1e-4, "threshold_mode": "rel", "cooldown": 0, "min_lr": [0.0, 0.0], "eps": 1e-8,
            "monitored_metric": "validation macro F1"},
        "checkpoint_rule": "strictly greater validation macro F1 than incumbent + 0.0001",
        "seeds": list(SEEDS), "maximum_successful_candidate_runs": 3,
        "controls": controls, "manifest_version_hash": manifests["manifest_version_hash"],
        "packages": packages, "python": platform.python_version(),
        "weight_identifier": "torchvision DenseNet201_Weights.IMAGENET1K_V1 (DEFAULT)",
        "weights_path": str(weights), "weights_sha256": weight_hash,
        "source_hashes": {p: sha256_file(REPO_ROOT / p) for p in SOURCES},
        "hardware_snapshot": hardware, "safety_policy": dataclasses.asdict(policy),
        "checkpoint_disk_estimate_bytes": checkpoint_bytes,
        "disk_reserve_bytes": policy.disk_reserve_bytes(checkpoint_bytes),
        "repairs": [{"repair": "CRLF to LF in four locked CSV manifests",
            "reason": "Git core.autocrlf converted the working files; all original SHA256 recovered exactly",
            "assignments_changed": False}],
        "provenance_limits": ["Historical runs saved configurations, ImageNet identity, checkpoints and predictions, but did not save initial-state hashes or per-run source commits; current source hashes and fresh initialization hashes are recorded."],
        "commands": {"prepare": ".venv\\Scripts\\python.exe model\\scripts\\run_ft011.py --prepare",
            "smoke": ".venv\\Scripts\\python.exe model\\scripts\\run_ft011.py --smoke",
            "train": ".venv\\Scripts\\python.exe model\\scripts\\run_ft011.py --run",
            "resume": ".venv\\Scripts\\python.exe model\\scripts\\run_ft011.py --run --resume"},
        "locked_test_images_accessed": False,
    }
    record["protocol_hash"] = canonical_hash(record)
    atomic_write_json(path, record)
    atomic_write_json(ROOT / "progress.json", {"status": "prepared", "updated_at": now_iso()})
    for seed in SEEDS:
        base = control.as_dict()
        base.update(seed=seed, run_id=f"ft011_densenet201_differential_lr_s{seed}")
        cfg = {"base_training_configuration": base, "learning_rate_policy": record["learning_rate_policy"],
               "protocol_hash": record["protocol_hash"]}
        cfg["candidate_configuration_hash"] = canonical_hash(cfg)
        atomic_write_json(ROOT / "configs" / f"seed{seed}.json", cfg)
    print(json.dumps({"status": "prepared", "protocol_hash": record["protocol_hash"],
        "screening_gates": controls["screening_gates"], "safety_policy": record["safety_policy"]}, indent=2), flush=True)
    return record


def verify(record):
    hashed = dict(record)
    expected = hashed.pop("protocol_hash")
    if canonical_hash(hashed) != expected:
        raise RuntimeError("Frozen FT-011 protocol hash mismatch")
    for source, expected in record["source_hashes"].items():
        if sha256_file(REPO_ROOT / source) != expected:
            raise RuntimeError(f"Frozen implementation changed: {source}; document repair before rerunning")
    validate_manifest_bundle(load_config(CONTROL_CONFIG).manifest_path)


def close_phase(summary, record):
    if not summary["terminal_decision_complete"]:
        raise RuntimeError("Cannot freeze a pending FT-011 decision")
    selected = summary["decision"] == "select FT-011"
    if selected:
        package = {"selection_id": "FINAL-FT011-DIFFERENTIAL-LR-S42", "status": "frozen_validation_selected_research_model",
            "frozen_at": now_iso(), "architecture": "densenet201", "deployable_seed": 42,
            "member_count": 1, "class_order": ["Normal", "Viral", "Bacterial"],
            "preprocessing": "VIRIDIS -> Global Histogram Equalization -> CLAHE",
            "prediction_rule": "softmax then argmax", "threshold_tuning": False,
            "checkpoint_selection_rule": record["checkpoint_rule"],
            "checkpoint": summary["selected_candidate_checkpoint"],
            "protocol_hash": record["protocol_hash"], "manifest_version_hash": record["manifest_version_hash"],
            "packages": record["packages"], "source_hashes": record["source_hashes"],
            "independent_external_evaluation_completed": False}
    else:
        path = REPO_ROOT / "model/results/final_ft004_ensemble/frozen_selection.json"
        package = {"selection_id": "RETAIN-EXISTING-FT004", "status": "frozen_validation_selected_research_model",
            "frozen_at": now_iso(), "existing_frozen_package_path": str(path),
            "existing_frozen_package_sha256": sha256_file(path), "package_definition": json.loads(path.read_text(encoding="utf-8")),
            "ft011_protocol_hash": record["protocol_hash"], "independent_external_evaluation_completed": False}
    atomic_write_json(ROOT / "final_decision.json", summary)
    atomic_write_json(ROOT / "frozen_selection.json", package)
    atomic_write_json(ROOT / "progress.json", {"status": "terminal_decision", "decision": summary["decision"],
        "tuning_phase_closed": True, "analysis_path": summary["output_directory"], "updated_at": now_iso(),
        "next_step": "independent evaluation of the frozen system"})


def run(record, smoke=False, resume=False):
    from analyze_ft011 import analyze_candidate
    from multiclass.ft011_training import run_ft011

    policy = SafetyPolicy(**record["safety_policy"])
    watchdog = SafetyWatchdog(ROOT / "monitoring", policy,
        record["checkpoint_disk_estimate_bytes"], REPO_ROOT).start()
    try:
        if watchdog.poll()["action"] != "continue":
            raise RuntimeError(f"Launch protection gate: {watchdog.poll()}")
        if smoke:
            watchdog.simulate_once(reason="gpu_temperature", after_polls=4, smoke=True)
            config = load_config(CONTROL_CONFIG, {"epochs": 1, "smoke_test": True,
                "max_train_samples": 16, "max_val_samples": 16, "tensorboard": False,
                "run_id": "ft011_gpu_smoke"})
            metrics = run_ft011(config, ROOT / "smoke", watchdog, protocol_hash=record["protocol_hash"], smoke=True)
            atomic_write_json(ROOT / "smoke_verified.json", {"status": "passed", "at": now_iso(),
                "smoke_metrics_are_execution_checks_only": True, "protocol_hash": record["protocol_hash"],
                "initialization": json.loads((ROOT / "smoke/initialization.json").read_text(encoding="utf-8"))})
            print("GPU smoke and simulated checkpoint/cooling/resume completed", flush=True)
            return
        if not (ROOT / "smoke_verified.json").exists():
            raise RuntimeError("Verified protected GPU smoke is required before full candidate training")
        screening_result = None
        for seed in SEEDS:
            if seed != 42 and not screening_result["screening_passed"]:
                print("Retain FT-004: seed-42 gate failed; remaining seeds are not run", flush=True)
                return
            candidate = json.loads((ROOT / "configs" / f"seed{seed}.json").read_text(encoding="utf-8"))
            config = load_config(CONTROL_CONFIG, {"seed": seed, "run_id": f"ft011_densenet201_differential_lr_s{seed}"})
            run_dir = ROOT / "runs" / config.run_id
            status_file = run_dir / "status.json"
            if status_file.exists():
                status = json.loads(status_file.read_text(encoding="utf-8"))
                if status.get("status") == "completed":
                    if status.get("protocol_hash") != record["protocol_hash"]:
                        raise RuntimeError("Completed candidate protocol mismatch")
                    if seed == 42:
                        screening_result = analyze_candidate(ROOT)
                    continue
                if not resume:
                    raise RuntimeError("Interrupted candidate requires --resume from full state")
            atomic_write_json(ROOT / "progress.json", {"status": "training", "seed": seed,
                "run_dir": str(run_dir), "updated_at": now_iso()})
            run_dir.mkdir(parents=True, exist_ok=True)
            candidate_path = run_dir / "candidate_configuration.json"
            if candidate_path.exists():
                if json.loads(candidate_path.read_text(encoding="utf-8")) != candidate:
                    raise RuntimeError("Existing candidate configuration differs from frozen settings")
            else:
                atomic_write_json(candidate_path, candidate)
            print(f"Starting FT-011 seed {seed}: backbone 0.00001, classifier 0.0001", flush=True)
            run_ft011(config, run_dir, watchdog, protocol_hash=record["protocol_hash"],
                      resume=bool(status_file.exists()))
            if seed == 42:
                screening_result = analyze_candidate(ROOT)
                print(json.dumps(screening_result, indent=2), flush=True)
                if not screening_result["screening_passed"]:
                    close_phase(screening_result, record)
                    return
        close_phase(analyze_candidate(ROOT, terminal="completed"), record)
    finally:
        atomic_write_json(ROOT / "monitoring/final_telemetry_summary.json", watchdog.status)
        watchdog.close()


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    record = prepare()
    verify(record)
    if not args.prepare:
        run(record, smoke=args.smoke, resume=args.resume)


if __name__ == "__main__":
    main()
