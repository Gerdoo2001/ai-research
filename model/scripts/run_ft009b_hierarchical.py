"""Contract-test, train, and evaluate the frozen FT-009B two-stage classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import _multiclass_bootstrap  # noqa: F401
from multiclass.config import ExperimentConfig, REPO_ROOT, load_config
from multiclass.hierarchical import evaluate_hierarchical_validation, validate_stage_pair
from multiclass.tracking import atomic_write_json, now_iso
from multiclass.trainer import run_training


DEFAULT_PROTOCOL = REPO_ROOT / "model/configs/fine_tuning/ft009b_hierarchical_protocol.json"
CONTRACT_DIR = REPO_ROOT / "model/results/ft009b_contracts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--contract-test",
        action="store_true",
        help="Run the dedicated unit suite plus two-stage smoke training and write the authorization marker.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_protocol(path: Path) -> tuple[dict[str, Any], ExperimentConfig, ExperimentConfig]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != "FT-009B-HIERARCHICAL-S42":
        raise ValueError("unexpected FT-009B protocol identity")
    if payload.get("locked_test_accessed") is not False:
        raise PermissionError("the FT-009B protocol must certify locked-test isolation")
    stage1 = load_config(resolve_repo_path(payload["stage1_config"]))
    stage2 = load_config(resolve_repo_path(payload["stage2_config"]))
    validate_stage_pair(stage1, stage2)
    return payload, stage1, stage2


def smoke_variant(config: ExperimentConfig, stage_name: str) -> ExperimentConfig:
    raw = config.as_dict()
    raw.update(
        {
            "weights": "none",
            "input_size": 64,
            "batch_size": 3,
            "epochs": 1,
            "early_stopping_patience": 2,
            "tensorboard": False,
            "smoke_test": True,
            "max_train_samples": 12,
            "max_val_samples": 12,
            "run_id": f"smoke_ft009b_{stage_name}_s42",
        }
    )
    return ExperimentConfig(**raw)


def component_run_dir(config: ExperimentConfig) -> Path:
    return config.results_path / config.effective_run_id()


def run_component(config: ExperimentConfig) -> dict[str, Any]:
    status_path = component_run_dir(config) / "status.json"
    resume = False
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "completed" and status.get("config_hash") == config.config_hash:
            return status
        if status.get("status") == "interrupted" and status.get("config_hash") == config.config_hash:
            resume = True
        elif status.get("config_hash") != config.config_hash:
            raise RuntimeError(f"existing run ID has a different configuration hash: {status_path}")
        elif status.get("status") in {"running", "failed"}:
            raise RuntimeError(f"component is not safely resumable from status={status.get('status')}: {status_path}")
    return run_training(config, resume=resume)


def apply_seed42_gate(metrics: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    observed = {
        "macro_f1": float(metrics["macro_f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "viral_recall": float(metrics["per_class"]["Viral"]["recall"]),
        "normal_recall": float(metrics["per_class"]["Normal"]["recall"]),
    }
    checks = {
        "macro_f1": observed["macro_f1"] >= float(gate["macro_f1_minimum"]),
        "balanced_accuracy": observed["balanced_accuracy"] >= float(gate["balanced_accuracy_minimum"]),
        "viral_recall": observed["viral_recall"] >= float(gate["viral_recall_minimum"]),
        "normal_recall": observed["normal_recall"] >= float(gate["normal_recall_minimum"]),
    }
    return {
        "gate_type": gate["type"],
        "observed": observed,
        "minimums": {
            "macro_f1": gate["macro_f1_minimum"],
            "balanced_accuracy": gate["balanced_accuracy_minimum"],
            "viral_recall": gate["viral_recall_minimum"],
            "normal_recall": gate["normal_recall_minimum"],
        },
        "checks": checks,
        "passed": all(checks.values()),
        "effect": "authorize seeds 123 and 2026 only" if all(checks.values()) else "stop FT-009B after seed 42",
    }


def run_contract_tests(
    protocol_path: Path,
    protocol: dict[str, Any],
    stage1: ExperimentConfig,
    stage2: ExperimentConfig,
) -> int:
    test_command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "model/tests",
        "-p",
        "test_ft009b_hierarchical.py",
        "-v",
    ]
    completed = subprocess.run(test_command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError("FT-009B dedicated unit contracts failed")

    smoke1 = smoke_variant(stage1, "stage1_normal_pneumonia")
    smoke2 = smoke_variant(stage2, "stage2_viral_bacterial")
    validate_stage_pair(smoke1, smoke2)
    status1 = run_component(smoke1)
    status2 = run_component(smoke2)
    smoke_output = REPO_ROOT / "model/results/smoke_ft009b_hierarchical_s42"
    result = evaluate_hierarchical_validation(
        smoke1,
        smoke2,
        component_run_dir(smoke1),
        component_run_dir(smoke2),
        smoke_output,
        max_rows=12,
    )
    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
    authorization = {
        "schema_version": 1,
        "created_at": now_iso(),
        "experiment_id": protocol["experiment_id"],
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "unit_tests_passed": True,
        "smoke_test_passed": True,
        "smoke_validation_rows": result["validation_rows"],
        "stage1_smoke_status": status1["status"],
        "stage2_smoke_status": status2["status"],
        "locked_test_accessed": False,
        "training_authorized": True,
        "authorization_scope": "FT-009B seed-42 development training only; no locked-test access and no multi-seed escalation",
    }
    atomic_write_json(CONTRACT_DIR / "authorization.json", authorization)
    print(json.dumps(authorization, indent=2))
    return 0


def require_authorization(protocol_path: Path) -> dict[str, Any]:
    path = CONTRACT_DIR / "authorization.json"
    if not path.exists():
        raise PermissionError("FT-009B training is blocked until --contract-test succeeds")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("training_authorized") is not True:
        raise PermissionError("FT-009B training authorization is false")
    if payload.get("locked_test_accessed") is not False:
        raise PermissionError("invalid FT-009B authorization: locked-test isolation is not certified")
    if payload.get("protocol_sha256") != sha256_file(protocol_path):
        raise PermissionError("FT-009B protocol changed after contract authorization")
    return payload


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol, stage1, stage2 = load_protocol(protocol_path)
    if args.contract_test:
        return run_contract_tests(protocol_path, protocol, stage1, stage2)

    authorization = require_authorization(protocol_path)
    status1 = run_component(stage1)
    status2 = run_component(stage2)
    output_dir = REPO_ROOT / "model/results/ft009b_hierarchical_s42"
    result = evaluate_hierarchical_validation(
        stage1,
        stage2,
        component_run_dir(stage1),
        component_run_dir(stage2),
        output_dir,
    )
    gate = apply_seed42_gate(result["metrics"], protocol["seed42_advancement_gate"])
    final = {
        "schema_version": 1,
        "created_at": now_iso(),
        "experiment_id": protocol["experiment_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "authorization": authorization,
        "stage1_status": status1,
        "stage2_status": status2,
        "hierarchical_summary": result,
        "seed42_advancement_gate": gate,
        "locked_test_accessed": False,
    }
    atomic_write_json(output_dir / "ft009b_result.json", final)
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
