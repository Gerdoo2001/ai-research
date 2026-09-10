"""Run the FT-010A label-smoothing confirmation at seeds 123 and 2026.

The seed-42 run is reused as the frozen reference. This controller invokes only
the normal training command, which reads the locked training and validation
manifests; it never invokes locked-test evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
TRAIN_SCRIPT = REPO_ROOT / "model" / "scripts" / "train_multiclass.py"
RUNS_ROOT = REPO_ROOT / "model" / "results" / "runs"
OUTPUT_ROOT = REPO_ROOT / "model" / "results" / "ft010_label_smoothing_multiseed"
PROGRESS_PATH = OUTPUT_ROOT / "progress.json"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
SUMMARY_CSV = OUTPUT_ROOT / "summary.csv"
PROTOCOL_PATH = (
    REPO_ROOT
    / "model"
    / "configs"
    / "fine_tuning"
    / "ft010_label_smoothing_multiseed_protocol.json"
)

REFERENCE = {
    "seed": 42,
    "run_id": "fine_tuning_densenet201_viridis_ge_clahe_label_smoothing005_s42",
    "config": REPO_ROOT
    / "model"
    / "configs"
    / "fine_tuning"
    / "densenet201_viridis_ge_clahe_label_smoothing005_s42.json",
}

TARGETS = [
    {
        "seed": 123,
        "run_id": "confirmation_densenet201_viridis_ge_clahe_label_smoothing005_s123",
        "config": REPO_ROOT
        / "model"
        / "configs"
        / "fine_tuning"
        / "densenet201_viridis_ge_clahe_label_smoothing005_s123.json",
    },
    {
        "seed": 2026,
        "run_id": "confirmation_densenet201_viridis_ge_clahe_label_smoothing005_s2026",
        "config": REPO_ROOT
        / "model"
        / "configs"
        / "fine_tuning"
        / "densenet201_viridis_ge_clahe_label_smoothing005_s2026.json",
    },
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    serialized = json.dumps(payload, indent=2)
    temporary.write_text(serialized, encoding="utf-8")

    # Windows can briefly deny os.replace when a monitor, indexer, or editor has
    # the destination JSON open. Reporting must never terminate a completed or
    # active training run, so retry the atomic replacement before using a safe
    # non-atomic overwrite. If the destination remains locked, retain a
    # ``.latest`` copy and allow the controller to keep training.
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(0.25 * (attempt + 1))

    try:
        path.write_text(serialized, encoding="utf-8")
        temporary.unlink(missing_ok=True)
        return
    except PermissionError:
        latest = path.with_name(f"{path.name}.latest")
        latest.write_text(serialized, encoding="utf-8")
        temporary.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    if protocol.get("experiment_id") != "FT-010A-LABEL-SMOOTHING-MULTISEED":
        raise ValueError("unexpected FT-010A protocol identity")
    if protocol.get("locked_test_accessed") is not False:
        raise PermissionError("FT-010A must certify locked-test isolation")
    expected = [42, 123, 2026]
    if protocol.get("seeds") != expected:
        raise ValueError(f"FT-010A seeds must remain fixed at {expected}")
    reference_metrics = read_json(RUNS_ROOT / REFERENCE["run_id"] / "metrics.json")
    if not reference_metrics or reference_metrics.get("split") != "validation":
        raise FileNotFoundError("completed seed-42 validation reference was not found")

    invariant_fields = [
        "architecture",
        "preprocessing",
        "task_mode",
        "input_size",
        "batch_size",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "loss",
        "label_smoothing",
        "epochs",
        "early_stopping_patience",
        "augmentation",
        "class_weighting",
        "manifest_dir",
    ]
    reference_config = read_json(REFERENCE["config"])
    for target in TARGETS:
        candidate = read_json(target["config"])
        if candidate.get("seed") != target["seed"] or candidate.get("run_id") != target["run_id"]:
            raise ValueError(f"seed/run identity mismatch in {target['config']}")
        for field in invariant_fields:
            if candidate.get(field) != reference_config.get(field):
                raise ValueError(f"uncontrolled FT-010A change in {field}: {target['config']}")
    return protocol


def metrics_row(item: dict[str, Any]) -> dict[str, Any]:
    run_dir = RUNS_ROOT / item["run_id"]
    status = read_json(run_dir / "status.json")
    metrics = read_json(run_dir / "metrics.json")
    per_class = metrics.get("per_class", {})
    return {
        "seed": item["seed"],
        "run_id": item["run_id"],
        "status": status.get("status", "pending"),
        "best_epoch": metrics.get("best_epoch", status.get("best_epoch")),
        "macro_f1": metrics.get("macro_f1"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "accuracy": metrics.get("accuracy"),
        "normal_recall": per_class.get("Normal", {}).get("recall"),
        "viral_recall": per_class.get("Viral", {}).get("recall"),
        "bacterial_recall": per_class.get("Bacterial", {}).get("recall"),
        "training_duration_seconds": metrics.get("training_duration_seconds"),
    }


def save_summary(protocol: dict[str, Any]) -> None:
    rows = [metrics_row(REFERENCE), *(metrics_row(item) for item in TARGETS)]
    macro_f1 = [float(row["macro_f1"]) for row in rows if row["macro_f1"] is not None]
    payload = {
        "schema_version": 1,
        "updated_at": now_iso(),
        "experiment_id": protocol["experiment_id"],
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "selection_split": "validation",
        "locked_test_accessed": False,
        "configuration": protocol["fixed_configuration"],
        "changed_variable": "random seed only",
        "runs": rows,
        "completed_seed_count": len(macro_f1),
        "target_seed_count": 3,
        "mean_macro_f1": statistics.mean(macro_f1) if macro_f1 else None,
        "sample_std_macro_f1": statistics.stdev(macro_f1) if len(macro_f1) > 1 else None,
        "interpretation_boundary": protocol["interpretation_boundary"],
    }
    atomic_json(SUMMARY_PATH, payload)
    with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_target(target: dict[str, Any], protocol: dict[str, Any]) -> bool:
    run_dir = RUNS_ROOT / target["run_id"]
    status = read_json(run_dir / "status.json")
    if status.get("status") == "completed":
        save_summary(protocol)
        return True

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stdout_path = OUTPUT_ROOT / f"seed_{target['seed']}.stdout.log"
    stderr_path = OUTPUT_ROOT / f"seed_{target['seed']}.stderr.log"
    command = [str(PYTHON), str(TRAIN_SCRIPT), "--config", str(target["config"])]
    if (run_dir / "last_checkpoint.pt").exists():
        command.append("--resume")

    with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        while process.poll() is None:
            atomic_json(
                PROGRESS_PATH,
                {
                    "schema_version": 1,
                    "experiment_id": protocol["experiment_id"],
                    "status": "running",
                    "controller_pid": os.getpid(),
                    "trainer_pid": process.pid,
                    "current_seed": target["seed"],
                    "current_run_id": target["run_id"],
                    "updated_at": now_iso(),
                    "locked_test_accessed": False,
                },
            )
            save_summary(protocol)
            time.sleep(2)
        return_code = process.returncode

    save_summary(protocol)
    return return_code == 0 and read_json(run_dir / "status.json").get("status") == "completed"


def main() -> int:
    protocol = validate_protocol()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(
        PROGRESS_PATH,
        {
            "schema_version": 1,
            "experiment_id": protocol["experiment_id"],
            "status": "initializing",
            "controller_pid": os.getpid(),
            "started_at": now_iso(),
            "locked_test_accessed": False,
        },
    )
    save_summary(protocol)
    for target in TARGETS:
        if not run_target(target, protocol):
            atomic_json(
                PROGRESS_PATH,
                {
                    "schema_version": 1,
                    "experiment_id": protocol["experiment_id"],
                    "status": "failed",
                    "controller_pid": os.getpid(),
                    "current_seed": target["seed"],
                    "current_run_id": target["run_id"],
                    "updated_at": now_iso(),
                    "locked_test_accessed": False,
                },
            )
            return 1

    save_summary(protocol)
    atomic_json(
        PROGRESS_PATH,
        {
            "schema_version": 1,
            "experiment_id": protocol["experiment_id"],
            "status": "completed",
            "controller_pid": os.getpid(),
            "completed_at": now_iso(),
            "locked_test_accessed": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
