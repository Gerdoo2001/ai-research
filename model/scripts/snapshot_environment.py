"""Capture a reproducible, non-mutating environment snapshot.

This script uses only the standard library for discovery. Optional ML frameworks
are imported defensively so a missing package is recorded instead of aborting the
snapshot.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "model"
OUTPUT_PATH = MODEL_ROOT / "results" / "environment.json"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except FileNotFoundError as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            versions[name] = distribution.version
    return dict(sorted(versions.items(), key=lambda item: item[0].lower()))


def tensorflow_runtime() -> dict[str, Any]:
    try:
        import tensorflow as tf

        build_info = tf.sysconfig.get_build_info()
        return {
            "available": True,
            "version": tf.__version__,
            "built_with_cuda": bool(tf.test.is_built_with_cuda()),
            "physical_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
            "logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
            "build_info": build_info,
        }
    except Exception as exc:  # pragma: no cover - environment-specific
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def torch_runtime() -> dict[str, Any]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        return {
            "available": True,
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
            if cuda_available
            else [],
        }
    except Exception as exc:  # pragma: no cover - environment-specific
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def directory_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "file_count": 0, "bytes": 0}
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "exists": True,
        "file_count": len(files),
        "bytes": sum(item.stat().st_size for item in files),
    }


def main() -> int:
    disk = shutil.disk_usage(REPO_ROOT)
    git_branch = run(["git", "branch", "--show-current"])
    git_head = run(["git", "rev-parse", "HEAD"])
    git_status = run(["git", "status", "--porcelain=v1"])
    pip_check = run([sys.executable, "-m", "pip", "check"])
    nvidia_smi = run(["nvidia-smi"])
    nvidia_query = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )

    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": now_iso(),
        "repository": {
            "root": str(REPO_ROOT),
            "branch": git_branch["stdout"],
            "head": git_head["stdout"],
            "status_porcelain": git_status["stdout"].splitlines(),
        },
        "system": {
            "platform": platform.platform(),
            "os_name": os.name,
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "timezone": datetime.now().astimezone().tzname(),
            "disk": {
                "path": str(REPO_ROOT.drive or REPO_ROOT.anchor),
                "total_bytes": disk.total,
                "used_bytes": disk.used,
                "free_bytes": disk.free,
            },
        },
        "packages": package_versions(),
        "pip_check": pip_check,
        "gpu": {
            "nvidia_smi": nvidia_smi,
            "query": nvidia_query,
            "tensorflow": tensorflow_runtime(),
            "torch": torch_runtime(),
        },
        "directories": {
            "model": directory_summary(MODEL_ROOT),
            "source_dataset": directory_summary(MODEL_ROOT / "datasets" / "chest_xray"),
            "legacy_augmented_data": directory_summary(MODEL_ROOT / "augmented_data"),
            "legacy_outputs": directory_summary(MODEL_ROOT / "outputs"),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote environment snapshot: {OUTPUT_PATH}")
    print(f"TensorFlow GPU visible: {bool(snapshot['gpu']['tensorflow'].get('physical_gpus'))}")
    print(f"PyTorch CUDA available: {bool(snapshot['gpu']['torch'].get('cuda_available'))}")
    print(f"pip check return code: {pip_check['returncode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
