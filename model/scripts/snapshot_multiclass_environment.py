"""Append the isolated PyTorch/CUDA profile to the Phase 0 environment JSON."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "model/results/environment.json"
PACKAGE_NAMES = [
    "torch", "torchvision", "ultralytics", "numpy", "pandas", "scikit-learn",
    "scipy", "Pillow", "opencv-python", "matplotlib", "seaborn", "PyYAML",
    "tensorboard", "tqdm", "optuna",
]


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    import torch

    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=False
    )
    environment_bytes = sum(
        path.stat().st_size for path in (REPO_ROOT / ".venv").rglob("*") if path.is_file()
    )
    disk = shutil.disk_usage(REPO_ROOT)
    profile = {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "isolated PyTorch CUDA environment for parallel three-class pipeline",
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "requirements_file": str(REPO_ROOT / "model/requirements.multiclass-cu121.txt"),
        "environment_bytes": environment_bytes,
        "packages": {name: version(name) for name in PACKAGE_NAMES},
        "pip_check": {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        },
        "gpu": {
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ],
        },
        "disk_free_bytes": disk.free,
        "legacy_tensorflow_environment_modified": False,
    }
    payload = json.loads(OUTPUT.read_text(encoding="utf-8")) if OUTPUT.exists() else {"schema_version": 1}
    payload["multiclass_environment"] = profile
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(profile, indent=2))
    return 0 if completed.returncode == 0 and torch.cuda.is_available() else 1


if __name__ == "__main__":
    raise SystemExit(main())
