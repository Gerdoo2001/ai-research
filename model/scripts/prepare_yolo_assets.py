"""Prepare pinned YOLO classification source/weights outside tracked source."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

import _multiclass_bootstrap
from multiclass.tracking import atomic_write_json, now_iso


YOLOV5_COMMIT = "ed3bdee6bbe696ce9b35f32efb261e83f72ce03c"
YOLOV5_REPO = "https://github.com/ultralytics/yolov5.git"
YOLOV5_WEIGHTS_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-cls.pt"
YOLOV8_WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-cls.pt"
CACHE_ROOT = _multiclass_bootstrap.REPO_ROOT / ".research_cache"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 100_000:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(destination)


def main() -> int:
    repo = CACHE_ROOT / "yolov5"
    weights = CACHE_ROOT / "weights"
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--filter=blob:none", YOLOV5_REPO, str(repo))
    run("git", "fetch", "origin", YOLOV5_COMMIT, cwd=repo)
    run("git", "checkout", "--detach", YOLOV5_COMMIT, cwd=repo)
    yolov5_weight = weights / "yolov5s-cls-v7.0.pt"
    yolov8_weight = weights / "yolov8n-cls-v8.3.0.pt"
    download(YOLOV5_WEIGHTS_URL, yolov5_weight)
    download(YOLOV8_WEIGHTS_URL, yolov8_weight)
    payload = {
        "generated_at": now_iso(),
        "yolov5": {
            "repository": YOLOV5_REPO,
            "commit": YOLOV5_COMMIT,
            "source_path": str(repo),
            "weights_url": YOLOV5_WEIGHTS_URL,
            "weights_path": str(yolov5_weight),
            "weights_sha256": sha256(yolov5_weight),
            "weights_bytes": yolov5_weight.stat().st_size,
        },
        "yolov8": {
            "implementation": "ultralytics==8.3.168",
            "weights_url": YOLOV8_WEIGHTS_URL,
            "weights_path": str(yolov8_weight),
            "weights_sha256": sha256(yolov8_weight),
            "weights_bytes": yolov8_weight.stat().st_size,
        },
        "yolov10": {
            "status": "unsupported",
            "reason": (
                "Ultralytics 8.3.168 contains YOLOv10 detection YAMLs with v10Detect heads but no "
                "YOLOv10 classification YAML or classification checkpoint."
            ),
        },
    }
    atomic_write_json(CACHE_ROOT / "yolo_assets.json", payload)
    atomic_write_json(
        _multiclass_bootstrap.REPO_ROOT / "model/results/yolo_support_audit.json", payload
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
