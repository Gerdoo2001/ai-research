"""Safe DOCX output helpers for report generators."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def versioned_path(path: Path, *, timestamp: str | None = None) -> Path:
    stamp = timestamp or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def save_docx_with_fallback(document, requested_path: Path) -> Path:
    """Save at the requested path, or at a timestamped sibling if it is locked."""

    requested_path = requested_path.resolve()
    requested_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        document.save(requested_path)
        return requested_path
    except PermissionError:
        fallback = versioned_path(requested_path)
        document.save(fallback)
        return fallback
