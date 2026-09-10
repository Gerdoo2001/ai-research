"""Tests for safe report output handling."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model" / "scripts"))

from document_io import save_docx_with_fallback  # noqa: E402


class LockedDocument:
    def __init__(self, locked_path: Path) -> None:
        self.locked_path = locked_path.resolve()
        self.calls: list[Path] = []

    def save(self, path) -> None:
        resolved = Path(path).resolve()
        self.calls.append(resolved)
        if resolved == self.locked_path:
            raise PermissionError("simulated open Word document")
        resolved.write_bytes(b"fallback")


class DocumentIOTests(unittest.TestCase):
    def test_locked_docx_uses_timestamped_sibling(self):
        with tempfile.TemporaryDirectory() as temp:
            requested = Path(temp) / "report.docx"
            document = LockedDocument(requested)
            saved = save_docx_with_fallback(document, requested)
            self.assertNotEqual(saved, requested)
            self.assertTrue(saved.exists())
            self.assertEqual(saved.suffix, ".docx")
            self.assertTrue(saved.stem.startswith("report_"))
            self.assertEqual(len(document.calls), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
