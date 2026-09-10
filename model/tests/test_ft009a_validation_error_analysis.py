"""Contracts for the matched FT-009A validation error analysis."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model/scripts"))

from analyze_ft009a_validation_errors import (  # noqa: E402
    SOURCES,
    align_predictions,
    error_type,
    load_predictions,
    sha256,
)


class Ft009aValidationErrorContracts(unittest.TestCase):
    def test_frozen_prediction_hashes_and_rows(self) -> None:
        for source in SOURCES.values():
            self.assertTrue(source["predictions"].exists())
            self.assertEqual(sha256(source["predictions"]), source["expected_sha256"])
            self.assertEqual(len(load_predictions(source["predictions"])), 874)

    def test_all_three_sources_have_identical_cohort_identity(self) -> None:
        frames = {key: load_predictions(source["predictions"]) for key, source in SOURCES.items()}
        aligned = align_predictions(frames)
        self.assertEqual(len(aligned), 874)
        self.assertEqual(aligned["sample_id"].nunique(), 874)

    def test_error_type_decomposes_the_three_class_task(self) -> None:
        self.assertEqual(error_type(0, 0), "correct")
        self.assertEqual(error_type(1, 2), "viral_bacterial_etiology")
        self.assertEqual(error_type(2, 1), "viral_bacterial_etiology")
        self.assertEqual(error_type(0, 1), "normal_pneumonia_boundary")
        self.assertEqual(error_type(2, 0), "normal_pneumonia_boundary")

    def test_alignment_fails_on_label_mismatch(self) -> None:
        frame = load_predictions(next(iter(SOURCES.values()))["predictions"]).head(2)
        changed = frame.copy()
        changed.loc[0, "relative_path"] = "different.jpeg"
        with self.assertRaises(ValueError):
            align_predictions({"first": frame, "changed": changed})

    def test_loading_fails_on_duplicate_sample_id(self) -> None:
        source = next(iter(SOURCES.values()))["predictions"]
        frame = pd.read_csv(source).head(2)
        frame.loc[1, "sample_id"] = frame.loc[0, "sample_id"]
        temporary = source.parent / "_ft009a_duplicate_contract.csv"
        try:
            frame.to_csv(temporary, index=False)
            with self.assertRaises(ValueError):
                load_predictions(temporary)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
