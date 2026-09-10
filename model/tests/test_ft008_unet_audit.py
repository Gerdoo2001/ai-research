"""CPU-safe contracts for the FT-008 U-Net audit queue."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model/scripts"))

from audit_unet_lung_roi import (  # noqa: E402
    center_square,
    largest_components,
    mask_bbox,
    quality_record,
    screening_records,
)


class Ft008GeometryTests(unittest.TestCase):
    def test_center_square_preserves_the_center(self):
        image = Image.fromarray(np.zeros((80, 120), dtype=np.uint8))
        self.assertEqual(center_square(image).size, (80, 80))

    def test_two_largest_components_preserve_both_lungs_and_remove_fragment(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:52, 5:25] = True
        mask[10:52, 39:59] = True
        mask[1:3, 1:3] = True
        cleaned, components = largest_components(mask, max_components=2)
        self.assertEqual(components, 3)
        self.assertTrue(cleaned[20, 15])
        self.assertTrue(cleaned[20, 49])
        self.assertFalse(cleaned[1, 1])

    def test_one_component_mode_reproduces_the_superseded_behavior(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:52, 5:25] = True
        mask[10:52, 39:59] = True
        cleaned, components = largest_components(mask, max_components=1)
        self.assertEqual(components, 2)
        self.assertNotEqual(bool(cleaned[20, 15]), bool(cleaned[20, 49]))

    def test_empty_mask_fails_closed(self):
        mask = np.zeros((64, 64), dtype=bool)
        record = quality_record(mask, mask, 0.18)
        self.assertIsNone(record["bbox"])
        self.assertFalse(record["automated_pass"])
        self.assertIn("empty_lung_mask", record["quality_flags"])

    def test_bbox_margin_is_bounded(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:90, 5:95] = True
        self.assertEqual(mask_bbox(mask, 0.18), (0, 0, 100, 100))


class Ft008DataContracts(unittest.TestCase):
    def test_source_is_the_locked_ft002c_validation_screening_cohort(self):
        path = REPO_ROOT / "model/results/roi_pspnet_independent_audit/summary.json"
        source = json.loads(path.read_text(encoding="utf-8"))
        records = screening_records(source)
        self.assertEqual(len(records), 90)
        self.assertFalse(source["locked_test_accessed"])

    def test_both_frozen_legacy_artifacts_exist(self):
        for relative_path in (
            "model/outputs/cxr_reg_weights.best.hdf5",
            "model/outputs/trained_model.hdf5",
        ):
            self.assertTrue((REPO_ROOT / relative_path).is_file())


if __name__ == "__main__":
    unittest.main()
