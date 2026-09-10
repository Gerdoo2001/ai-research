"""Fail-closed contracts for the frozen FT-004 ensemble package."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model/src"))

from multiclass.inference import average_member_probabilities  # noqa: E402


SELECTION = REPO_ROOT / "model/results/final_ft004_ensemble/frozen_selection.json"


class FrozenSelectionTests(unittest.TestCase):
    def test_frozen_selection_contains_exactly_three_seed42_members(self):
        selection = json.loads(SELECTION.read_text(encoding="utf-8"))
        self.assertEqual(selection["status"], "frozen_validation_selected_research_model")
        self.assertEqual(selection["deployable_member_count"], 3)
        self.assertEqual(selection["deployable_seed"], 42)
        self.assertEqual(len(selection["members"]), 3)
        self.assertEqual({member["seed"] for member in selection["members"]}, {42})
        self.assertFalse(selection["locked_test_accessed_for_this_selection"])

    def test_all_frozen_checkpoint_hashes_match(self):
        import hashlib

        selection = json.loads(SELECTION.read_text(encoding="utf-8"))
        for member in selection["members"]:
            path = Path(member["checkpoint_path"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, member["checkpoint_sha256"])


class ProbabilityRuleTests(unittest.TestCase):
    def test_equal_arithmetic_mean_and_argmax_contract(self):
        result = average_member_probabilities(
            [
                {"Normal": 0.8, "Viral": 0.1, "Bacterial": 0.1},
                {"Normal": 0.2, "Viral": 0.7, "Bacterial": 0.1},
                {"Normal": 0.3, "Viral": 0.3, "Bacterial": 0.4},
            ]
        )
        self.assertAlmostEqual(result["Normal"], 1.3 / 3)
        self.assertAlmostEqual(result["Viral"], 1.1 / 3)
        self.assertAlmostEqual(result["Bacterial"], 0.6 / 3)
        self.assertEqual(max(result, key=result.__getitem__), "Normal")

    def test_member_count_and_probability_validation_fail_closed(self):
        valid = {"Normal": 0.8, "Viral": 0.1, "Bacterial": 0.1}
        with self.assertRaises(ValueError):
            average_member_probabilities([valid, valid])
        with self.assertRaises(ValueError):
            average_member_probabilities(
                [valid, valid, {"Normal": 0.9, "Viral": 0.9, "Bacterial": -0.8}]
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
