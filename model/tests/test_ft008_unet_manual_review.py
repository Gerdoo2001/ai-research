"""Contracts for FT-008A2 persistent two-model manual review."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model/scripts"))

from prepare_unet_roi_manual_review import (  # noqa: E402
    DECISIONS,
    MODEL_KEYS,
    SUMMARY_PATH,
    automatic_manual_agreement,
    model_is_accepted,
    summarize,
    validate_summary,
)


class Ft008a2ManualReviewContracts(unittest.TestCase):
    def test_decision_mapping_is_model_specific(self) -> None:
        self.assertTrue(model_is_accepted("classic_only", MODEL_KEYS[0]))
        self.assertFalse(model_is_accepted("classic_only", MODEL_KEYS[1]))
        self.assertFalse(model_is_accepted("legacy_only", MODEL_KEYS[0]))
        self.assertTrue(model_is_accepted("legacy_only", MODEL_KEYS[1]))
        self.assertTrue(model_is_accepted("both", MODEL_KEYS[0]))
        self.assertTrue(model_is_accepted("both", MODEL_KEYS[1]))
        self.assertFalse(model_is_accepted("neither", MODEL_KEYS[0]))
        self.assertFalse(model_is_accepted("neither", MODEL_KEYS[1]))

    def test_summary_counts_each_model_and_class(self) -> None:
        records = [
            {"audit_index": 1, "class_name": "Normal", "user_decision": "classic_only"},
            {"audit_index": 2, "class_name": "Viral", "user_decision": "both"},
            {"audit_index": 3, "class_name": "Bacterial", "user_decision": "neither"},
            {"audit_index": 4, "class_name": "Bacterial", "user_decision": None},
        ]
        result = summarize(records)
        self.assertEqual(result["reviewed_images"], 3)
        self.assertEqual(result["pending_count"], 1)
        self.assertEqual(result["next_pending_index"], 4)
        self.assertEqual(result["per_model"][MODEL_KEYS[0]]["accepted"], 2)
        self.assertEqual(result["per_model"][MODEL_KEYS[1]]["accepted"], 1)
        self.assertEqual(result["decision_counts"], {key: int(key in {"classic_only", "both", "neither"}) for key in DECISIONS})

    def test_source_is_complete_corrected_screen(self) -> None:
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        validate_summary(summary)
        self.assertEqual(len(summary["records"]), 90)
        self.assertEqual(summary["max_components"], 2)
        self.assertFalse(summary["locked_test_accessed"])
        self.assertFalse(summary["classifier_training_started"])

    def test_automatic_manual_agreement_table(self) -> None:
        records = [
            {
                "audit_index": 1,
                "class_name": "Normal",
                "user_decision": "classic_only",
                "models": {
                    MODEL_KEYS[0]: {"automated_pass": True},
                    MODEL_KEYS[1]: {"automated_pass": False},
                },
            },
            {
                "audit_index": 2,
                "class_name": "Viral",
                "user_decision": "neither",
                "models": {
                    MODEL_KEYS[0]: {"automated_pass": True},
                    MODEL_KEYS[1]: {"automated_pass": False},
                },
            },
        ]
        result = automatic_manual_agreement(records)
        self.assertEqual(result[MODEL_KEYS[0]]["automatic_pass_human_accept"], 1)
        self.assertEqual(result[MODEL_KEYS[0]]["automatic_pass_human_reject"], 1)
        self.assertEqual(result[MODEL_KEYS[1]]["automatic_flag_human_reject"], 2)

    def test_all_comparison_images_exist(self) -> None:
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        missing = [record["comparison_path"] for record in summary["records"] if not __import__("pathlib").Path(record["comparison_path"]).exists()]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
