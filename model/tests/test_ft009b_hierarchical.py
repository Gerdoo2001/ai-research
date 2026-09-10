"""Dedicated fail-closed contracts for FT-009B hierarchical classification."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model/src"))
sys.path.insert(0, str(REPO_ROOT / "model/scripts"))

from multiclass.config import ExperimentConfig, load_config  # noqa: E402
from multiclass.data import (  # noqa: E402
    build_evaluation_loader,
    filter_rows_for_task,
    read_manifest,
    remap_class_index,
)
from multiclass.hierarchical import (  # noqa: E402
    assert_exact_path_alignment,
    compose_hierarchical_probabilities,
    validate_probability_matrix,
    validate_stage_pair,
)
from run_ft009b_hierarchical import apply_seed42_gate, load_protocol  # noqa: E402


PROTOCOL = REPO_ROOT / "model/configs/fine_tuning/ft009b_hierarchical_protocol.json"


class StageConfigurationTests(unittest.TestCase):
    def test_frozen_protocol_and_stage_pair(self):
        protocol, stage1, stage2 = load_protocol(PROTOCOL)
        self.assertFalse(protocol["locked_test_accessed"])
        self.assertEqual(stage1.resolved_class_names, ("Normal", "Pneumonia"))
        self.assertEqual(stage2.resolved_class_names, ("Viral", "Bacterial"))
        validate_stage_pair(stage1, stage2)

    def test_stage_pair_rejects_uncontrolled_configuration_drift(self):
        _, stage1, stage2 = load_protocol(PROTOCOL)
        changed = stage2.as_dict()
        changed["learning_rate"] = 2e-4
        with self.assertRaises(ValueError):
            validate_stage_pair(stage1, ExperimentConfig(**changed))

    def test_invalid_stage2_class_contract_is_rejected(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(task_mode="viral_bacterial", class_names=["Bacterial", "Viral"])

    def test_historical_ft_ref_hash_remains_unchanged(self):
        config = load_config(
            REPO_ROOT
            / "model/results/runs/staged_densenet201_pseudo_color_then_histogram_equalization_then_clahe_s42/config.json"
        )
        self.assertEqual(
            config.config_hash,
            "ef029d89d066687af73d2973be80caf5b16bb3d648000f617b77a73306ab2691",
        )


class StageDataTests(unittest.TestCase):
    def test_stage2_filters_normal_without_rewriting_locked_manifests(self):
        train = read_manifest(REPO_ROOT / "model/data_manifests/train_manifest.csv")
        validation = read_manifest(REPO_ROOT / "model/data_manifests/val_manifest.csv")
        stage2_train = filter_rows_for_task(train, "viral_bacterial")
        stage2_validation = filter_rows_for_task(validation, "viral_bacterial")
        self.assertEqual(len(stage2_train), 2971)
        self.assertEqual(len(stage2_validation), 637)
        self.assertEqual({row["class_name"] for row in stage2_train}, {"Viral", "Bacterial"})
        self.assertEqual(remap_class_index(1, "viral_bacterial"), 0)
        self.assertEqual(remap_class_index(2, "viral_bacterial"), 1)
        with self.assertRaises(ValueError):
            remap_class_index(0, "viral_bacterial")

    def test_locked_test_remains_fail_closed_for_both_stages(self):
        _, stage1, stage2 = load_protocol(PROTOCOL)
        with self.assertRaises(PermissionError):
            build_evaluation_loader(stage1, "test")
        with self.assertRaises(PermissionError):
            build_evaluation_loader(stage2, "test")


class ProbabilityCompositionTests(unittest.TestCase):
    def test_probability_products_are_exact_and_sum_to_one(self):
        stage1 = np.asarray([[0.8, 0.2], [0.1, 0.9]])
        stage2 = np.asarray([[0.25, 0.75], [0.6, 0.4]])
        combined = compose_hierarchical_probabilities(stage1, stage2)
        np.testing.assert_allclose(combined, [[0.8, 0.05, 0.15], [0.1, 0.54, 0.36]])
        np.testing.assert_allclose(combined.sum(axis=1), 1.0)
        self.assertEqual(combined.argmax(axis=1).tolist(), [0, 1])

    def test_malformed_probabilities_fail_instead_of_being_silently_repaired(self):
        with self.assertRaises(ValueError):
            validate_probability_matrix(np.asarray([[0.4, 0.4]]), "bad")
        with self.assertRaises(ValueError):
            compose_hierarchical_probabilities(
                np.asarray([[0.5, 0.5]]),
                np.asarray([[np.nan, np.nan]]),
            )

    def test_float32_softmax_precision_is_accepted(self):
        import torch

        logits = torch.tensor([[7.25, -1.5], [-2.0, 3.75]], dtype=torch.float16)
        probabilities = torch.softmax(logits.float(), dim=1).numpy()
        validated = validate_probability_matrix(probabilities, "float32_softmax")
        np.testing.assert_allclose(validated.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)

    def test_path_alignment_is_order_sensitive(self):
        assert_exact_path_alignment(["a", "b"], ["a", "b"], "stage")
        with self.assertRaises(RuntimeError):
            assert_exact_path_alignment(["a", "b"], ["b", "a"], "stage")


class AdvancementGateTests(unittest.TestCase):
    def test_all_seed42_requirements_must_pass(self):
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        gate = protocol["seed42_advancement_gate"]
        metrics = {
            "macro_f1": gate["macro_f1_minimum"],
            "balanced_accuracy": gate["balanced_accuracy_minimum"],
            "per_class": {
                "Viral": {"recall": gate["viral_recall_minimum"]},
                "Normal": {"recall": gate["normal_recall_minimum"]},
            },
        }
        self.assertTrue(apply_seed42_gate(metrics, gate)["passed"])
        metrics["per_class"]["Viral"]["recall"] -= 1e-6
        result = apply_seed42_gate(metrics, gate)
        self.assertFalse(result["passed"])
        self.assertEqual(result["effect"], "stop FT-009B after seed 42")


if __name__ == "__main__":
    unittest.main(verbosity=2)
