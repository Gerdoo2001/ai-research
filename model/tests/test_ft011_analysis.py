"""Decision-boundary, identity and paired-cluster contracts for FT-011."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model/scripts"))
from analyze_ft011 import (
    CLASSES, PROBABILITY_COLUMNS, aggregate_metrics, aligned_predictions,
    analyze_candidate, cluster_strata, draw_cluster_indices, evaluate_gates,
    fast_metrics, paired_bootstrap, canonical_hash, verify_candidate_configuration,
)
from multiclass.metrics import calculate_metrics


def fixture_manifest():
    return [{"relative_path": f"legacy_test/image_{i}.jpeg", "assigned_split": "val",
             "class_name": CLASSES[label], "class_index": str(label),
             "split_group_id": f"proxy_{label}"}
            for i, label in enumerate((0, 0, 1, 1, 2, 2))]


class DecisionGateTests(unittest.TestCase):
    def test_macro_gate_uses_absolute_full_precision_increment(self):
        threshold = 0.8563107953930739 + 0.005
        gate = [{"metric": "macro_f1", "operator": ">=", "threshold": threshold}]
        self.assertFalse(evaluate_gates({"macro_f1": 0.86131}, gate)[0]["passed"])
        self.assertTrue(evaluate_gates({"macro_f1": threshold}, gate)[0]["passed"])
        self.assertAlmostEqual(threshold - 0.8563107953930739, 0.005)

    def test_replacement_strict_tie_fails_while_recall_tie_passes(self):
        gates = [{"metric": "macro_f1", "operator": ">", "threshold": 0.853},
                 {"metric": "viral_recall", "operator": ">=", "threshold": 0.66}]
        result = evaluate_gates({"macro_f1": 0.853, "viral_recall": 0.66}, gates)
        self.assertFalse(result[0]["passed"])
        self.assertTrue(result[1]["passed"])

    def test_nonfinite_gate_cannot_pass(self):
        with self.assertRaises(ValueError):
            evaluate_gates({"macro_f1": float("nan")}, [{"metric": "macro_f1", "operator": ">", "threshold": 0.8}])

    def test_candidate_outcomes_require_frozen_controls_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text(json.dumps({"experiment_id": "FT011"}))
            with self.assertRaisesRegex(RuntimeError, "freeze control artifacts"):
                analyze_candidate(root)


class ValidationIdentityTests(unittest.TestCase):
    def write_predictions(self, path, manifest, transform=lambda row: row):
        fields = ("sample_id", "relative_path", "true_index", "true_class", *PROBABILITY_COLUMNS)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, assigned in reversed(list(enumerate(manifest))):
                label = int(assigned["class_index"])
                row = {"sample_id": f"validation_{index:05d}", "relative_path": assigned["relative_path"],
                       "true_index": label, "true_class": CLASSES[label],
                       **{name: float(i == label) for i, name in enumerate(PROBABILITY_COLUMNS)}}
                writer.writerow(transform(row))

    def test_aligns_manifest_identity_regardless_csv_order_or_legacy_prefix(self):
        manifest = fixture_manifest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            self.write_predictions(path, manifest)
            rows, labels, probabilities = aligned_predictions(path, manifest)
            self.assertEqual(rows[0]["sample_id"], "validation_00000")
            np.testing.assert_array_equal(labels, [0, 0, 1, 1, 2, 2])
            np.testing.assert_array_equal(probabilities.argmax(axis=1), labels)

    def test_matching_label_alone_cannot_hide_sample_id_mismatch(self):
        manifest = fixture_manifest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            self.write_predictions(path, manifest, lambda row: {**row, "sample_id": "validation_99999"} if row["sample_id"] == "validation_00000" else row)
            with self.assertRaisesRegex(RuntimeError, "sample_id/relative_path/label"):
                aligned_predictions(path, manifest)


    def test_nonfinite_probabilities_fail(self):
        manifest = fixture_manifest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            self.write_predictions(path, manifest, lambda row: {**row, "probability_normal": "nan"})
            with self.assertRaisesRegex(RuntimeError, "invalid validation probabilities"):
                aligned_predictions(path, manifest)


class CandidateConfigurationTests(unittest.TestCase):
    def write_fixture(self, run):
        base = {"seed": 42, "run_id": "ft011_densenet201_differential_lr_s42", "minimum_delta": 0.0001, "epochs": 12}
        protocol = {"control_configuration": base, "protocol_hash": "fixed-protocol",
                    "scheduler_defaults": {"mode": "max", "min_lr": [0.0, 0.0]}}
        wrapper = {"base_training_configuration": base, "learning_rate_policy": {"backbone": 1e-5, "classifier": 1e-4}, "protocol_hash": "fixed-protocol"}
        wrapper["candidate_configuration_hash"] = canonical_hash(wrapper)
        status = {"status": "completed", "protocol_hash": "fixed-protocol", "smoke_execution_only": False, "best_epoch": 2}
        groups = {"disjoint": True, "exhaustive": True, "backbone_parameter_names": ["features.weight"],
                  "classifier_parameter_names": ["classifier.weight"], "backbone_parameters": 4,
                  "classifier_parameters": 2, "initial_learning_rates": [1e-5, 1e-4]}
        initialization = {"fresh_imagenet_initialization": True, "seed": 42,
                          "model_state_sha256": "a" * 64, "classifier_state_sha256": "b" * 64}
        for name, value in (("config.json", base), ("candidate_configuration.json", wrapper),
                            ("status.json", status), ("optimizer_groups.json", groups),
                            ("initialization.json", initialization),
                            ("scheduler_initial.json", {"mode": "max", "min_lrs": [0.0, 0.0]})):
            (run / name).write_text(json.dumps(value))
        metrics = {"protocol_hash": "fixed-protocol", "smoke_execution_only": False,
                   "trainable_parameter_count": 6, "best_epoch": 2, "macro_f1": 0.86}
        history = [{"epoch": str(i + 1), "val_macro_f1": str(score),
                    "backbone_lr_used": "0.00001", "classifier_lr_used": "0.0001"}
                   for i, score in enumerate((0.85, 0.86, 0.86005))]
        return protocol, metrics, history

    def test_within_run_minimum_delta_keeps_earlier_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            protocol, metrics, history = self.write_fixture(run)
            verify_candidate_configuration(run, 42, protocol, metrics, history)
            with self.assertRaisesRegex(RuntimeError, "selected checkpoint"):
                verify_candidate_configuration(run, 42, protocol, {**metrics, "best_epoch": 3}, history)

    def test_optimizer_name_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            protocol, metrics, history = self.write_fixture(run)
            groups = json.loads((run / "optimizer_groups.json").read_text())
            groups["classifier_parameter_names"] = groups["backbone_parameter_names"]
            (run / "optimizer_groups.json").write_text(json.dumps(groups))
            with self.assertRaisesRegex(RuntimeError, "disjoint and exhaustive"):
                verify_candidate_configuration(run, 42, protocol, metrics, history)

    def test_changed_differential_policy_cannot_be_hidden_by_old_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            protocol, metrics, history = self.write_fixture(run)
            wrapper = json.loads((run / "candidate_configuration.json").read_text())
            wrapper["learning_rate_policy"]["backbone"] = 0.0001
            (run / "candidate_configuration.json").write_text(json.dumps(wrapper))
            with self.assertRaisesRegex(RuntimeError, "configuration/protocol hash"):
                verify_candidate_configuration(run, 42, protocol, metrics, history)

class PairedBootstrapTests(unittest.TestCase):
    def test_whole_proxy_clusters_remain_intact_with_class_stratification(self):
        manifest = fixture_manifest()
        strata, description = cluster_strata(manifest)
        indices = draw_cluster_indices(strata, np.random.default_rng(42))
        self.assertEqual(description["group_count"], 3)
        self.assertEqual(description["repeated_image_group_count"], 3)
        for group in ((0, 1), (2, 3), (4, 5)):
            self.assertEqual(sum(indices == group[0]), sum(indices == group[1]))
        self.assertIn("not verified patient", description["limitation"])

    def test_missing_group_cannot_silently_switch_to_image_bootstrap(self):
        manifest = fixture_manifest()
        manifest[0]["split_group_id"] = ""
        with self.assertRaisesRegex(RuntimeError, "proxy split_group_id"):
            cluster_strata(manifest)

    def test_identical_systems_have_exact_zero_paired_interval_for_every_seed(self):
        manifest = fixture_manifest()
        labels = np.array([0, 0, 1, 1, 2, 2])
        rng = np.random.default_rng(123)
        runs = {seed: rng.dirichlet(np.ones(3), size=6) for seed in (42, 123, 2026)}
        result = paired_bootstrap(labels, {name: runs for name in ("FT011", "FT010", "FT004")}, manifest, (42, 123, 2026), iterations=50)
        for comparison in result["comparisons"]:
            for metric in comparison["metrics"].values():
                self.assertEqual(metric["point_difference"], 0)
                self.assertEqual(metric["ci_95_percentile"], [0, 0])

    def test_comparison_averages_seed_metrics_not_seed_probability_vectors(self):
        manifest = fixture_manifest()
        labels = np.array([0, 0, 1, 1, 2, 2])
        candidate = {42: np.eye(3)[labels], 123: np.eye(3)[(labels + 1) % 3]}
        comparator = {42: np.eye(3)[(labels + 2) % 3], 123: np.eye(3)[(labels + 2) % 3]}
        result = paired_bootstrap(labels, {"FT011": candidate, "FT010": comparator, "FT004": comparator}, manifest, (42, 123), iterations=10)
        self.assertEqual(result["comparisons"][0]["metrics"]["macro_f1"]["point_difference"], 0.5)
        self.assertIn("no cross-seed probability averaging", result["seed_handling"])

    def test_fast_bootstrap_metrics_match_original_evaluator(self):
        labels = np.tile(np.arange(3), 25)
        probabilities = np.random.default_rng(2026).dirichlet(np.ones(3), size=len(labels))
        original = calculate_metrics(labels, probabilities, CLASSES)
        actual = fast_metrics(labels, probabilities)
        for name in ("accuracy", "balanced_accuracy", "macro_f1"):
            self.assertAlmostEqual(actual[name], original[name], places=14)
        for name in CLASSES:
            self.assertAlmostEqual(actual[f"{name.lower()}_recall"], original["per_class"][name]["recall"], places=14)

    def test_single_seed_has_no_sample_standard_deviation(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        metrics = calculate_metrics(labels, np.eye(3)[labels], CLASSES)
        self.assertIsNone(aggregate_metrics({42: metrics})["macro_f1"]["sample_std"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
