"""CPU-safe contract tests for the parallel classification pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model/src"))
sys.path.insert(0, str(REPO_ROOT / "model/scripts"))

from multiclass.config import ExperimentConfig, load_config  # noqa: E402
from multiclass.data import (  # noqa: E402
    apply_normalized_roi_crop,
    build_evaluation_loader,
    load_roi_bbox_records,
    read_manifest,
    validate_manifest_bundle,
)
from multiclass.metrics import calculate_metrics  # noqa: E402
from multiclass.models import UnsupportedArchitectureError, assert_output_contract, build_model  # noqa: E402
from multiclass.preprocessing import (  # noqa: E402
    PREPROCESSING_PIPELINES,
    ImageTransform,
    preprocessing_metadata,
)
from multiclass.trainer import FocalLoss  # noqa: E402
from generate_gradcam_review import target_layer  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_locked_class_order_and_stable_hash(self):
        config = load_config(REPO_ROOT / "model/configs/multiclass_baseline.json")
        self.assertEqual(config.resolved_class_names, ("Normal", "Viral", "Bacterial"))
        self.assertEqual(config.num_classes, 3)
        self.assertEqual(config.config_hash, config.config_hash)

    def test_invalid_class_order_is_rejected(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(class_names=["Bacterial", "Viral", "Normal"])

    def test_spatial_augmentation_validation(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(augmentation={"translation_fraction": 0.2})
        with self.assertRaises(ValueError):
            ExperimentConfig(augmentation={"scale_min": 1.1, "scale_max": 0.9})

    def test_disabled_roi_preserves_historical_winner_hash(self):
        config = load_config(
            REPO_ROOT
            / "model/results/runs/staged_densenet201_pseudo_color_then_histogram_equalization_then_clahe_s42/config.json"
        )
        self.assertEqual(
            config.config_hash,
            "ef029d89d066687af73d2973be80caf5b16bb3d648000f617b77a73306ab2691",
        )

    def test_roi_configuration_is_fail_closed(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(roi={"mode": "pspnet_bbox"})
        with self.assertRaises(ValueError):
            ExperimentConfig(roi={"mode": "none", "threshold": 0.2})

    def test_focal_gamma_is_explicit_and_historical_default_is_preserved(self):
        implicit = ExperimentConfig(loss="focal")
        explicit = ExperimentConfig(loss="focal", focal_gamma=2.0)
        self.assertEqual(implicit.effective_focal_gamma, 2.0)
        self.assertEqual(explicit.effective_focal_gamma, 2.0)
        self.assertNotEqual(implicit.config_hash, explicit.config_hash)
        with self.assertRaises(ValueError):
            ExperimentConfig(loss="focal", focal_gamma=0.0)
        with self.assertRaises(ValueError):
            ExperimentConfig(loss="cross_entropy", focal_gamma=2.0)

    def test_label_smoothing_range_is_fail_closed(self):
        self.assertEqual(ExperimentConfig(label_smoothing=0.05).label_smoothing, 0.05)
        with self.assertRaises(ValueError):
            ExperimentConfig(label_smoothing=-0.1)
        with self.assertRaises(ValueError):
            ExperimentConfig(label_smoothing=1.0)


class ManifestTests(unittest.TestCase):
    def test_locked_bundle_and_split_counts(self):
        manifest_dir = REPO_ROOT / "model/data_manifests"
        summary = validate_manifest_bundle(manifest_dir)
        self.assertEqual(summary["manifest_version"], "three_class_seed42_3b815cf769c8")
        self.assertEqual(len(read_manifest(manifest_dir / "train_manifest.csv")), 4076)
        self.assertEqual(len(read_manifest(manifest_dir / "val_manifest.csv")), 874)
        self.assertEqual(len(read_manifest(manifest_dir / "test_manifest.csv")), 874)

    def test_locked_test_requires_explicit_unlock(self):
        config = load_config(REPO_ROOT / "model/configs/multiclass_baseline.json", {"device": "cpu"})
        with self.assertRaises(PermissionError):
            build_evaluation_loader(config, "test")


class TransformTests(unittest.TestCase):
    def test_normalized_roi_crop_reproduces_center_square_geometry(self):
        image = Image.fromarray(np.zeros((80, 120, 3), dtype=np.uint8))
        cropped = apply_normalized_roi_crop(image, [0.25, 0.125, 0.75, 0.875])
        self.assertEqual(cropped.size, (40, 60))

    def test_roi_manifest_requires_complete_matching_records(self):
        expected_row = {
            "relative_path": "train/NORMAL/example.jpeg",
            "assigned_split": "train",
            "cryptographic_hash": "abc123",
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "roi.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "locked_test_accessed": False,
                        "policy": {
                            "mode": "pspnet_bbox",
                            "segmentation_model": "torchxrayvision_chestx_det_pspnet",
                            "threshold": 0.2,
                            "margin_fraction": 0.18,
                        },
                        "records": [
                            {
                                **expected_row,
                                "bbox_normalized": [0.1, 0.1, 0.9, 0.9],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = ExperimentConfig(
                roi={
                    "mode": "pspnet_bbox",
                    "bbox_manifest": str(manifest_path),
                    "segmentation_model": "torchxrayvision_chestx_det_pspnet",
                    "threshold": 0.2,
                    "margin_fraction": 0.18,
                }
            )
            records = load_roi_bbox_records(config, [expected_row], split="train")
            self.assertEqual(records[expected_row["relative_path"]]["bbox_normalized"], [0.1, 0.1, 0.9, 0.9])
            with self.assertRaises(RuntimeError):
                load_roi_bbox_records(config, [{**expected_row, "cryptographic_hash": "changed"}], split="train")

    def test_all_supported_transforms_return_three_normalized_channels(self):
        image = Image.fromarray(np.tile(np.arange(96, dtype=np.uint8), (64, 1)))
        self.assertEqual(len(PREPROCESSING_PIPELINES), 16)
        for name in PREPROCESSING_PIPELINES:
            with self.subTest(name=name):
                transform = ImageTransform(name, 64, training=False)
                output = transform(image)
                self.assertEqual(tuple(output.shape), (3, 64, 64))
                self.assertTrue(bool(output.isfinite().all()))
                self.assertEqual(preprocessing_metadata(name).name, name)

    def test_combined_contrast_metadata_preserves_operation_order(self):
        he_then_clahe = preprocessing_metadata("histogram_equalization_then_clahe")
        clahe_then_he = preprocessing_metadata("clahe_then_histogram_equalization")
        self.assertTrue(he_then_clahe.parameters["stages"][0].startswith("global"))
        self.assertTrue(clahe_then_he.parameters["stages"][0].startswith("OpenCV CLAHE"))

    def test_all_ordered_subsets_are_declared_exactly_once(self):
        sequences = list(PREPROCESSING_PIPELINES.values())
        self.assertEqual(len(sequences), len(set(sequences)))
        self.assertEqual({len(sequence) for sequence in sequences}, {0, 1, 2, 3})
        self.assertEqual(sum(len(sequence) == 2 for sequence in sequences), 6)
        self.assertEqual(sum(len(sequence) == 3 for sequence in sequences), 6)
        for sequence in sequences:
            self.assertEqual(len(sequence), len(set(sequence)))

    def test_order_changes_the_pre_normalization_pixels(self):
        x = np.linspace(0, 255, 96, dtype=np.uint8)
        gradient = np.tile(x, (64, 1))
        image = Image.fromarray(np.stack((gradient, np.flipud(gradient), gradient), axis=2))
        first = np.asarray(
            ImageTransform("clahe_then_pseudo_color", 64, training=False)._enhance(image)
        )
        second = np.asarray(
            ImageTransform("pseudo_color_then_clahe", 64, training=False)._enhance(image)
        )
        self.assertFalse(np.array_equal(first, second))
        self.assertGreater(np.max(np.abs(first.astype(np.int16) - second.astype(np.int16))), 0)

    def test_optional_spatial_jitter_preserves_tensor_contract(self):
        image = Image.fromarray(np.tile(np.arange(96, dtype=np.uint8), (64, 1)))
        transform = ImageTransform(
            "pseudo_color_then_histogram_equalization_then_clahe",
            64,
            training=True,
            translation_fraction=0.03,
            scale_min=0.95,
            scale_max=1.05,
        )
        output = transform(image)
        self.assertEqual(tuple(output.shape), (3, 64, 64))
        self.assertTrue(bool(output.isfinite().all()))


class ModelTests(unittest.TestCase):
    def test_all_supported_heads_emit_three_logits(self):
        import torch

        for architecture in (
            "densenet121", "densenet169", "densenet201",
            "densenet201_sas_mff",
            "resnet50", "resnet101", "resnet152",
            "yolov5_cls", "yolov8_cls",
        ):
            with self.subTest(architecture=architecture):
                config = load_config(
                    REPO_ROOT / "model/configs/multiclass_baseline.json",
                    {"architecture": architecture, "weights": "none", "input_size": 64, "device": "cpu"},
                )
                model, metadata = build_model(config)
                self.assertEqual(metadata.num_classes, 3)
                self.assertEqual(assert_output_contract(model, 64, 3, torch.device("cpu")), (2, 3))
                self.assertIsNotNone(target_layer(model, architecture))
                if architecture == "densenet201_sas_mff":
                    self.assertLess(metadata.parameter_count, 20_000_000)

    def test_yolov10_detection_config_is_not_accepted_as_classification(self):
        config = load_config(
            REPO_ROOT / "model/configs/multiclass_baseline.json",
            {"architecture": "yolov10_cls", "weights": "none", "input_size": 64, "device": "cpu"},
        )
        with self.assertRaises(UnsupportedArchitectureError):
            build_model(config)


class MetricTests(unittest.TestCase):
    def test_perfect_three_class_metrics(self):
        labels = np.asarray([0, 1, 2, 0, 1, 2])
        probabilities = np.eye(3)[labels] * 0.98 + 0.02 / 3
        metrics = calculate_metrics(labels, probabilities, ("Normal", "Viral", "Bacterial"))
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0, 0], [0, 2, 0], [0, 0, 2]])

    def test_focal_probability_is_not_distorted_by_class_weight(self):
        import torch

        logits = torch.tensor([[3.0, 0.5, -1.0], [0.1, 1.0, 0.2]], dtype=torch.float32)
        targets = torch.tensor([0, 1])
        unweighted = FocalLoss(gamma=2.0)(logits, targets)
        unit_weighted = FocalLoss(weight=torch.ones(3), gamma=2.0)(logits, targets)
        doubled = FocalLoss(weight=torch.full((3,), 2.0), gamma=2.0)(logits, targets)
        self.assertTrue(torch.allclose(unweighted, unit_weighted))
        self.assertTrue(torch.allclose(doubled, unweighted * 2.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
