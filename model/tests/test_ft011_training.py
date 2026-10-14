"""Synthetic CPU checks for FT-011 optimizer and exact interruption recovery."""

from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model/src"))

from multiclass.config import ExperimentConfig  # noqa: E402
from multiclass.ft011_training import (  # noqa: E402
    ProtectiveStop,
    ReplayBatchSampler,
    differential_optimizer,
    run_ft011,
)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 2, 1), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(2, 3)

    def forward(self, inputs):
        return self.classifier(self.features(inputs).flatten(1))


class SyntheticDataset(Dataset):
    """Augmentation draws from all three historical host RNG sources."""

    def __init__(self, count, split, records):
        self.count, self.split, self.records = count, split, records
        self.rows = [{"assigned_split": split, "class_index": str(i % 3)} for i in range(count)]

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        draws = (random.random(), float(np.random.rand()), float(torch.rand(()))) if self.split == "train" else (0., 0., 0.)
        self.records.append((self.split, index, *draws))
        image = torch.full((3, 64, 64), index / 24 + sum(draws), dtype=torch.float32)
        return image, index % 3, f"synthetic/{self.split}/{index}"


class WatchdogStub:
    def __init__(self, stop_at=None):
        self.stop_at, self.calls = stop_at, 0
        self.status = {"synthetic_cpu_test": True}

    def poll(self):
        self.calls += 1
        return {"action": "stop" if self.calls == self.stop_at else "continue", "reason": "synthetic interruption"}

    def update_progress(self, **kwargs):
        pass


class DifferentialOptimizerTests(unittest.TestCase):
    def test_groups_are_disjoint_exhaustive_with_historical_adamw_defaults(self):
        model = TinyClassifier()
        optimizer, audit = differential_optimizer(model)
        groups = optimizer.param_groups
        self.assertEqual([g["name"] for g in groups], ["backbone", "classifier"])
        self.assertEqual([g["lr"] for g in groups], [1e-5, 1e-4])
        self.assertEqual([g["weight_decay"] for g in groups], [1e-4, 1e-4])
        self.assertEqual([g["betas"] for g in groups], [(0.9, 0.999)] * 2)
        self.assertEqual([g["eps"] for g in groups], [1e-8, 1e-8])
        backbone, classifier = [{id(p) for p in g["params"]} for g in groups]
        self.assertFalse(backbone & classifier)
        self.assertEqual(backbone | classifier, {id(p) for p in model.parameters()})
        self.assertEqual(classifier, {id(p) for p in model.classifier.parameters()})
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertTrue(audit["disjoint"] and audit["exhaustive"])
        self.assertEqual(audit["classifier_parameters"], sum(p.numel() for p in model.classifier.parameters()))

    def test_frozen_backbone_is_rejected(self):
        model = TinyClassifier()
        next(model.features.parameters()).requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError, "fully trainable"):
            differential_optimizer(model)

    def test_missing_classifier_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "classifier"):
            differential_optimizer(nn.Linear(2, 3))


class ReplaySamplerTests(unittest.TestCase):
    def _run_orders(self, replay):
        generator = torch.Generator().manual_seed(42)
        train = DataLoader(TensorDataset(torch.arange(25)), batch_size=8, shuffle=True, num_workers=0, generator=generator)
        val = DataLoader(TensorDataset(torch.arange(17)), batch_size=8, shuffle=False, num_workers=0, generator=generator)
        if replay:
            for loader in (train, val):
                object.__setattr__(loader, "batch_sampler", ReplayBatchSampler(loader.batch_sampler))
        orders, states = [], []
        for _ in range(2):
            epoch = []
            for loader in (train, val):
                if replay:
                    loader.batch_sampler.reset()
                epoch.append([batch[0].tolist() for batch in loader])
                states.append(generator.get_state().clone())
            orders.append(epoch)
        return orders, states

    def test_two_epochs_match_original_shared_generator_and_sample_order(self):
        expected_orders, expected_states = self._run_orders(False)
        orders, states = self._run_orders(True)
        self.assertEqual(orders, expected_orders)
        for actual, expected in zip(states, expected_states):
            self.assertTrue(torch.equal(actual, expected))

    def test_invalid_restored_permutation_is_rejected(self):
        loader = DataLoader(TensorDataset(torch.arange(10)), batch_size=8, shuffle=True)
        sampler = ReplayBatchSampler(loader.batch_sampler)
        with self.assertRaisesRegex(RuntimeError, "complete permutation"):
            sampler.load_state_dict({"order": [0] * 10, "next_batch": 1})


class RecoveryIntegrationTests(unittest.TestCase):
    def _execute(self, directory, records, *, stop_at=None, resume=False):
        config = ExperimentConfig(architecture="densenet201", batch_size=8, input_size=64,
                                  epochs=2, num_workers=0, amp=False, label_smoothing=0.05,
                                  tensorboard=False, run_id="synthetic_ft011")

        def loaders(_):
            generator = torch.Generator().manual_seed(config.seed)
            train = DataLoader(SyntheticDataset(24, "train", records), batch_size=8,
                               shuffle=True, num_workers=0, generator=generator)
            val = DataLoader(SyntheticDataset(18, "val", records), batch_size=8,
                             shuffle=False, num_workers=0, generator=generator)
            return train, val, {"manifest_version_hash": "synthetic-manifest"}

        def model_factory(_):
            model = TinyClassifier()
            count = sum(p.numel() for p in model.parameters())
            return model, SimpleNamespace(architecture="synthetic", parameter_count=count,
                                          trainable_parameter_count=count, num_classes=3,
                                          weight_source="synthetic_cpu_test")

        def save_metrics(run_dir, metrics, *args, **kwargs):
            (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

        with ExitStack() as stack:
            for target, replacement in {
                "multiclass.ft011_training.build_dataloaders": loaders,
                "multiclass.ft011_training.build_model": model_factory,
                "multiclass.ft011_training.resolve_device": lambda _: torch.device("cpu"),
                "multiclass.ft011_training.save_evaluation_artifacts": save_metrics,
                "multiclass.ft011_training.plot_training_history": lambda *args: None,
                "torch.cuda.reset_peak_memory_stats": lambda *args: None,
                "torch.cuda.max_memory_allocated": lambda *args: 0,
                "torch.cuda.synchronize": lambda *args: None,
                "torch.cuda.get_rng_state_all": lambda: [],
                "torch.cuda.set_rng_state_all": lambda *args: None,
                "torch.cuda.is_available": lambda: False,
            }.items():
                stack.enter_context(patch(target, replacement))
            return run_ft011(config, directory, WatchdogStub(stop_at), protocol_hash="synthetic-protocol",
                             resume=resume, smoke=True)

    def assert_nested_equal(self, actual, expected):
        if isinstance(actual, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(actual, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(actual, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_pending_augmented_batch_and_every_training_component_survive_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            full_records, resumed_records = [], []
            self._execute(root / "full", full_records)
            with self.assertRaises(ProtectiveStop):
                self._execute(root / "resumed", resumed_records, stop_at=4)
            pending = torch.load(root / "resumed/last_checkpoint.pt", map_location="cpu", weights_only=False)
            required = {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict",
                        "rng_state", "state", "train_sampler", "val_sampler", "protocol_hash", "manifest_hash", "config_hash"}
            self.assertTrue(required <= pending.keys())
            self.assertEqual(pending["train_sampler"]["next_batch"], 1)
            self.assertEqual(len(resumed_records), 16)
            self.assertIsNotNone(pending["state"]["pending_batch"])
            inputs, targets, paths = pending["state"]["pending_batch"]
            self.assertEqual(list(paths), [f"synthetic/train/{r[1]}" for r in full_records[8:16]])
            for image, record in zip(inputs, full_records[8:16]):
                self.assertTrue(torch.equal(image, torch.full_like(image, record[1] / 24 + sum(record[2:]))))
            self.assertEqual(targets.tolist(), [r[1] % 3 for r in full_records[8:16]])
            rng = random.Random()
            rng.setstate(pending["rng_state"]["python"])
            self.assertEqual(rng.random(), full_records[16][2])
            self._execute(root / "resumed", resumed_records, resume=True)
            self.assertEqual(resumed_records, full_records)
            final = torch.load(root / "resumed/last_checkpoint.pt", map_location="cpu", weights_only=False)
            expected = torch.load(root / "full/last_checkpoint.pt", map_location="cpu", weights_only=False)
            for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict",
                        "rng_state", "train_sampler", "val_sampler"):
                with self.subTest(component=key):
                    self.assert_nested_equal(final[key], expected[key])
            self.assertEqual(final["state"]["best_epoch"], expected["state"]["best_epoch"])
            for left, right in zip(final["state"]["history"], expected["state"]["history"]):
                for key in ("train_loss", "val_loss", "val_macro_f1", "backbone_lr_used", "classifier_lr_used"):
                    self.assertEqual(left[key], right[key])


if __name__ == "__main__":
    unittest.main()
