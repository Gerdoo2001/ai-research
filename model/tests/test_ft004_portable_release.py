"""Portable path/hash-context and offline checkpoint-loading contracts."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model/src"))
sys.path.insert(0, str(REPO_ROOT / "model/scripts"))

import multiclass.config as config_module
from multiclass.config import ExperimentConfig
from multiclass.inference import FT004EnsemblePredictor, MulticlassPredictor
from multiclass.models import ModelMetadata
from package_ft004_release import package_release, sha256, verify_package


def frozen_fixture(root):
    members = []
    for index in range(1, 4):
        configuration = root / f"source_config_{index}.json"
        config = ExperimentConfig(architecture="densenet201", seed=42)
        configuration.write_text(json.dumps(config.as_dict()))
        checkpoint = root / f"source_checkpoint_{index}.pt"
        checkpoint.write_bytes(f"fixture checkpoint {index}".encode())
        members.append({"member_order": index, "run_id": f"fixture_run_{index}",
                        "seed": 42, "architecture": "densenet201",
                        "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                        "checkpoint_size_bytes": checkpoint.stat().st_size,
                        "configuration_path": str(configuration), "configuration_sha256": sha256(configuration),
                        "configuration_hash_recorded": config.config_hash})
    source = root / "original_frozen_selection.json"
    source.write_text(json.dumps({"selection_id": "fixture-ft004", "status": "frozen_validation_selected_research_model",
        "deployable_member_count": 3, "deployable_seed": 42, "class_order": ["Normal", "Viral", "Bacterial"],
        "inference_rule": {"method": "equal arithmetic mean of three class-probability vectors", "learned_ensemble_weights": False},
        "members": members}))
    return source


class PackageIntegrityTests(unittest.TestCase):
    def test_package_preserves_source_and_checkpoint_bytes_with_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = frozen_fixture(root)
            original_hash = sha256(source)
            output = root / "publication"
            result = package_release(source, output)
            self.assertEqual(result["status"], "verified")
            self.assertEqual(sha256(source), original_hash)
            portable = json.loads((output / "selection.json").read_text())
            original = json.loads(source.read_text())
            for member, origin in zip(portable["members"], original["members"]):
                self.assertFalse(Path(member["checkpoint_path"]).is_absolute())
                self.assertEqual((output / member["checkpoint_path"]).read_bytes(), Path(origin["checkpoint_path"]).read_bytes())
                self.assertEqual(set(member["configuration_hash_paths"]), {"manifest_dir", "results_dir"})
            self.assertEqual(package_release(source, output)["selection_sha256"], result["selection_sha256"])

    def test_portable_package_verifies_after_repository_hash_context_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "publication"
            package_release(frozen_fixture(root), output)
            with patch.object(config_module, "REPO_ROOT", root / "a_different_clone"):
                self.assertEqual(verify_package(output)["status"], "verified")

    def test_changed_publication_checkpoint_is_detected_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = frozen_fixture(root), root / "publication"
            package_release(source, output)
            checkpoint = output / "member_1_viridis_ge_clahe_s42.pt"
            checkpoint.write_bytes(b"changed publication")
            with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                package_release(source, output)
            self.assertEqual(checkpoint.read_bytes(), b"changed publication")


class OfflineInferenceTests(unittest.TestCase):
    def test_saved_imagenet_config_is_retained_but_architecture_does_not_download_weights(self):
        config = ExperimentConfig(architecture="densenet201", weights="imagenet")
        saved = {"config": config.as_dict(), "config_hash": config.config_hash, "model_state_dict": {}}
        unchanged = copy.deepcopy(saved)
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        metadata = ModelMetadata("densenet201", "random_initialization", 3, 0, 0)
        with patch("torch.load", return_value=saved), patch("multiclass.inference.build_model", return_value=(model, metadata)) as factory:
            predictor = MulticlassPredictor("fixture.pt", device="cpu")
        self.assertEqual(factory.call_args.args[0].weights, "none")
        self.assertEqual(predictor.config.weights, "imagenet")
        self.assertEqual(predictor.configuration_hash, config.config_hash)
        self.assertEqual(saved, unchanged)

    def test_original_config_hash_can_be_verified_in_a_new_clone(self):
        config = ExperimentConfig(architecture="densenet201")
        payload = config.hash_payload()
        paths = {name: payload[name] for name in ("manifest_dir", "results_dir")}
        saved = {"config": config.as_dict(), "config_hash": config.config_hash, "model_state_dict": {}}
        metadata = ModelMetadata("densenet201", "random_initialization", 3, 0, 0)
        with tempfile.TemporaryDirectory() as directory, patch.object(config_module, "REPO_ROOT", Path(directory)):
            self.assertNotEqual(ExperimentConfig(**saved["config"]).config_hash, saved["config_hash"])
            with patch("torch.load", return_value=saved), patch("multiclass.inference.build_model", return_value=(Mock(), metadata)):
                predictor = MulticlassPredictor("fixture.pt", device="cpu", configuration_hash_paths=paths)
            self.assertEqual(predictor.configuration_hash, saved["config_hash"])

    def test_ensemble_resolves_members_relative_to_selection_and_keeps_absolute_legacy_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = frozen_fixture(root)
            record = json.loads(selection.read_text())
            expected = []
            for index, member in enumerate(record["members"]):
                checkpoint = Path(member["checkpoint_path"])
                expected.append(checkpoint)
                if index < 2:
                    member["checkpoint_path"] = checkpoint.name
            selection.write_text(json.dumps(record))
            predictors = []
            for member in record["members"]:
                predictor = Mock()
                predictor.config.resolved_class_names = ("Normal", "Viral", "Bacterial")
                predictor.configuration_hash = member["configuration_hash_recorded"]
                predictors.append(predictor)
            with patch("multiclass.inference.MulticlassPredictor", side_effect=predictors) as factory:
                ensemble = FT004EnsemblePredictor(selection, device="cpu")
            self.assertEqual(len(ensemble.members), 3)
            self.assertEqual([call.args[0].resolve() for call in factory.call_args_list], [path.resolve() for path in expected])


if __name__ == "__main__":
    unittest.main(verbosity=2)
