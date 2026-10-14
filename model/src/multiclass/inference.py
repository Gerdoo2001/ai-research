"""Checkpoint loading and stable class-probability inference contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from PIL import Image

from .config import ExperimentConfig
from .models import build_model
from .preprocessing import build_transform


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_member_probabilities(
    member_probabilities: Iterable[dict[str, float]],
    class_names: tuple[str, ...] = ("Normal", "Viral", "Bacterial"),
) -> dict[str, float]:
    """Validate and equally average three member probability dictionaries."""

    import numpy as np

    members = list(member_probabilities)
    if len(members) != 3:
        raise ValueError("FT-004 requires exactly three member probability vectors")
    matrix = np.asarray(
        [[float(member[name]) for name in class_names] for member in members],
        dtype=np.float64,
    )
    if matrix.shape != (3, len(class_names)) or not np.isfinite(matrix).all():
        raise ValueError("ensemble probabilities must be finite three-class vectors")
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError("ensemble probabilities must remain within [0, 1]")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("each member probability vector must sum to one")
    mean = matrix.mean(axis=0)
    return {name: float(value) for name, value in zip(class_names, mean)}


class MulticlassPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        *,
        configuration_hash_paths: dict[str, str] | None = None,
    ) -> None:
        import torch

        self.checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.config = ExperimentConfig(**checkpoint["config"])
        hash_payload = self.config.hash_payload()
        if configuration_hash_paths is not None:
            if set(configuration_hash_paths) != {"manifest_dir", "results_dir"} or any(
                not isinstance(value, str) or not value for value in configuration_hash_paths.values()
            ):
                raise ValueError("portable configuration hash paths require the two original path strings")
            # Historical configuration hashes include absolute research paths.
            # Preserve their original strings solely for integrity verification;
            # inference does not access either manifest or training-results path.
            hash_payload.update(configuration_hash_paths)
        encoded = json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.configuration_hash = hashlib.sha256(encoded).hexdigest()
        if checkpoint.get("config_hash") != self.configuration_hash:
            raise RuntimeError("checkpoint configuration hash does not match its embedded configuration")
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        # Every trained parameter and buffer is supplied below. Constructing the
        # architecture without ImageNet weights avoids an unnecessary download.
        architecture_config = replace(self.config, weights="none")
        self.model, self.model_metadata = build_model(architecture_config)
        self.model_metadata = replace(
            self.model_metadata, weight_source=f"trained_checkpoint::{self.checkpoint_path.name}"
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()
        self.transform = build_transform(self.config, training=False)

    def predict(self, image: Image.Image) -> dict:
        import torch

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0].cpu().tolist()
        predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        mapping = {index: name for index, name in enumerate(self.config.resolved_class_names)}
        return {
            "predicted_index": predicted_index,
            "predicted_class": mapping[predicted_index],
            "class_probabilities": {mapping[index]: float(value) for index, value in enumerate(probabilities)},
            "confidence": float(probabilities[predicted_index]),
            "class_mapping": mapping,
            "task_mode": self.config.task_mode,
            "configuration_hash": self.configuration_hash,
        }


class FT004EnsemblePredictor:
    """Frozen three-member FT-004 research inference package.

    The default selection loads the original seed-42 ensemble definition. The
    three-seed experiment is evidence about repeatability; inference still uses
    exactly three networks, not nine.
    """

    def __init__(
        self,
        selection_path: str | Path,
        device: str = "cuda",
    ) -> None:
        self.selection_path = Path(selection_path).resolve()
        selection = json.loads(self.selection_path.read_text(encoding="utf-8"))
        if selection.get("status") != "frozen_validation_selected_research_model":
            raise RuntimeError("FT-004 selection is not frozen")
        if selection.get("deployable_member_count") != 3:
            raise RuntimeError("FT-004 must contain exactly three deployable members")
        if selection.get("class_order") != ["Normal", "Viral", "Bacterial"]:
            raise RuntimeError("FT-004 class order has drifted")
        inference_rule = selection.get("inference_rule", {})
        if inference_rule.get("method") != "equal arithmetic mean of three class-probability vectors":
            raise RuntimeError("FT-004 averaging rule has drifted")
        if inference_rule.get("learned_ensemble_weights") is not False:
            raise RuntimeError("FT-004 does not permit learned ensemble weights")

        self.selection = selection
        self.class_names = tuple(selection["class_order"])
        self.members: list[tuple[dict, MulticlassPredictor]] = []
        for member in sorted(selection["members"], key=lambda item: item["member_order"]):
            checkpoint_path = Path(member["checkpoint_path"])
            if not checkpoint_path.is_absolute():
                checkpoint_path = self.selection_path.parent / checkpoint_path
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"missing frozen FT-004 checkpoint: {checkpoint_path}")
            if _sha256(checkpoint_path) != member["checkpoint_sha256"]:
                raise RuntimeError(f"FT-004 checkpoint hash mismatch: {member['run_id']}")
            predictor = MulticlassPredictor(
                checkpoint_path, device=device,
                configuration_hash_paths=member.get("configuration_hash_paths"),
            )
            if tuple(predictor.config.resolved_class_names) != self.class_names:
                raise RuntimeError(f"FT-004 member class order mismatch: {member['run_id']}")
            if predictor.configuration_hash != member["configuration_hash_recorded"]:
                raise RuntimeError(f"FT-004 configuration hash mismatch: {member['run_id']}")
            self.members.append((member, predictor))

    def predict(self, image: Image.Image) -> dict:
        outputs = []
        for member, predictor in self.members:
            output = predictor.predict(image)
            outputs.append(
                {
                    "run_id": member["run_id"],
                    "preprocessing": member["preprocessing"],
                    "predicted_class": output["predicted_class"],
                    "confidence": output["confidence"],
                    "class_probabilities": output["class_probabilities"],
                }
            )
        probabilities = average_member_probabilities(
            [output["class_probabilities"] for output in outputs], self.class_names
        )
        predicted_class = max(probabilities, key=probabilities.__getitem__)
        predicted_index = self.class_names.index(predicted_class)
        return {
            "selection_id": self.selection["selection_id"],
            "task_mode": "multiclass",
            "predicted_index": predicted_index,
            "predicted_class": predicted_class,
            "confidence": probabilities[predicted_class],
            "class_probabilities": probabilities,
            "ensemble_method": "equal arithmetic mean",
            "member_count": 3,
            "member_outputs": outputs,
            "research_use_only": True,
        }
