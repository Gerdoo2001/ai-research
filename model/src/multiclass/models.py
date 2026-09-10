"""Model factory for supported three-class image classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import REPO_ROOT


TORCHVISION_ARCHITECTURES = {
    "densenet121",
    "densenet169",
    "densenet201",
    "resnet50",
    "resnet101",
    "resnet152",
}
EXPERIMENTAL_ARCHITECTURES = {"densenet201_sas_mff"}
YOLO_ARCHITECTURES = {"yolov5_cls", "yolov8_cls"}
SUPPORTED_ARCHITECTURES = TORCHVISION_ARCHITECTURES | EXPERIMENTAL_ARCHITECTURES | YOLO_ARCHITECTURES


class UnsupportedArchitectureError(ValueError):
    pass


class ClassificationLogitsAdapter:
    """Normalize third-party classifier eval outputs to raw logits.

    Ultralytics classification models return raw logits while training and a
    `(probabilities, logits)` tuple while evaluating. The common trainer and
    loss contract always consume logits.
    """

    def __new__(cls, model):
        import torch.nn as nn

        class _Adapter(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, inputs):
                output = self.wrapped(inputs)
                if isinstance(output, tuple):
                    if len(output) < 2:
                        raise RuntimeError("classification tuple does not contain logits")
                    return output[-1]
                return output

        return _Adapter(model)


@dataclass(frozen=True)
class ModelMetadata:
    architecture: str
    weight_source: str
    num_classes: int
    parameter_count: int
    trainable_parameter_count: int


class _ConvBNLeaky(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        groups: int = 1,
    ):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )


class SpatialAttentionSuperposition(nn.Module):
    """Lightweight classification adaptation of Li et al.'s SAS module.

    The source paper leaves several layer-level implementation details implicit.
    This adaptation preserves its documented contract: initial mean/max spatial
    attention, split feature branches, a wider-receptive-field encoder/decoder
    branch, and superposition of the two attended feature groups. A learnable
    residual scale starts small so ImageNet features are not destroyed at the
    beginning of fine-tuning.
    """

    def __init__(self, channels: int):
        super().__init__()
        if channels % 2:
            raise ValueError("SAS requires an even channel count")
        half = channels // 2
        self.initial_attention = _ConvBNLeaky(2, 1, 7)
        # Depthwise branches preserve the source module's two-path structure
        # without adding roughly another DenseNet-201 worth of parameters.
        self.branch_local = _ConvBNLeaky(half, half, 1, groups=half)
        self.branch_context_encoder = _ConvBNLeaky(half, half, 3, stride=2, groups=half)
        self.branch_context_decoder = _ConvBNLeaky(half, half, 3, groups=half)
        self.local_attention = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
        self.context_attention = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    @staticmethod
    def _mean_max(features: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (features.mean(dim=1, keepdim=True), features.amax(dim=1, keepdim=True)),
            dim=1,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        initial = torch.sigmoid(self.initial_attention(self._mean_max(features)))
        local, context = torch.chunk(features, 2, dim=1)
        local_features = self.branch_local(local)
        context_features = self.branch_context_encoder(context)
        context_features = functional.interpolate(
            context_features,
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        context_features = self.branch_context_decoder(context_features)
        local_gate = torch.sigmoid(self.local_attention(self._mean_max(local_features))) * initial
        context_gate = torch.sigmoid(self.context_attention(self._mean_max(context_features))) * initial
        attended = torch.cat((local * local_gate, context * context_gate), dim=1)
        return features + self.residual_scale * attended


class MultilayerFeatureFusion(nn.Module):
    """Channel-ranking MFF adaptation for a classification feature map.

    Channel importance is estimated with the paper's GAP plus one-dimensional
    convolution strategy. The most important half is retained in ranked order;
    the less-important half is neutralized to its mean representation. Weighting
    the maps before ranking keeps the attention estimator trainable.
    """

    def __init__(self, channels: int, positive_ratio: float = 0.5):
        super().__init__()
        if not 0.0 < positive_ratio < 1.0:
            raise ValueError("positive_ratio must be strictly between zero and one")
        self.channels = channels
        self.positive_channels = max(1, min(channels - 1, round(channels * positive_ratio)))
        self.channel_attention = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        descriptors = functional.adaptive_avg_pool2d(features, 1).flatten(2).transpose(1, 2)
        weights = torch.sigmoid(self.channel_attention(descriptors)).transpose(1, 2).unsqueeze(-1)
        weighted = features * weights
        ranked = torch.argsort(weights.flatten(1), dim=1, descending=True)
        gather_index = ranked[:, :, None, None].expand(-1, -1, features.shape[2], features.shape[3])
        ranked_features = torch.gather(weighted, dim=1, index=gather_index)
        positive = ranked_features[:, : self.positive_channels]
        negative = ranked_features[:, self.positive_channels :]
        neutral = negative.mean(dim=1, keepdim=True).expand_as(negative)
        return torch.cat((positive, neutral), dim=1)


class DenseNet201SASMFF(nn.Module):
    """ImageNet DenseNet-201 backbone with SAS and MFF before classification."""

    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.features = backbone.features
        feature_channels = int(backbone.classifier.in_features)
        self.sas = SpatialAttentionSuperposition(feature_channels)
        self.mff = MultilayerFeatureFusion(feature_channels, positive_ratio=0.5)
        self.classifier = nn.Linear(feature_channels, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = functional.relu(self.features(inputs), inplace=True)
        features = self.sas(features)
        features = self.mff(features)
        pooled = functional.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        return self.classifier(pooled)


def _torchvision_weights(models, architecture: str, weights: str):
    if weights == "none":
        return None, "random_initialization"
    if weights != "imagenet":
        raise ValueError("weights must be 'imagenet' or 'none' for torchvision architectures")
    enum_name = {
        "densenet121": "DenseNet121_Weights",
        "densenet169": "DenseNet169_Weights",
        "densenet201": "DenseNet201_Weights",
        "resnet50": "ResNet50_Weights",
        "resnet101": "ResNet101_Weights",
        "resnet152": "ResNet152_Weights",
    }[architecture]
    enum = getattr(models, enum_name)
    return enum.DEFAULT, f"torchvision::{enum_name}.DEFAULT"


def build_model(config):
    from torchvision import models

    architecture = config.architecture.lower()
    if architecture in YOLO_ARCHITECTURES:
        return _build_yolo_model(config)
    if architecture == "densenet201_sas_mff":
        weights, source = _torchvision_weights(models, "densenet201", config.weights)
        model = DenseNet201SASMFF(models.densenet201(weights=weights), config.num_classes)
        head_parameters = set(model.sas.parameters()) | set(model.mff.parameters()) | set(model.classifier.parameters())
        if config.freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = parameter in head_parameters
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        return model, ModelMetadata(
            architecture=architecture,
            weight_source=f"{source}; SAS-MFF classification adaptation initialized locally",
            num_classes=config.num_classes,
            parameter_count=total,
            trainable_parameter_count=trainable,
        )
    if architecture not in TORCHVISION_ARCHITECTURES:
        reasons = {
            "yolov5_cls": "YOLOv5 classification is handled by the verified YOLO adapter, not torchvision",
            "yolov8_cls": "YOLOv8 classification is handled by the verified Ultralytics adapter, not torchvision",
            "yolov10_cls": "YOLOv10 classification support has not been verified",
        }
        raise UnsupportedArchitectureError(reasons.get(architecture, f"unknown architecture: {architecture}"))

    constructor = getattr(models, architecture)
    weights, source = _torchvision_weights(models, architecture, config.weights)
    model = constructor(weights=weights)
    if architecture.startswith("densenet"):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, config.num_classes)
        head_parameters = set(model.classifier.parameters())
    else:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, config.num_classes)
        head_parameters = set(model.fc.parameters())
    if config.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = parameter in head_parameters
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    metadata = ModelMetadata(
        architecture=architecture,
        weight_source=source,
        num_classes=config.num_classes,
        parameter_count=total,
        trainable_parameter_count=trainable,
    )
    return model, metadata


def _reset_parameters(module) -> None:
    reset = getattr(module, "reset_parameters", None)
    if callable(reset):
        reset()


def _replace_ultralytics_head(model, num_classes: int):
    import torch.nn as nn

    head = model.model[-1]
    if head.__class__.__name__ != "Classify" or not hasattr(head, "linear"):
        raise UnsupportedArchitectureError(
            f"expected a Classify head but found {head.__class__.__name__}; detection heads are not accepted"
        )
    head.linear = nn.Linear(head.linear.in_features, num_classes)
    head.nc = num_classes
    model.nc = num_classes
    model.names = {index: name for index, name in enumerate(("Normal", "Viral", "Bacterial")[:num_classes])}
    return head


def _build_yolo_model(config):
    import torch

    cache = REPO_ROOT / ".research_cache"
    audit = cache / "yolo_assets.json"
    if not audit.exists():
        raise FileNotFoundError(
            "YOLO assets are not prepared; run .venv\\Scripts\\python.exe "
            "model\\scripts\\prepare_yolo_assets.py"
        )
    architecture = config.architecture
    if architecture == "yolov8_cls":
        if config.weights == "none":
            from ultralytics.nn.tasks import ClassificationModel

            model = ClassificationModel("yolov8n-cls.yaml", nc=config.num_classes, verbose=False)
            source = "random_initialization::ultralytics_yolov8n-cls.yaml"
        else:
            from ultralytics import YOLO

            path = cache / "weights/yolov8n-cls-v8.3.0.pt"
            model = YOLO(str(path), task="classify").model.float()
            _replace_ultralytics_head(model, config.num_classes)
            source = f"Ultralytics v8.3.0 asset::{path.name}"
    elif architecture == "yolov5_cls":
        repo = cache / "yolov5"
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        path = cache / "weights/yolov5s-cls-v7.0.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = checkpoint["model"].float()
        head = _replace_ultralytics_head(model, config.num_classes)
        if config.weights == "none":
            model.apply(_reset_parameters)
            source = "random_initialization::YOLOv5s-cls_v7.0_architecture"
        else:
            source = f"Ultralytics YOLOv5 v7.0 release::{path.name}"
    else:
        raise AssertionError(architecture)

    head = model.model[-1]
    head_parameters = set(head.parameters())
    if config.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = parameter in head_parameters
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    adapted = ClassificationLogitsAdapter(model)
    return adapted, ModelMetadata(
        architecture=architecture,
        weight_source=source,
        num_classes=config.num_classes,
        parameter_count=total,
        trainable_parameter_count=trainable,
    )


def assert_output_contract(model, input_size: int, num_classes: int, device) -> tuple[int, ...]:
    import torch

    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, input_size, input_size, device=device))
    if tuple(output.shape) != (2, num_classes):
        raise RuntimeError(f"expected output shape (2, {num_classes}), got {tuple(output.shape)}")
    return tuple(output.shape)
