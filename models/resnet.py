"""ResNet-18 feature extractor with checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .classifier import IncrementalClassifier


try:
    from torchvision.models import ResNet18_Weights, resnet18
except Exception as exc:  # pragma: no cover - handled at runtime
    ResNet18_Weights = None
    resnet18 = None
    _TORCHVISION_IMPORT_ERROR = exc
else:
    _TORCHVISION_IMPORT_ERROR = None


class FeatureExtractor(nn.Module):
    """Wrap a backbone and classifier, returning both logits and features."""

    def __init__(self, backbone: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.features: torch.Tensor | None = None

        if hasattr(self.backbone, "avgpool"):
            self.backbone.avgpool.register_forward_hook(self._store_avgpool_features)

    def _store_avgpool_features(self, _module, _inputs, output) -> None:
        self.features = torch.flatten(output, 1).detach()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        self.features = features
        logits = self.classifier(features)
        return logits, features


def build_resnet18(
    feature_dim: int,
    num_classes: int,
    pretrained: bool = False,
    incremental_head: bool = True,
) -> FeatureExtractor:
    """Build ResNet-18 with the final FC replaced by an identity layer."""
    if resnet18 is None:
        raise ImportError("torchvision is required to build ResNet-18.") from _TORCHVISION_IMPORT_ERROR

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    backbone = resnet18(weights=weights)
    detected_dim = backbone.fc.in_features
    if feature_dim != detected_dim:
        raise ValueError(
            f"ResNet-18 feature_dim is {detected_dim}; got config feature_dim={feature_dim}."
        )
    backbone.fc = nn.Identity()

    if incremental_head:
        classifier: nn.Module = IncrementalClassifier(feature_dim, num_classes)
    else:
        classifier = nn.Linear(feature_dim, num_classes)
    return FeatureExtractor(backbone, classifier)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    task_id: int,
    path: str | Path,
    **extra: Any,
) -> None:
    """Save model, optimizer, and task metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "task_id": task_id,
        **extra,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module | None, torch.optim.Optimizer | None, int, dict[str, Any]]:
    """Load a checkpoint, optionally restoring model and optimizer objects."""
    checkpoint = torch.load(path, map_location=map_location)
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, int(checkpoint["task_id"]), checkpoint

