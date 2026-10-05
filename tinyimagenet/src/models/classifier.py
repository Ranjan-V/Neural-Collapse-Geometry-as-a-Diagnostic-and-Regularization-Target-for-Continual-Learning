"""Expandable classifier heads for task-incremental training."""

from __future__ import annotations

import torch
from torch import nn


class IncrementalClassifier(nn.Module):
    """Linear classifier that can grow as new classes arrive."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        self.in_features = in_features
        self.classifier = nn.Linear(in_features, num_classes)
        self._freeze_until = 0
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []

    @property
    def out_features(self) -> int:
        return self.classifier.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.classifier.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.classifier.bias

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def expand(self, num_new_classes: int, freeze_old: bool = False) -> None:
        """Add output neurons while preserving existing logits."""
        if num_new_classes <= 0:
            raise ValueError("num_new_classes must be positive.")

        old_layer = self.classifier
        old_classes = old_layer.out_features
        new_classes = old_classes + num_new_classes
        new_layer = nn.Linear(self.in_features, new_classes)
        new_layer.to(device=old_layer.weight.device, dtype=old_layer.weight.dtype)

        with torch.no_grad():
            new_layer.weight[:old_classes].copy_(old_layer.weight)
            new_layer.bias[:old_classes].copy_(old_layer.bias)
            nn.init.kaiming_uniform_(new_layer.weight[old_classes:], a=5**0.5)
            fan_in = self.in_features
            bound = fan_in**-0.5
            new_layer.bias[old_classes:].uniform_(-bound, bound)

        self.classifier = new_layer
        self._clear_hooks()
        if freeze_old:
            self.freeze_old_classes(old_classes)

    def freeze_old_classes(self, num_old_classes: int | None = None) -> None:
        """Mask gradients for classifier rows belonging to previous tasks."""
        if num_old_classes is None:
            num_old_classes = self.out_features
        if not 0 <= num_old_classes <= self.out_features:
            raise ValueError("num_old_classes must be within classifier output size.")
        self._freeze_until = num_old_classes
        self._clear_hooks()

        if self._freeze_until == 0:
            return

        def mask_weight_grad(grad: torch.Tensor) -> torch.Tensor:
            grad = grad.clone()
            grad[: self._freeze_until].zero_()
            return grad

        def mask_bias_grad(grad: torch.Tensor) -> torch.Tensor:
            grad = grad.clone()
            grad[: self._freeze_until].zero_()
            return grad

        self._hook_handles.append(self.classifier.weight.register_hook(mask_weight_grad))
        if self.classifier.bias is not None:
            self._hook_handles.append(self.classifier.bias.register_hook(mask_bias_grad))

    def unfreeze_all(self) -> None:
        self._freeze_until = 0
        self._clear_hooks()

    def _clear_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()


def get_task_logits(logits: torch.Tensor, task_id: int, classes_per_task: int) -> torch.Tensor:
    """Extract logits for a task-aware training/evaluation slice."""
    start = task_id * classes_per_task
    end = start + classes_per_task
    if start < 0 or end > logits.shape[1]:
        raise ValueError(
            f"Task slice [{start}, {end}) is outside logits with {logits.shape[1]} classes."
        )
    return logits[:, start:end]

