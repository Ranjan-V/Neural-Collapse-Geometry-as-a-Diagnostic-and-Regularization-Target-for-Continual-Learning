"""Shared method interfaces and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class LossOutput:
    """Loss tensor plus scalar values for logging."""

    loss: torch.Tensor
    logs: dict[str, float] = field(default_factory=dict)


class ContinualMethod:
    """Base class for continual-learning methods."""

    name = "base"

    def before_task(self, model: torch.nn.Module, task_id: int, config: dict[str, Any]) -> None:
        """Hook called immediately before a task starts training."""

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch,
        task_id: int,
        config: dict[str, Any],
        device: torch.device,
    ) -> LossOutput:
        raise NotImplementedError

    def after_task(
        self,
        model: torch.nn.Module,
        dataloader,
        task_id: int,
        config: dict[str, Any],
        device: torch.device,
    ) -> None:
        """Hook called after a task has finished training."""


def global_class_offset(task_id: int, classes_per_task: int) -> int:
    return int(task_id) * int(classes_per_task)


def to_device_batch(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = batch
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

