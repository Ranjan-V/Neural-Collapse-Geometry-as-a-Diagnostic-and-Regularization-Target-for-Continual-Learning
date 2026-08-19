"""Baseline 1: naive finetuning."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from models.classifier import get_task_logits

from .base import ContinualMethod, LossOutput, to_device_batch


class SimpleFinetuner(ContinualMethod):
    """Standard SGD training with cross entropy and no forgetting protection."""

    name = "finetune"

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch,
        task_id: int,
        config: dict[str, Any],
        device: torch.device,
    ) -> LossOutput:
        images, labels = to_device_batch(batch, device)
        logits, _features = model(images)
        task_logits = get_task_logits(logits, task_id, int(config["classes_per_task"]))
        loss = F.cross_entropy(task_logits, labels)
        with torch.no_grad():
            accuracy = (task_logits.argmax(dim=1) == labels).float().mean()
        return LossOutput(
            loss=loss,
            logs={
                "loss": float(loss.detach().cpu()),
                "ce_loss": float(loss.detach().cpu()),
                "batch_accuracy": float(accuracy.detach().cpu()),
            },
        )

    def train_task(self, model, train_loader, task_id: int, config: dict[str, Any]):
        """Compatibility wrapper for the roadmap API."""
        from training.trainer import Trainer

        trainer = Trainer(config)
        return trainer.train_task(model, self, train_loader, task_id)

