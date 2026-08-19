"""Baseline 3: Learning without Forgetting."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F

from models.classifier import get_task_logits

from .base import ContinualMethod, LossOutput, to_device_batch


class LwF(ContinualMethod):
    """Learning without Forgetting with a frozen teacher from the previous task."""

    name = "lwf"

    def __init__(self) -> None:
        self.teacher: torch.nn.Module | None = None
        self.old_num_classes = 0

    def before_task(self, model: torch.nn.Module, task_id: int, config: dict[str, Any]) -> None:
        if task_id == 0:
            self.teacher = None
            self.old_num_classes = 0
            return
        self.teacher = deepcopy(model).eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.old_num_classes = task_id * int(config["classes_per_task"])

    def knowledge_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """KL divergence between softened old-class distributions."""
        log_student = F.log_softmax(student_logits / temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
        return F.kl_div(log_student, soft_teacher, reduction="batchmean") * (temperature**2)

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
        ce_loss = F.cross_entropy(task_logits, labels)

        kd_loss = logits.new_tensor(0.0)
        if self.teacher is not None and self.old_num_classes > 0:
            self.teacher.to(device)
            with torch.no_grad():
                teacher_logits, _ = self.teacher(images)
            kd_loss = self.knowledge_distillation_loss(
                logits[:, : self.old_num_classes],
                teacher_logits[:, : self.old_num_classes],
                float(config["lwf_temperature"]),
            )

        total_loss = ce_loss + float(config["lwf_alpha"]) * kd_loss
        with torch.no_grad():
            accuracy = (task_logits.argmax(dim=1) == labels).float().mean()
        return LossOutput(
            loss=total_loss,
            logs={
                "loss": float(total_loss.detach().cpu()),
                "ce_loss": float(ce_loss.detach().cpu()),
                "kd_loss": float(kd_loss.detach().cpu()),
                "batch_accuracy": float(accuracy.detach().cpu()),
            },
        )

    def train_task(self, model, train_loader, task_id: int, config: dict[str, Any]):
        """Compatibility wrapper for the roadmap API."""
        from training.trainer import Trainer

        trainer = Trainer(config)
        return trainer.train_task(model, self, train_loader, task_id)

