"""Hybrid method: Learning without Forgetting plus ETF Anchor."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F

from models.classifier import get_task_logits

from .base import ContinualMethod, LossOutput, to_device_batch
from .etf_anchor import ETFAnchor
from .lwf import LwF


class LwFETFAnchor(ContinualMethod):
    """Combine output distillation with geometric ETF anchoring."""

    name = "lwf_etf"

    def __init__(self) -> None:
        self.teacher: torch.nn.Module | None = None
        self.old_num_classes = 0
        self.anchor = ETFAnchor()

    @property
    def frozen_means(self):
        return self.anchor.frozen_means

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
        return LwF().knowledge_distillation_loss(student_logits, teacher_logits, temperature)

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

        self.anchor.current_epoch = getattr(self, "current_epoch", 0)
        self.anchor.current_batch_index = getattr(self, "current_batch_index", 0)
        self.anchor.current_global_step = getattr(self, "current_global_step", 0)
        etf_loss, centroid_term, angle_term = self.anchor.etf_anchor_loss(model, config, device)

        total_loss = (
            ce_loss
            + float(config["lwf_alpha"]) * kd_loss
            + float(config["alpha"]) * etf_loss
        )
        with torch.no_grad():
            accuracy = (task_logits.argmax(dim=1) == labels).float().mean()
        return LossOutput(
            loss=total_loss,
            logs={
                "loss": float(total_loss.detach().cpu()),
                "ce_loss": float(ce_loss.detach().cpu()),
                "kd_loss": float(kd_loss.detach().cpu()),
                "etf_loss": float(etf_loss.detach().cpu()),
                "centroid_loss": float(centroid_term.detach().cpu()),
                "angle_loss": float(angle_term.detach().cpu()),
                "batch_accuracy": float(accuracy.detach().cpu()),
            },
        )

    def after_task(
        self,
        model: torch.nn.Module,
        dataloader,
        task_id: int,
        config: dict[str, Any],
        device: torch.device,
    ) -> None:
        if task_id == 0 or bool(config.get("anchor_all_seen_tasks", False)):
            self.anchor.freeze_geometry(model, dataloader, task_id, config, device=device)

