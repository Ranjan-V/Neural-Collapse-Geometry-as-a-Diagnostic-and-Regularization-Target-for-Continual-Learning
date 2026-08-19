"""Baseline 2: Elastic Weight Consolidation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F

from models.classifier import get_task_logits

from .base import ContinualMethod, LossOutput, to_device_batch


class EWC(ContinualMethod):
    """Diagonal Fisher EWC after Kirkpatrick et al. (2017)."""

    name = "ewc"

    def __init__(self) -> None:
        self.fisher_diagonals: list[dict[str, torch.Tensor]] = []
        self.parameter_snapshots: list[dict[str, torch.Tensor]] = []

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
        penalty = self.ewc_loss(model)
        total_loss = ce_loss + float(config["ewc_lambda"]) * penalty
        with torch.no_grad():
            accuracy = (task_logits.argmax(dim=1) == labels).float().mean()
        return LossOutput(
            loss=total_loss,
            logs={
                "loss": float(total_loss.detach().cpu()),
                "ce_loss": float(ce_loss.detach().cpu()),
                "ewc_loss": float(penalty.detach().cpu()),
                "batch_accuracy": float(accuracy.detach().cpu()),
            },
        )

    def compute_fisher(
        self,
        model: torch.nn.Module,
        dataloader,
        task_id: int,
        config: dict[str, Any],
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the diagonal Fisher approximation on one task."""
        if device is None:
            device = next(model.parameters()).device

        model.eval()
        fisher = {
            name: torch.zeros_like(param, device=device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        num_batches = 0

        for batch in dataloader:
            images, labels = to_device_batch(batch, device)
            model.zero_grad(set_to_none=True)
            logits, _features = model(images)
            task_logits = get_task_logits(logits, task_id, int(config["classes_per_task"]))
            log_probs = F.log_softmax(task_logits, dim=1)
            loss = F.nll_loss(log_probs, labels)
            loss.backward()
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2)
            num_batches += 1

        if num_batches == 0:
            raise ValueError("Cannot compute Fisher on an empty dataloader.")
        for name in fisher:
            fisher[name] /= num_batches
        return fisher

    def ewc_loss(self, model: torch.nn.Module) -> torch.Tensor:
        """Quadratic penalty against previous task optima."""
        if not self.fisher_diagonals:
            return next(model.parameters()).new_tensor(0.0)

        params = dict(model.named_parameters())
        total = next(model.parameters()).new_tensor(0.0)
        for fisher, snapshot in zip(self.fisher_diagonals, self.parameter_snapshots):
            for name, fisher_value in fisher.items():
                if name not in params:
                    continue
                current = params[name]
                previous = snapshot[name].to(device=current.device, dtype=current.dtype)
                fisher_value = fisher_value.to(device=current.device, dtype=current.dtype)
                if current.shape != previous.shape:
                    slices = tuple(slice(0, min(a, b)) for a, b in zip(current.shape, previous.shape))
                    current_part = current[slices]
                    previous = previous[slices]
                    fisher_value = fisher_value[slices]
                else:
                    current_part = current
                total = total + (fisher_value * (current_part - previous).pow(2)).sum()
        return 0.5 * total

    def after_task(
        self,
        model: torch.nn.Module,
        dataloader,
        task_id: int,
        config: dict[str, Any],
        device: torch.device,
    ) -> None:
        fisher = self.compute_fisher(model, dataloader, task_id, config, device=device)
        snapshot = {
            name: deepcopy(param.detach()).cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.fisher_diagonals.append({name: value.detach().cpu() for name, value in fisher.items()})
        self.parameter_snapshots.append(snapshot)

    def train_task(self, model, train_loader, task_id: int, config: dict[str, Any]):
        """Compatibility wrapper for the roadmap API."""
        from training.trainer import Trainer

        trainer = Trainer(config)
        return trainer.train_task(model, self, train_loader, task_id)

