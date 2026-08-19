"""Single-task training and evaluation utilities."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler, autocast
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm

from methods.base import ContinualMethod
from models.classifier import get_task_logits


class Trainer:
    """Reusable trainer for one task at a time."""

    def __init__(self, config: dict[str, Any], device: torch.device | None = None) -> None:
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(config.get("mixed_precision", True)) and self.device.type == "cuda"
        self.scaler = GradScaler(self.device.type, enabled=self.use_amp)
        self.last_optimizer: Optimizer | None = None
        self.last_scheduler = None
        self.global_step = 0

    def build_optimizer(self, model: torch.nn.Module) -> Optimizer:
        return SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(self.config["learning_rate"]),
            momentum=float(self.config["momentum"]),
            weight_decay=float(self.config["weight_decay"]),
        )

    def build_scheduler(self, optimizer: Optimizer, steps_per_epoch: int):
        scheduler_name = str(self.config.get("lr_scheduler", "cosine")).lower()
        epochs = int(self.config["epochs_per_task"])
        if scheduler_name == "cosine":
            return CosineAnnealingLR(optimizer, T_max=max(1, epochs * steps_per_epoch))
        if scheduler_name == "multistep":
            milestones = self.config.get("lr_milestones", [60, 80])
            return MultiStepLR(optimizer, milestones=milestones, gamma=float(self.config.get("lr_gamma", 0.1)))
        if scheduler_name in {"none", "constant"}:
            return None
        raise ValueError(f"Unsupported lr_scheduler: {scheduler_name!r}")

    def train_epoch(
        self,
        model: torch.nn.Module,
        method: ContinualMethod,
        loader,
        optimizer: Optimizer,
        scheduler,
        task_id: int,
        epoch: int = 0,
        step_callback=None,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        model.train()
        running: dict[str, list[float]] = defaultdict(list)
        step_logs: list[dict[str, Any]] = []
        progress = tqdm(loader, desc=f"task {task_id} epoch {epoch}", leave=False)

        for batch_index, batch in enumerate(progress):
            optimizer.zero_grad(set_to_none=True)
            setattr(method, "current_epoch", epoch)
            setattr(method, "current_batch_index", batch_index)
            setattr(method, "current_global_step", self.global_step)
            with autocast(self.device.type, enabled=self.use_amp):
                output = method.compute_loss(model, batch, task_id, self.config, self.device)
                loss = output.loss

            self.scaler.scale(loss).backward()
            max_norm = float(self.config.get("gradient_clip_max_norm", 0.0))
            if max_norm > 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            old_scale = self.scaler.get_scale()
            self.scaler.step(optimizer)
            self.scaler.update()
            new_scale = self.scaler.get_scale()
            optimizer_step_was_skipped = self.use_amp and new_scale < old_scale
            if scheduler is not None and not isinstance(scheduler, MultiStepLR):
                if not optimizer_step_was_skipped:
                    scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            for key, value in output.logs.items():
                running[key].append(float(value))
            running["lr"].append(float(lr))
            step_row: dict[str, Any] = {
                "record_type": "step",
                "global_step": self.global_step,
                "task_id": task_id,
                "epoch": epoch,
                "batch_index": batch_index,
                "lr": float(lr),
            }
            step_row.update({key: float(value) for key, value in output.logs.items()})
            step_logs.append(step_row)
            if step_callback is not None:
                step_callback(model, method, step_row)
            self.global_step += 1
            if "loss" in output.logs:
                progress.set_postfix(loss=f"{output.logs['loss']:.4f}", lr=f"{lr:.3e}")

        if scheduler is not None and isinstance(scheduler, MultiStepLR):
            scheduler.step()

        return {key: sum(values) / max(1, len(values)) for key, values in running.items()}, step_logs

    def train_task(
        self,
        model: torch.nn.Module,
        method: ContinualMethod,
        loader,
        task_id: int,
        optimizer: Optimizer | None = None,
        scheduler=None,
        step_callback=None,
    ) -> list[dict[str, Any]]:
        model.to(self.device)
        if optimizer is None:
            optimizer = self.build_optimizer(model)
        if scheduler is None:
            scheduler = self.build_scheduler(optimizer, max(1, len(loader)))
        self.last_optimizer = optimizer
        self.last_scheduler = scheduler

        logs: list[dict[str, Any]] = []
        for epoch in range(int(self.config["epochs_per_task"])):
            epoch_log, step_logs = self.train_epoch(
                model,
                method,
                loader,
                optimizer,
                scheduler,
                task_id,
                epoch,
                step_callback=step_callback,
            )
            logs.extend(step_logs)
            epoch_log["epoch"] = epoch
            epoch_log["task_id"] = task_id
            epoch_log["record_type"] = "epoch"
            logs.append(epoch_log)
        return logs

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, loader, task_id: int) -> dict[str, float]:
        model.eval()
        model.to(self.device)
        classes_per_task = int(self.config["classes_per_task"])
        top1_correct = 0
        top5_correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits, _features = model(images)
            task_logits = get_task_logits(logits, task_id, classes_per_task)
            total += labels.numel()
            predictions = task_logits.argmax(dim=1)
            top1_correct += int((predictions == labels).sum().item())
            k = min(5, task_logits.shape[1])
            topk = task_logits.topk(k=k, dim=1).indices
            top5_correct += int((topk == labels.unsqueeze(1)).any(dim=1).sum().item())

        if total == 0:
            raise ValueError("Cannot evaluate an empty dataloader.")
        return {
            "top1": top1_correct / total,
            "top5": top5_correct / total,
        }

    @torch.no_grad()
    def collect_outputs(
        self,
        model: torch.nn.Module,
        loader,
        task_id: int,
        max_batches: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collect task-local logits, features, and local labels."""
        model.eval()
        model.to(self.device)
        classes_per_task = int(self.config["classes_per_task"])
        all_features: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_logits: list[torch.Tensor] = []

        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits, features = model(images)
            task_logits = get_task_logits(logits, task_id, classes_per_task)
            all_features.append(features.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_logits.append(task_logits.detach().cpu())

        if not all_features:
            raise ValueError("Cannot collect outputs from an empty dataloader.")
        return (
            torch.cat(all_features, dim=0),
            torch.cat(all_labels, dim=0),
            torch.cat(all_logits, dim=0),
        )


def save_training_log(logs: list[dict[str, Any]], path: str | Path) -> None:
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(logs).to_csv(path, index=False)
