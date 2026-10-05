"""ETF Anchor regularization: the proposed Geometric Memory method."""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from metrics.nc_metrics import TensorDict, centered_normalized_means, compute_class_means
from models.classifier import get_task_logits

from .base import ContinualMethod, LossOutput, to_device_batch


class MeanBuffer:
    """FIFO feature buffer for running class-mean estimates."""

    def __init__(self, buffer_size: int) -> None:
        self.buffer_size = int(buffer_size)
        self.storage: dict[int, deque[torch.Tensor]] = defaultdict(lambda: deque(maxlen=self.buffer_size))

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        with torch.no_grad():
            for feature, label in zip(features.detach().cpu(), labels.detach().cpu()):
                self.storage[int(label.item())].append(feature.clone())

    def get_means(self) -> TensorDict:
        means: TensorDict = {}
        for class_id, values in self.storage.items():
            if values:
                means[class_id] = torch.stack(list(values), dim=0).mean(dim=0)
        return means


@dataclass
class ExemplarBatch:
    images: torch.Tensor
    labels: torch.Tensor


class ExemplarBuffer:
    """Stores old-task images so anchor means are differentiable under the current model."""

    def __init__(self, buffer_size: int) -> None:
        self.buffer_size = int(buffer_size)
        self.images: dict[int, deque[torch.Tensor]] = defaultdict(lambda: deque(maxlen=self.buffer_size))
        self.labels: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=self.buffer_size))

    def add_batch(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        images = images.detach().cpu()
        labels = labels.detach().cpu()
        for image, label in zip(images, labels):
            class_id = int(label.item())
            self.images[class_id].append(image.clone())
            self.labels[class_id].append(class_id)

    def is_empty(self) -> bool:
        return not any(len(values) for values in self.images.values())

    def class_ids(self) -> list[int]:
        return sorted(class_id for class_id, values in self.images.items() if values)

    def as_batch(self, device: torch.device, max_per_class: int | None = None) -> ExemplarBatch:
        if self.is_empty():
            raise ValueError("Cannot sample from an empty exemplar buffer.")
        images: list[torch.Tensor] = []
        labels: list[int] = []
        for class_id in self.class_ids():
            class_images = list(self.images[class_id])
            if max_per_class is not None:
                class_images = class_images[-max_per_class:]
            images.extend(class_images)
            labels.extend([class_id] * len(class_images))
        return ExemplarBatch(
            images=torch.stack(images, dim=0).to(device, non_blocking=True),
            labels=torch.tensor(labels, dtype=torch.long, device=device),
        )

    def sample_balanced(self, device: torch.device, batch_size: int) -> ExemplarBatch:
        """Sample a roughly class-balanced anchor mini-batch."""
        if self.is_empty():
            raise ValueError("Cannot sample from an empty exemplar buffer.")
        class_ids = self.class_ids()
        per_class = max(1, int(batch_size) // max(1, len(class_ids)))
        images: list[torch.Tensor] = []
        labels: list[int] = []
        for class_id in class_ids:
            class_images = list(self.images[class_id])
            count = min(per_class, len(class_images))
            if count == len(class_images):
                selected = class_images
            else:
                permutation = torch.randperm(len(class_images))[:count].tolist()
                selected = [class_images[index] for index in permutation]
            images.extend(selected)
            labels.extend([class_id] * len(selected))

        if len(images) > batch_size:
            permutation = torch.randperm(len(images))[:batch_size].tolist()
            images = [images[index] for index in permutation]
            labels = [labels[index] for index in permutation]

        return ExemplarBatch(
            images=torch.stack(images, dim=0).to(device, non_blocking=True),
            labels=torch.tensor(labels, dtype=torch.long, device=device),
        )


class ETFAnchor(ContinualMethod):
    """ETF centroid and angular anchor regularizer."""

    name = "etf_anchor"

    def __init__(self) -> None:
        self.frozen_means: TensorDict = {}
        self.frozen_etf_matrix: torch.Tensor | None = None
        self.exemplar_buffer: ExemplarBuffer | None = None
        self.mean_buffer: MeanBuffer | None = None

    def freeze_geometry(
        self,
        model: torch.nn.Module,
        dataloader,
        task_id: int,
        config: dict[str, Any],
        device: torch.device | None = None,
    ) -> None:
        """Store task centroids, centered ETF directions, and exemplar images."""
        if device is None:
            device = next(model.parameters()).device
        buffer_size = int(config["buffer_size"])
        self.exemplar_buffer = ExemplarBuffer(buffer_size)
        self.mean_buffer = MeanBuffer(buffer_size)

        model.eval()
        all_features: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in dataloader:
                images, labels = to_device_batch(batch, device)
                self.exemplar_buffer.add_batch(images, labels)
                logits, features = model(images)
                del logits
                self.mean_buffer.update(features, labels)
                all_features.append(features.detach().cpu())
                all_labels.append(labels.detach().cpu())

        if not all_features:
            raise ValueError("Cannot freeze ETF geometry from an empty dataloader.")
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        self.frozen_means = compute_class_means(features, labels)
        _, normalized, _, _ = centered_normalized_means(self.frozen_means)
        self.frozen_etf_matrix = normalized.detach().cpu()

    def compute_centroid_term(
        self,
        current_means: TensorDict,
        frozen_means: TensorDict,
        mode: str = "raw",
    ) -> torch.Tensor:
        common_ids = sorted(set(current_means).intersection(frozen_means))
        if not common_ids:
            raise ValueError("No overlapping classes for centroid anchor.")

        mode = mode.lower()
        if mode not in {"raw", "centered", "normalized"}:
            raise ValueError("centroid_loss_mode must be raw, centered, or normalized.")

        if mode in {"centered", "normalized"}:
            current_stack = torch.stack([current_means[class_id] for class_id in common_ids], dim=0)
            frozen_stack = torch.stack(
                [
                    frozen_means[class_id].to(device=current_stack.device, dtype=current_stack.dtype)
                    for class_id in common_ids
                ],
                dim=0,
            )
            current_stack = current_stack - current_stack.mean(dim=0, keepdim=True)
            frozen_stack = frozen_stack - frozen_stack.mean(dim=0, keepdim=True)
            if mode == "normalized":
                current_stack = F.normalize(current_stack, dim=1)
                frozen_stack = F.normalize(frozen_stack, dim=1)
            return (current_stack - frozen_stack).pow(2).sum(dim=1).mean()

        terms = []
        for class_id in common_ids:
            current = current_means[class_id]
            frozen = frozen_means[class_id].to(device=current.device, dtype=current.dtype)
            terms.append((current - frozen).pow(2).sum())
        return torch.stack(terms).mean()

    def compute_angle_term(self, current_means: TensorDict, frozen_means: TensorDict | None = None) -> torch.Tensor:
        if frozen_means is not None:
            common_ids = sorted(set(current_means).intersection(frozen_means))
            current_means = {class_id: current_means[class_id] for class_id in common_ids}
        ids, normalized, _, _ = centered_normalized_means(current_means)
        num_classes = len(ids)
        if num_classes < 2:
            return normalized.new_tensor(0.0)
        gram = normalized @ normalized.T
        upper = torch.triu_indices(num_classes, num_classes, offset=1, device=gram.device)
        target = -1.0 / (num_classes - 1)
        return (gram[upper[0], upper[1]] - target).pow(2).mean()

    @contextmanager
    def _batchnorm_eval_context(self, model: torch.nn.Module, enabled: bool = True):
        """Temporarily stop BatchNorm running-stat updates during anchor forwards."""
        if not enabled:
            yield
            return

        batchnorm_modules = [
            module
            for module in model.modules()
            if isinstance(
                module,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.SyncBatchNorm,
                ),
            )
        ]
        states = [module.training for module in batchnorm_modules]
        try:
            for module in batchnorm_modules:
                module.eval()
            yield
        finally:
            for module, was_training in zip(batchnorm_modules, states):
                module.train(was_training)

    def etf_anchor_loss(
        self,
        model: torch.nn.Module,
        config: dict[str, Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute differentiable ETF anchor loss on stored old-task exemplars."""
        if not self.frozen_means or self.exemplar_buffer is None or self.exemplar_buffer.is_empty():
            zero = next(model.parameters()).new_tensor(0.0)
            return zero, zero, zero

        warmup_epochs = int(config.get("etf_warmup_epochs", 0) or 0)
        current_epoch = int(getattr(self, "current_epoch", 0))
        if current_epoch < warmup_epochs:
            zero = next(model.parameters()).new_tensor(0.0)
            return zero, zero, zero

        anchor_every = int(config.get("anchor_every_n_steps", 1) or 1)
        current_step = int(getattr(self, "current_global_step", 0))
        if anchor_every > 1 and current_step % anchor_every != 0:
            zero = next(model.parameters()).new_tensor(0.0)
            return zero, zero, zero

        anchor_batch_size = config.get("anchor_batch_size")
        if anchor_batch_size is None:
            max_per_class = int(config.get("anchor_samples_per_class", config["buffer_size"]))
            exemplar_batch = self.exemplar_buffer.as_batch(device, max_per_class=max_per_class)
        else:
            exemplar_batch = self.exemplar_buffer.sample_balanced(device, int(anchor_batch_size))
        freeze_bn = bool(config.get("freeze_bn_during_anchor", True))
        with self._batchnorm_eval_context(model, enabled=freeze_bn):
            _logits, features = model(exemplar_batch.images)
        current_means = compute_class_means(features, exemplar_batch.labels)
        centroid_term = self.compute_centroid_term(
            current_means,
            self.frozen_means,
            mode=str(config.get("centroid_loss_mode", "raw")),
        )
        angle_term = self.compute_angle_term(current_means, self.frozen_means)
        total = float(config.get("centroid_weight", 1.0)) * centroid_term + float(config["lambda_angle"]) * angle_term
        return total, centroid_term, angle_term

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

        etf_loss, centroid_term, angle_term = self.etf_anchor_loss(model, config, device)
        total_loss = ce_loss + float(config["alpha"]) * etf_loss
        with torch.no_grad():
            accuracy = (task_logits.argmax(dim=1) == labels).float().mean()
        return LossOutput(
            loss=total_loss,
            logs={
                "loss": float(total_loss.detach().cpu()),
                "ce_loss": float(ce_loss.detach().cpu()),
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
            self.freeze_geometry(model, dataloader, task_id, config, device=device)

    def train_task(self, model, train_loader, task_id: int, config: dict[str, Any]):
        """Compatibility wrapper for the roadmap API."""
        from training.trainer import Trainer

        trainer = Trainer(config)
        return trainer.train_task(model, self, train_loader, task_id)
