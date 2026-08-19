"""Sequential task orchestration for continual-learning experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.dataset import ContinualDataset
from methods.base import ContinualMethod
from methods.etf_anchor import ETFAnchor
from methods.ewc import EWC
from methods.finetune import SimpleFinetuner
from methods.lwf import LwF
from methods.lwf_etf import LwFETFAnchor
from metrics.nc_metrics import NCMetricsLogger
from models.resnet import build_resnet18, save_checkpoint
from utils import ensure_output_dirs, set_random_seed

from .trainer import Trainer, save_training_log


METHOD_REGISTRY = {
    "finetune": SimpleFinetuner,
    "ewc": EWC,
    "lwf": LwF,
    "lwf_etf": LwFETFAnchor,
    "lwf-etf": LwFETFAnchor,
    "etf_anchor": ETFAnchor,
    "etf-anchor": ETFAnchor,
}


def build_method(method_name: str) -> ContinualMethod:
    key = method_name.lower()
    if key not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method {method_name!r}. Available: {sorted(METHOD_REGISTRY)}")
    return METHOD_REGISTRY[key]()


class ContinualTrainer:
    """Runs task-incremental experiments and records paper metrics."""

    def __init__(
        self,
        config: dict[str, Any],
        dataset: ContinualDataset | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        set_random_seed(int(config["seed"]), deterministic=bool(config.get("deterministic", True)))
        ensure_output_dirs(config)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset or ContinualDataset(config)
        self.trainer = Trainer(config, device=self.device)

    def build_model(self) -> torch.nn.Module:
        model = build_resnet18(
            feature_dim=int(self.config["feature_dim"]),
            num_classes=int(self.config["classes_per_task"]),
            pretrained=bool(self.config.get("pretrained", False)),
        )
        return model.to(self.device)

    def _experiment_dir(self, base_key: str) -> Path:
        experiment_name = str(self.config.get("experiment_name", "geometric_memory"))
        safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in experiment_name)
        return Path(self.config[base_key]) / safe_name

    def _expand_for_task(self, model: torch.nn.Module, task_id: int) -> None:
        if task_id == 0:
            return
        classifier = getattr(model, "classifier", None)
        if hasattr(classifier, "expand"):
            classifier.expand(
                int(self.config["classes_per_task"]),
                freeze_old=bool(self.config.get("freeze_old_classifier", False)),
            )
        else:
            raise TypeError("Model classifier must support expand() for continual training.")
        model.to(self.device)

    def _classifier_weights_for_task(self, model: torch.nn.Module, task_id: int) -> torch.Tensor:
        classes_per_task = int(self.config["classes_per_task"])
        start = task_id * classes_per_task
        end = start + classes_per_task
        weights = model.classifier.weight.detach().cpu()
        return weights[start:end]

    def _compute_nc_metrics_for_seen_tasks(
        self,
        model: torch.nn.Module,
        trained_task_id: int,
        nc_logger: NCMetricsLogger,
        method: ContinualMethod,
    ) -> None:
        max_batches = self.config.get("nc_eval_max_batches")
        if max_batches is not None:
            max_batches = int(max_batches)
        for eval_task_id in range(trained_task_id + 1):
            loader = self.dataset.get_task(eval_task_id, split="test")
            features, labels, task_logits = self.trainer.collect_outputs(
                model, loader, eval_task_id, max_batches=max_batches
            )
            frozen_means = None
            if eval_task_id == 0 and hasattr(method, "frozen_means") and method.frozen_means:
                frozen_means = method.frozen_means
            row = nc_logger.log(
                step=trained_task_id,
                task_id=eval_task_id,
                features=features,
                labels=labels,
                classifier_weights=self._classifier_weights_for_task(model, eval_task_id),
                logits=task_logits,
                frozen_means=frozen_means,
            )
            row["trained_task_id"] = trained_task_id

    def _make_step_callback(
        self,
        task_id: int,
        nc_logger: NCMetricsLogger,
        method: ContinualMethod,
        accuracy_matrix: np.ndarray,
    ):
        eval_every = int(self.config.get("eval_every_n_steps", 0) or 0)
        if eval_every <= 0:
            return None

        def callback(model: torch.nn.Module, callback_method: ContinualMethod, step_row: dict[str, Any]) -> None:
            global_step = int(step_row["global_step"])
            if global_step % eval_every != 0:
                return
            if task_id == 0:
                return

            was_training = model.training
            loader = self.dataset.get_task(0, split="test")
            max_batches = self.config.get("nc_eval_max_batches")
            if max_batches is not None:
                max_batches = int(max_batches)
            features, labels, task_logits = self.trainer.collect_outputs(
                model, loader, 0, max_batches=max_batches
            )
            frozen_means = None
            if hasattr(callback_method, "frozen_means") and callback_method.frozen_means:
                frozen_means = callback_method.frozen_means
            weights = self._classifier_weights_for_task(model, 0)
            row = nc_logger.log(
                step=global_step,
                task_id=0,
                features=features,
                labels=labels,
                classifier_weights=weights,
                logits=task_logits,
                frozen_means=frozen_means,
            )
            task1_metrics = self.trainer.evaluate(model, loader, 0)
            row["trained_task_id"] = task_id
            row["task1_accuracy"] = task1_metrics["top1"]
            if np.isfinite(accuracy_matrix[0, 0]):
                row["task1_forgetting"] = float(accuracy_matrix[0, 0] - task1_metrics["top1"])
            if was_training:
                model.train()

        return callback

    def run_experiment(self, method_name: str) -> dict[str, Any]:
        method = build_method(method_name)
        model = self.build_model()
        num_tasks = int(self.config["num_tasks"])
        accuracy_matrix = np.full((num_tasks, num_tasks), np.nan, dtype=float)
        all_training_logs: list[dict[str, Any]] = []
        nc_logger = NCMetricsLogger()

        for task_id in range(num_tasks):
            method.before_task(model, task_id, self.config)
            self._expand_for_task(model, task_id)

            train_loader = self.dataset.get_task(task_id, split="train")
            step_callback = self._make_step_callback(task_id, nc_logger, method, accuracy_matrix)
            task_logs = self.trainer.train_task(
                model,
                method,
                train_loader,
                task_id,
                step_callback=step_callback,
            )
            for row in task_logs:
                row["method"] = method.name
            all_training_logs.extend(task_logs)

            method.after_task(model, train_loader, task_id, self.config, self.device)

            checkpoint_path = (
                self._experiment_dir("checkpoint_dir")
                / method.name
                / f"seed_{self.config['seed']}_task_{task_id}.pt"
            )
            save_checkpoint(model, self.trainer.last_optimizer, task_id, checkpoint_path, method=method.name)

            for eval_task_id in range(task_id + 1):
                test_loader = self.dataset.get_task(eval_task_id, split="test")
                metrics = self.trainer.evaluate(model, test_loader, eval_task_id)
                accuracy_matrix[task_id, eval_task_id] = metrics["top1"]

            self._compute_nc_metrics_for_seen_tasks(model, task_id, nc_logger, method)

        result_dir = self._experiment_dir("log_dir") / method.name / f"seed_{self.config['seed']}"
        result_dir.mkdir(parents=True, exist_ok=True)
        np.save(result_dir / "accuracy_matrix.npy", accuracy_matrix)
        save_training_log(all_training_logs, result_dir / "training_log.csv")
        nc_logger.save(result_dir / "nc_metrics.csv")

        summary = {
            "method": method.name,
            "seed": int(self.config["seed"]),
            "accuracy_matrix": accuracy_matrix.tolist(),
            "training_log_path": str(result_dir / "training_log.csv"),
            "nc_metrics_path": str(result_dir / "nc_metrics.csv"),
        }
        with (result_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary
