"""Neural collapse and ETF drift metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


TensorDict = dict[int, torch.Tensor]


def _sorted_class_ids(class_means: TensorDict) -> list[int]:
    if not class_means:
        raise ValueError("class_means must contain at least one class.")
    return sorted(int(k) for k in class_means)


def _stack_means(class_means: TensorDict, class_ids: Iterable[int] | None = None) -> torch.Tensor:
    ids = list(class_ids) if class_ids is not None else _sorted_class_ids(class_means)
    return torch.stack([class_means[int(class_id)] for class_id in ids], dim=0)


def compute_class_means(features: torch.Tensor, labels: torch.Tensor) -> TensorDict:
    """Compute class means as a dict `{class_id: mean_vector}`."""
    if features.ndim != 2:
        raise ValueError("features must have shape [num_samples, feature_dim].")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must have shape [num_samples].")

    labels = labels.to(device=features.device)
    class_means: TensorDict = {}
    for class_id in torch.unique(labels, sorted=True):
        mask = labels == class_id
        class_means[int(class_id.item())] = features[mask].mean(dim=0)
    return class_means


def centered_normalized_means(
    class_means: TensorDict,
    eps: float = 1e-12,
) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ids, centered-normalized means, centered means, and global mean."""
    ids = _sorted_class_ids(class_means)
    means = _stack_means(class_means, ids)
    global_mean = means.mean(dim=0)
    centered = means - global_mean
    normalized = centered / centered.norm(dim=1, keepdim=True).clamp_min(eps)
    return ids, normalized, centered, global_mean


def compute_within_class_cov(features: torch.Tensor, labels: torch.Tensor) -> TensorDict:
    """Compute within-class covariance matrices using the NC1 definition."""
    class_means = compute_class_means(features, labels)
    labels = labels.to(device=features.device)
    covariances: TensorDict = {}
    for class_id, mean in class_means.items():
        class_features = features[labels == class_id]
        centered = class_features - mean
        covariances[class_id] = centered.T @ centered / class_features.shape[0]
    return covariances


def compute_between_class_cov(class_means: TensorDict) -> torch.Tensor:
    """Compute between-class covariance from class means."""
    _, _, centered, _ = centered_normalized_means(class_means)
    return centered.T @ centered / centered.shape[0]


def compute_nc1(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """NC1: average trace of within-class covariance times between-class pseudoinverse."""
    class_means = compute_class_means(features, labels)
    within_covs = compute_within_class_cov(features, labels)
    between_cov = compute_between_class_cov(class_means)
    between_inv = torch.linalg.pinv(between_cov)
    traces = [torch.trace(cov @ between_inv) for cov in within_covs.values()]
    return torch.stack(traces).mean()


def compute_nc2(class_means: TensorDict) -> torch.Tensor:
    """NC2: mean squared deviation from simplex ETF pairwise cosines."""
    ids, normalized, _, _ = centered_normalized_means(class_means)
    num_classes = len(ids)
    if num_classes < 2:
        return normalized.new_tensor(0.0)

    gram = normalized @ normalized.T
    upper = torch.triu_indices(num_classes, num_classes, offset=1, device=gram.device)
    target = -1.0 / (num_classes - 1)
    residual = gram[upper[0], upper[1]] - target
    return residual.pow(2).mean()


def compute_nc3(class_means: TensorDict, classifier_weights: torch.Tensor) -> torch.Tensor:
    """NC3: average one-minus-cosine between centered means and classifier weights."""
    ids, _, centered, _ = centered_normalized_means(class_means)
    max_id = max(ids)
    if classifier_weights.shape[0] <= max_id:
        raise ValueError("classifier_weights must contain one row per class id.")
    weights = torch.stack([classifier_weights[class_id] for class_id in ids], dim=0)
    cosines = F.cosine_similarity(weights, centered, dim=1)
    return torch.abs(1.0 - cosines).mean()


def compute_nc4(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_means: TensorDict,
    logits: torch.Tensor,
) -> torch.Tensor:
    """NC4: fraction of samples where nearest-centroid and softmax predictions agree."""
    if logits.shape[0] != features.shape[0]:
        raise ValueError("logits and features must have the same batch dimension.")
    ids = _sorted_class_ids(class_means)
    means = _stack_means(class_means, ids).to(device=features.device, dtype=features.dtype)
    distances = torch.cdist(features, means)
    ncc_indices = distances.argmin(dim=1)
    ncc_preds = torch.tensor(ids, device=features.device, dtype=labels.dtype)[ncc_indices]
    softmax_preds = logits.argmax(dim=1).to(dtype=labels.dtype)
    return (ncc_preds == softmax_preds).float().mean()


def compute_etf_drift(current_means: TensorDict, frozen_means: TensorDict) -> torch.Tensor:
    """ETF centroid drift: average squared movement from frozen means."""
    common_ids = sorted(set(current_means).intersection(frozen_means))
    if not common_ids:
        raise ValueError("current_means and frozen_means share no class ids.")
    drifts = [
        (current_means[class_id] - frozen_means[class_id].to(current_means[class_id])).pow(2).sum()
        for class_id in common_ids
    ]
    return torch.stack(drifts).mean()


def compute_angle_drift(current_means: TensorDict, frozen_means: TensorDict | None = None) -> torch.Tensor:
    """Angle drift of current old-task means relative to the ETF target."""
    if frozen_means is None:
        selected_means = current_means
    else:
        common_ids = sorted(set(current_means).intersection(frozen_means))
        if not common_ids:
            raise ValueError("current_means and frozen_means share no class ids.")
        selected_means = {class_id: current_means[class_id] for class_id in common_ids}
    return compute_nc2(selected_means)


def compute_all_nc_metrics(
    features: torch.Tensor,
    labels: torch.Tensor,
    classifier_weights: torch.Tensor,
    logits: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute NC1--NC4 as Python floats for logging."""
    with torch.no_grad():
        class_means = compute_class_means(features, labels)
        metrics = {
            "nc1": float(compute_nc1(features, labels).detach().cpu()),
            "nc2": float(compute_nc2(class_means).detach().cpu()),
            "nc3": float(compute_nc3(class_means, classifier_weights).detach().cpu()),
        }
        if logits is not None:
            metrics["nc4"] = float(compute_nc4(features, labels, class_means, logits).detach().cpu())
        else:
            metrics["nc4"] = float("nan")
    return metrics


@dataclass
class NCMetricsLogger:
    """In-memory logger for NC metrics and geometric drift."""

    rows: list[dict[str, float | int]] = field(default_factory=list)

    def log(
        self,
        step: int,
        task_id: int,
        features: torch.Tensor,
        labels: torch.Tensor,
        classifier_weights: torch.Tensor,
        logits: torch.Tensor | None = None,
        frozen_means: TensorDict | None = None,
    ) -> dict[str, float | int]:
        with torch.no_grad():
            class_means = compute_class_means(features, labels)
            metrics = compute_all_nc_metrics(features, labels, classifier_weights, logits)
            row: dict[str, float | int] = {
                "step": int(step),
                "task_id": int(task_id),
                **metrics,
            }
            if frozen_means is not None:
                row["etf_drift"] = float(compute_etf_drift(class_means, frozen_means).detach().cpu())
                row["angle_drift"] = float(compute_angle_drift(class_means, frozen_means).detach().cpu())
            self.rows.append(row)
        return row

    def get_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.rows)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe = self.get_dataframe()
        if path.suffix.lower() == ".json":
            dataframe.to_json(path, orient="records", indent=2)
        else:
            dataframe.to_csv(path, index=False)

