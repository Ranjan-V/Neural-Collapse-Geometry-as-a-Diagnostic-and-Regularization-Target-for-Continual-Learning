"""Publication-quality figures for the Geometric Memory paper."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


SINGLE_COLUMN_WIDTH = 3.5
DOUBLE_COLUMN_WIDTH = 7.0
DPI = 300

METHOD_COLORS = {
    "finetune": "#C43B3B",
    "ewc": "#3B6FB6",
    "lwf": "#2E8B57",
    "etf_anchor": "#6A4C93",
    "etf-anchor": "#6A4C93",
}


def _mpl():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.6,
            "patch.linewidth": 0.8,
        }
    )
    return plt


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig, output_dir: str | Path, stem: str) -> dict[str, str]:
    output_dir = _ensure_dir(output_dir)
    paths = {
        "pdf": output_dir / f"{stem}.pdf",
        "png": output_dir / f"{stem}.png",
    }
    fig.tight_layout()
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["png"], bbox_inches="tight")
    return {key: str(value) for key, value in paths.items()}


def _color_for(method: str) -> str:
    return METHOD_COLORS.get(method, "#333333")


def plot_nc_metrics(nc_metrics: pd.DataFrame, output_dir: str | Path, stem: str = "fig1_nc_metrics"):
    """Figure 1: NC1--NC4 over training steps."""
    plt = _mpl()
    metrics = [
        ("nc1", "NC1"),
        ("nc2", "NC2"),
        ("nc3", "NC3"),
        ("nc4", "NC4"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_WIDTH, 4.6), sharex=True)
    axes = axes.ravel()
    if "method" not in nc_metrics:
        nc_metrics = nc_metrics.assign(method="run")

    for ax, (metric, label) in zip(axes, metrics):
        if metric not in nc_metrics:
            ax.set_visible(False)
            continue
        for method, group in nc_metrics.groupby("method"):
            group = group.sort_values("step")
            ax.plot(group["step"], group[metric], label=method, color=_color_for(method))
        ax.set_title(label)
        ax.set_xlabel("Training step")
        ax.set_ylabel(label)
        ax.grid(True, color="#E6E6E6", linewidth=0.6)
    axes[0].legend(frameon=False, ncol=2)
    return _save(fig, output_dir, stem)


def plot_drift_vs_forgetting(
    drift_frame: pd.DataFrame,
    accuracy_by_method: Mapping[str, np.ndarray],
    output_dir: str | Path,
    stem: str = "fig2_drift_forgetting",
):
    """Figure 2: ETF drift against task-1 forgetting."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.75))
    rows = []

    for method, matrix in accuracy_by_method.items():
        if "method" in drift_frame:
            group = drift_frame[drift_frame["method"] == method]
        else:
            group = drift_frame
        if "etf_drift" not in group or group.empty:
            continue
        group = group[np.isfinite(group["etf_drift"].to_numpy(dtype=float))]
        if group.empty:
            continue
        if "task1_forgetting" in group:
            x = group["etf_drift"].to_numpy(dtype=float)
            y = group["task1_forgetting"].to_numpy(dtype=float)
        else:
            matrix = np.asarray(matrix, dtype=float)
            forgetting = matrix[0, 0] - matrix[:, 0]
            length = min(len(group), len(forgetting))
            x = group["etf_drift"].to_numpy(dtype=float)[:length]
            y = forgetting[:length]
        ax.scatter(x, y, s=22, alpha=0.82, label=method, color=_color_for(method), edgecolor="white", linewidth=0.4)
        rows.extend(zip(x, y))

    if len(rows) >= 2:
        x_all = np.asarray([row[0] for row in rows], dtype=float)
        y_all = np.asarray([row[1] for row in rows], dtype=float)
        mask = np.isfinite(x_all) & np.isfinite(y_all)
        if mask.sum() >= 2 and np.std(x_all[mask]) > 0 and np.std(y_all[mask]) > 0:
            r = np.corrcoef(x_all[mask], y_all[mask])[0, 1]
            ax.text(0.03, 0.95, f"Pearson $r={r:.2f}$", transform=ax.transAxes, va="top")

    ax.set_xlabel("$\\Delta_{ETF}$")
    ax.set_ylabel("Task-1 accuracy drop")
    ax.grid(True, color="#E6E6E6", linewidth=0.6)
    ax.legend(frameon=False)
    return _save(fig, output_dir, stem)


def plot_accuracy_heatmaps(
    accuracy_by_method: Mapping[str, np.ndarray],
    output_dir: str | Path,
    stem: str = "fig3_accuracy_heatmaps",
):
    """Figure 3: accuracy matrix heatmaps."""
    plt = _mpl()
    num_methods = len(accuracy_by_method)
    cols = min(2, max(1, num_methods))
    rows = int(np.ceil(num_methods / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(DOUBLE_COLUMN_WIDTH, 3.0 * rows), squeeze=False)
    axes_flat = axes.ravel()

    image = None
    for ax, (method, matrix) in zip(axes_flat, accuracy_by_method.items()):
        matrix = np.asarray(matrix, dtype=float)
        masked = np.ma.masked_invalid(matrix)
        image = ax.imshow(masked, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(method)
        ax.set_xlabel("Evaluated task")
        ax.set_ylabel("After training task")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(matrix.shape[0]))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    text_color = "white" if matrix[i, j] < 0.55 else "black"
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6, color=text_color)

    for ax in axes_flat[num_methods:]:
        ax.set_visible(False)
    if image is not None:
        fig.colorbar(image, ax=axes_flat[:num_methods], fraction=0.035, pad=0.02, label="Accuracy")
    return _save(fig, output_dir, stem)


def plot_average_accuracy_comparison(
    summary: pd.DataFrame,
    output_dir: str | Path,
    stem: str = "fig4_average_accuracy",
):
    """Figure 4: average final accuracy by method with seed-level error bars."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.55))
    if "average_accuracy_mean" in summary:
        methods = summary["method"].tolist()
        means = summary["average_accuracy_mean"].to_numpy(dtype=float)
        stds = summary.get("average_accuracy_std", pd.Series(np.zeros(len(summary)))).fillna(0.0).to_numpy(dtype=float)
    else:
        grouped = summary.groupby("method")["average_accuracy"].agg(["mean", "std"]).reset_index()
        methods = grouped["method"].tolist()
        means = grouped["mean"].to_numpy(dtype=float)
        stds = grouped["std"].fillna(0.0).to_numpy(dtype=float)

    colors = [_color_for(method) for method in methods]
    ax.bar(methods, means, yerr=stds, capsize=3, color=colors, edgecolor="#222222", linewidth=0.5)
    ax.set_ylabel("Average accuracy")
    ax.set_ylim(0.0, max(1.0, float(np.nanmax(means + stds)) * 1.1))
    ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.6)
    ax.tick_params(axis="x", rotation=20)
    return _save(fig, output_dir, stem)


def plot_forgetting_curves(
    accuracy_by_method: Mapping[str, np.ndarray],
    output_dir: str | Path,
    stem: str = "fig5_forgetting_curves",
):
    """Figure 5: forgetting after each learned task."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.6))
    for method, matrix in accuracy_by_method.items():
        matrix = np.asarray(matrix, dtype=float)
        values = []
        for task_index in range(matrix.shape[0]):
            if task_index == 0:
                values.append(0.0)
            else:
                old = matrix[: task_index + 1, :task_index]
                final = matrix[task_index, :task_index]
                best = np.nanmax(old, axis=0)
                values.append(float(np.nanmean(best - final)))
        ax.plot(range(1, len(values) + 1), values, marker="o", markersize=3, label=method, color=_color_for(method))
    ax.set_xlabel("Task index")
    ax.set_ylabel("Average forgetting")
    ax.grid(True, color="#E6E6E6", linewidth=0.6)
    ax.legend(frameon=False)
    return _save(fig, output_dir, stem)


def plot_alpha_sensitivity(
    ablation_frame: pd.DataFrame,
    output_dir: str | Path,
    stem: str = "fig6_alpha_sensitivity",
):
    """Figure 6: alpha sensitivity for accuracy and forgetting."""
    plt = _mpl()
    fig, ax1 = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.75))
    frame = ablation_frame.sort_values("alpha")
    ax1.plot(frame["alpha"], frame["average_accuracy"], marker="o", color="#2E8B57", label="Avg. acc.")
    ax1.set_xscale("log")
    ax1.set_xlabel("$\\alpha$")
    ax1.set_ylabel("Average accuracy", color="#2E8B57")
    ax1.tick_params(axis="y", labelcolor="#2E8B57")
    ax1.grid(True, color="#E6E6E6", linewidth=0.6)

    ax2 = ax1.twinx()
    ax2.plot(frame["alpha"], frame["average_forgetting"], marker="s", color="#C43B3B", label="Forgetting")
    ax2.set_ylabel("Forgetting", color="#C43B3B")
    ax2.tick_params(axis="y", labelcolor="#C43B3B")
    return _save(fig, output_dir, stem)


def _embed_features(features: np.ndarray, random_state: int = 42) -> np.ndarray:
    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(features)
    except Exception:
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=random_state).fit_transform(features)


def plot_umap_feature_space(
    features_before,
    labels_before,
    features_after,
    labels_after,
    output_dir: str | Path,
    stem: str = "fig7_umap_feature_space",
    random_state: int = 42,
):
    """Figure 7: 2D feature projection before and after later-task training."""
    plt = _mpl()
    before = np.asarray(features_before, dtype=float)
    after = np.asarray(features_after, dtype=float)
    labels_before = np.asarray(labels_before)
    labels_after = np.asarray(labels_after)
    combined = np.vstack([before, after])
    embedding = _embed_features(combined, random_state=random_state)
    emb_before = embedding[: len(before)]
    emb_after = embedding[len(before) :]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 2.9), sharex=True, sharey=True)
    for ax, emb, labels, title in [
        (axes[0], emb_before, labels_before, "Before Task 2"),
        (axes[1], emb_after, labels_after, "After Task 2"),
    ]:
        scatter = ax.scatter(
            emb[:, 0],
            emb[:, 1],
            c=labels,
            s=7,
            alpha=0.82,
            cmap="tab20",
            linewidth=0.0,
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Projection 1")
        ax.set_ylabel("Projection 2")
    fig.colorbar(scatter, ax=axes, fraction=0.035, pad=0.02, label="Class")
    return _save(fig, output_dir, stem)
