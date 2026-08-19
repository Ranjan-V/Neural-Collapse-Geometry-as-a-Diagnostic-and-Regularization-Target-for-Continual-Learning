"""Generate extended TNNLS-style figures and tables from real experiment logs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import yaml


ROOT = Path(__file__).resolve().parents[1]
CAPSTONE_ROOT = ROOT.parents[1]

PAPER_RUNS = [
    ("cifar10", "kaggle_cifar10_full", "finetune", "Finetune"),
    ("cifar10", "kaggle_cifar10_full", "ewc", "EWC"),
    ("cifar10", "kaggle_cifar10_full", "lwf", "LwF"),
    ("cifar10", "kaggle_cifar10_full", "etf_anchor", "ETF Anchor"),
    ("cifar10", "kaggle_cifar10_lwf_etf_strong", "lwf_etf", "LwF+ETF Strong"),
    ("cifar100", "kaggle_cifar_full", "finetune", "Finetune"),
    ("cifar100", "kaggle_cifar_full", "ewc", "EWC"),
    ("cifar100", "kaggle_cifar_full", "lwf", "LwF"),
    ("cifar100", "kaggle_etf_sparse", "etf_anchor", "ETF Anchor Sparse"),
    ("cifar100", "kaggle_lwf_etf_strong", "lwf_etf", "LwF+ETF Strong"),
]

METHOD_ORDER = [
    "Finetune",
    "EWC",
    "ETF Anchor",
    "ETF Anchor Sparse",
    "LwF",
    "LwF+ETF Strong",
]

SHORT_LABELS = {
    "Finetune": "FT",
    "EWC": "EWC",
    "ETF Anchor": "ETF",
    "ETF Anchor Sparse": "ETF-S",
    "LwF": "LwF",
    "LwF+ETF Strong": "LwF+ETF",
}

COLORS = {
    "Finetune": "#4C78A8",
    "EWC": "#72B7B2",
    "ETF Anchor": "#E45756",
    "ETF Anchor Sparse": "#F58518",
    "LwF": "#54A24B",
    "LwF+ETF Strong": "#B279A2",
}

METRIC_LABELS = {
    "nc1": "NC1",
    "nc2": "NC2",
    "nc3": "NC3",
    "nc4": "NC4",
    "etf_drift": "ETF drift",
    "angle_drift": "Angle drift",
    "task1_accuracy": "Task-1 accuracy",
    "task1_forgetting": "Task-1 forgetting",
}


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _dataset_label(dataset: str) -> str:
    return dataset.upper().replace("CIFAR", "CIFAR-")


def _dataset_key(dataset: str) -> str:
    return dataset.lower().replace("cifar-", "cifar")


def _finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def _finite_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.std(ddof=1)) if values.size > 1 else float("nan")


def _nanstd_with_counts(stack: np.ndarray, axis: int = 0) -> np.ndarray:
    valid = np.isfinite(stack)
    count = valid.sum(axis=axis)
    mean = np.divide(
        np.nansum(stack, axis=axis),
        count,
        out=np.full(stack.shape[1:], np.nan, dtype=float),
        where=count > 0,
    )
    expanded_mean = np.expand_dims(mean, axis=axis)
    squared = np.where(valid, (stack - expanded_mean) ** 2, 0.0)
    variance = np.divide(
        np.sum(squared, axis=axis),
        count - 1,
        out=np.full_like(mean, np.nan, dtype=float),
        where=count > 1,
    )
    return np.sqrt(variance)


def _mean_std(values: np.ndarray, latex: bool = False) -> str:
    mean = _finite_mean(values)
    std = _finite_std(values)
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(std):
        return f"{mean:.4f}"
    sep = "$\\pm$" if latex else "+/-"
    return f"{mean:.4f} {sep} {std:.4f}"


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def _write_table(frame: pd.DataFrame, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{name}.csv", index=False)
    cols = list(frame.columns)
    aligns = "".join("l" if idx < 2 else "c" for idx in range(len(cols)))
    lines = [f"\\begin{{tabular}}{{{aligns}}}", "\\hline"]
    lines.append(" & ".join(_latex_escape(col) for col in cols) + " \\\\")
    lines.append("\\hline")
    for _, row in frame.iterrows():
        lines.append(" & ".join(_latex_escape(row[col]) for col in cols) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    (output_dir / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def _paper_mapping() -> pd.DataFrame:
    return pd.DataFrame(PAPER_RUNS, columns=["dataset", "experiment_name", "method", "paper_method"])


def _paper_seed_rows(seed_rows: pd.DataFrame) -> pd.DataFrame:
    frame = seed_rows.merge(_paper_mapping(), on=["dataset", "experiment_name", "method"], how="inner")
    frame["seed"] = frame["seed"].astype(int)
    frame["dataset_label"] = frame["dataset"].map(_dataset_label)
    frame["method_order"] = frame["paper_method"].map({method: idx for idx, method in enumerate(METHOD_ORDER)})
    return frame.sort_values(["dataset_label", "method_order", "seed"]).drop(columns=["method_order"])


def _matrix_path(row: pd.Series) -> Path:
    return (
        CAPSTONE_ROOT
        / str(row["source_folder"])
        / "logs"
        / str(row["experiment_name"])
        / str(row["method"])
        / f"seed_{int(row['seed'])}"
        / "accuracy_matrix.npy"
    )


def _nc_path(row: pd.Series) -> Path:
    return _matrix_path(row).with_name("nc_metrics.csv")


def _summary_path(row: pd.Series) -> Path:
    return _matrix_path(row).with_name("summary.json")


def _training_path(row: pd.Series) -> Path:
    return _matrix_path(row).with_name("training_log.csv")


def _load_matrices(seed_rows: pd.DataFrame) -> list[dict]:
    matrices = []
    for _, row in seed_rows.iterrows():
        path = _matrix_path(row)
        if not path.exists():
            continue
        matrices.append(
            {
                "dataset": row["dataset_label"],
                "method": row["paper_method"],
                "seed": int(row["seed"]),
                "matrix": np.load(path),
                "path": str(path),
            }
        )
    return matrices


def _load_nc_records(seed_rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for _, row in seed_rows.iterrows():
        path = _nc_path(row)
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "task_id" in frame:
            frame = frame[frame["task_id"] == 0].copy()
        if frame.empty:
            continue
        frame = frame.sort_values("step").reset_index(drop=True)
        denominator = max(1, len(frame) - 1)
        frame["progress"] = np.arange(len(frame), dtype=float) / denominator
        for _, metric_row in frame.iterrows():
            rec = {
                "dataset": row["dataset_label"],
                "method": row["paper_method"],
                "seed": int(row["seed"]),
                "progress": float(metric_row["progress"]),
                "step": float(metric_row.get("step", float("nan"))),
                "trained_task_id": float(metric_row.get("trained_task_id", float("nan"))),
            }
            for metric in [
                "nc1",
                "nc2",
                "nc3",
                "nc4",
                "etf_drift",
                "angle_drift",
                "task1_accuracy",
                "task1_forgetting",
            ]:
                rec[metric] = pd.to_numeric(pd.Series([metric_row.get(metric, np.nan)]), errors="coerce").iloc[0]
            records.append(rec)
    return pd.DataFrame(records)


def _load_training_records(seed_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[pd.DataFrame] = []
    availability = []
    for _, row in seed_rows.iterrows():
        path = _training_path(row)
        exists = path.exists()
        row_count = 0
        if exists:
            frame = pd.read_csv(path)
            row_count = len(frame)
            frame["dataset"] = row["dataset_label"]
            frame["method"] = row["paper_method"]
            frame["seed"] = int(row["seed"])
            frame["source_path"] = str(path)
            step = pd.to_numeric(frame.get("global_step", pd.Series(np.arange(len(frame)))), errors="coerce")
            step_min = float(step.min())
            step_span = max(1.0, float(step.max() - step_min))
            frame["progress"] = (step - step_min) / step_span
            records.append(frame)
        availability.append(
            {
                "dataset": row["dataset_label"],
                "method": row["paper_method"],
                "seed": int(row["seed"]),
                "training_log_available": exists,
                "rows": row_count,
                "source_path": str(path) if exists else "",
            }
        )
    data = pd.concat(records, ignore_index=True, sort=False) if records else pd.DataFrame()
    return data, pd.DataFrame(availability)


def _ordered_methods(frame: pd.DataFrame) -> list[str]:
    present = set(frame["method"])
    return [method for method in METHOD_ORDER if method in present]


def _ordered_paper_methods(frame: pd.DataFrame) -> list[str]:
    present = set(frame["paper_method"])
    return [method for method in METHOD_ORDER if method in present]


def _legend_for_methods(methods: list[str]) -> tuple[list[Line2D], list[str]]:
    ordered = [method for method in METHOD_ORDER if method in set(methods)]
    handles = [Line2D([0], [0], color=COLORS[method], lw=1.4, marker="o", markersize=3.5) for method in ordered]
    labels = [SHORT_LABELS[method] for method in ordered]
    return handles, labels


def _matrix_records_to_frames(matrices: list[dict]) -> dict[str, pd.DataFrame]:
    final_rows = []
    forgetting_rows = []
    trajectory_rows = []
    stability_rows = []

    for rec in matrices:
        mat = rec["matrix"].astype(float)
        tasks = mat.shape[0]
        diagonal = np.array([mat[t, t] for t in range(tasks)], dtype=float)
        final = mat[tasks - 1, :]
        forgetting = np.array(
            [
                np.nanmax(mat[task:, task]) - final[task]
                if np.isfinite(final[task])
                else float("nan")
                for task in range(tasks)
            ],
            dtype=float,
        )
        for task in range(tasks):
            final_rows.append({**rec, "task": task, "final_accuracy": final[task]})
            forgetting_rows.append({**rec, "task": task, "task_forgetting": forgetting[task]})
        for after_task in range(tasks):
            seen = mat[after_task, : after_task + 1]
            trajectory_rows.append(
                {
                    **rec,
                    "after_task": after_task,
                    "seen_task_average_accuracy": _finite_mean(seen),
                }
            )
        stability_rows.append(
            {
                **rec,
                "diagonal_accuracy": _finite_mean(diagonal),
                "final_average_accuracy": _finite_mean(final),
                "previous_task_final_accuracy": _finite_mean(final[:-1]),
                "average_forgetting_from_matrix": _finite_mean(forgetting[:-1]),
                "retention_ratio": _finite_mean(final) / _finite_mean(diagonal),
            }
        )

    drop_cols = ["matrix", "path"]
    return {
        "final": pd.DataFrame(final_rows).drop(columns=drop_cols, errors="ignore"),
        "forgetting": pd.DataFrame(forgetting_rows).drop(columns=drop_cols, errors="ignore"),
        "trajectory": pd.DataFrame(trajectory_rows).drop(columns=drop_cols, errors="ignore"),
        "stability": pd.DataFrame(stability_rows).drop(columns=drop_cols, errors="ignore"),
    }


def _summary_by_task(frame: pd.DataFrame, value_col: str, task_col: str) -> pd.DataFrame:
    rows = []
    for (dataset, method, task), group in frame.groupby(["dataset", "method", task_col], sort=False):
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                task_col: int(task),
                "mean": _finite_mean(group[value_col].to_numpy()),
                "std": _finite_std(group[value_col].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def plot_seed_level_performance(seed_rows: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.35), sharex=False)
    specs = [
        ("average_accuracy", "Average accuracy", (0.0, 1.02)),
        ("average_forgetting", "Average forgetting", (0.0, None)),
    ]
    for row_idx, dataset in enumerate(["CIFAR-10", "CIFAR-100"]):
        subset = seed_rows[seed_rows["dataset_label"] == dataset]
        methods = _ordered_paper_methods(subset)
        for col_idx, (metric, title, ylim) in enumerate(specs):
            ax = axes[row_idx, col_idx]
            values = []
            stds = []
            colors = []
            for method in methods:
                group = subset[subset["paper_method"] == method][metric].astype(float).to_numpy()
                values.append(_finite_mean(group))
                stds.append(_finite_std(group))
                colors.append(COLORS[method])
            x = np.arange(len(methods))
            ax.bar(x, values, yerr=stds, capsize=3, color=colors, edgecolor="#222222", linewidth=0.45, alpha=0.86)
            for idx, method in enumerate(methods):
                group = subset[subset["paper_method"] == method]
                jitter = np.linspace(-0.12, 0.12, len(group)) if len(group) > 1 else np.array([0.0])
                ax.scatter(
                    idx + jitter,
                    group[metric].astype(float),
                    s=13,
                    color="#222222",
                    alpha=0.72,
                    zorder=3,
                    linewidths=0,
                )
            ax.set_title(f"{dataset}: {title}")
            ax.set_ylabel(title)
            ax.set_xticks(x)
            ax.set_xticklabels([SHORT_LABELS[m] for m in methods], rotation=0)
            upper = ylim[1] if ylim[1] is not None else max(values + np.nan_to_num(stds, nan=0.0)) * 1.18
            ax.set_ylim(ylim[0], upper)
            ax.grid(axis="y", color="#D7D7D7", linewidth=0.55, alpha=0.8)
    fig.tight_layout()
    _save(fig, output_dir, "fig_seed_level_performance")


def plot_accuracy_forgetting_tradeoff(seed_rows: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), sharey=True)
    plotted_methods: set[str] = set()
    for ax, dataset in zip(axes, ["CIFAR-10", "CIFAR-100"]):
        subset = seed_rows[seed_rows["dataset_label"] == dataset].copy()
        for method in _ordered_paper_methods(subset):
            group = subset[subset["paper_method"] == method]
            plotted_methods.add(method)
            ax.scatter(
                group["average_forgetting"],
                group["average_accuracy"],
                s=28,
                color=COLORS[method],
                edgecolor="#222222",
                linewidth=0.3,
                alpha=0.85,
                label=SHORT_LABELS[method],
            )
        ax.set_title(dataset)
        ax.set_xlabel("Average forgetting")
        ax.set_xlim(left=-0.01)
        ax.grid(color="#D7D7D7", linewidth=0.55, alpha=0.8)
    axes[0].set_ylabel("Average accuracy")
    handles, labels = _legend_for_methods(list(plotted_methods))
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _save(fig, output_dir, "fig_accuracy_forgetting_tradeoff")


def _plot_task_profile(summary: pd.DataFrame, value_col: str, task_col: str, ylabel: str, name: str, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), sharey=True)
    plotted_methods: set[str] = set()
    for ax, dataset in zip(axes, ["CIFAR-10", "CIFAR-100"]):
        subset = summary[summary["dataset"] == dataset].copy()
        for method in _ordered_methods(subset):
            group = subset[subset["method"] == method].sort_values(task_col)
            plotted_methods.add(method)
            x = group[task_col].to_numpy(dtype=float)
            y = group["mean"].to_numpy(dtype=float)
            std = group["std"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x, y, marker="o", markersize=3, linewidth=1.05, color=COLORS[method], label=SHORT_LABELS[method])
            ax.fill_between(x, y - std, y + std, color=COLORS[method], alpha=0.12, linewidth=0)
        ax.set_title(dataset)
        ax.set_xlabel("Task index")
        ax.set_xticks(sorted(subset[task_col].dropna().unique()))
        ax.grid(color="#D7D7D7", linewidth=0.55, alpha=0.8)
    axes[0].set_ylabel(ylabel)
    handles, labels = _legend_for_methods(list(plotted_methods))
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _save(fig, output_dir, name)


def plot_seen_task_trajectory(summary: pd.DataFrame, output_dir: Path) -> None:
    _plot_task_profile(
        summary,
        "seen_task_average_accuracy",
        "after_task",
        "Mean accuracy over seen tasks",
        "fig_seen_task_average_trajectory",
        output_dir,
    )


def plot_plasticity_retention(stability: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), sharey=True)
    width = 0.36
    for ax, dataset in zip(axes, ["CIFAR-10", "CIFAR-100"]):
        subset = stability[stability["dataset"] == dataset]
        methods = _ordered_methods(subset)
        x = np.arange(len(methods))
        diag_means, diag_stds, final_means, final_stds = [], [], [], []
        for method in methods:
            group = subset[subset["method"] == method]
            diag_means.append(_finite_mean(group["diagonal_accuracy"].to_numpy()))
            diag_stds.append(_finite_std(group["diagonal_accuracy"].to_numpy()))
            final_means.append(_finite_mean(group["final_average_accuracy"].to_numpy()))
            final_stds.append(_finite_std(group["final_average_accuracy"].to_numpy()))
        ax.bar(x - width / 2, diag_means, yerr=diag_stds, width=width, capsize=2.6, color="#B8B8B8", edgecolor="#222222", linewidth=0.4, label="Diagonal")
        ax.bar(x + width / 2, final_means, yerr=final_stds, width=width, capsize=2.6, color="#4C78A8", edgecolor="#222222", linewidth=0.4, label="Final")
        ax.set_title(dataset)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS[m] for m in methods], rotation=0)
        ax.set_xlabel("Method")
        ax.grid(axis="y", color="#D7D7D7", linewidth=0.55, alpha=0.8)
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(fig, output_dir, "fig_plasticity_retention_gap")


def plot_diagnostic_bars(diag_summary: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["nc1", "nc2", "nc3", "nc4", "etf_drift", "angle_drift"]
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.25), sharex=True)
    for row_idx, dataset in enumerate(["CIFAR-10", "CIFAR-100"]):
        subset = diag_summary[diag_summary["dataset"] == dataset].set_index("diagnostic_metric")
        for col_idx, (metric_col, ylabel) in enumerate([("pearson_r_mean", "Pearson r"), ("spearman_r_mean", "Spearman rho")]):
            ax = axes[row_idx, col_idx]
            values = np.array([subset.loc[m, metric_col] if m in subset.index else np.nan for m in metrics], dtype=float)
            std_col = metric_col.replace("_mean", "_std")
            stds = np.array([subset.loc[m, std_col] if m in subset.index else np.nan for m in metrics], dtype=float)
            x = np.arange(len(metrics))
            ax.bar(x, values, yerr=stds, capsize=2.7, color="#6C8EBF", edgecolor="#222222", linewidth=0.4)
            ax.axhline(0.0, color="#222222", linewidth=0.7)
            ax.axhline(0.8, color="#222222", linestyle="--", linewidth=0.75, alpha=0.65)
            ax.axhline(-0.8, color="#222222", linestyle="--", linewidth=0.75, alpha=0.65)
            ax.set_title(f"{dataset}: {ylabel}")
            ax.set_ylabel(ylabel)
            ax.set_ylim(-1.05, 1.05)
            ax.set_xticks(x)
            ax.set_xticklabels(["NC1", "NC2", "NC3", "NC4", "ETF", "Angle"], rotation=25, ha="right")
            ax.grid(axis="y", color="#D7D7D7", linewidth=0.55, alpha=0.8)
    fig.tight_layout()
    _save(fig, output_dir, "fig_diagnostic_correlation_bars")


def _paper_diag_by_method(diag: pd.DataFrame, seed_rows: pd.DataFrame) -> pd.DataFrame:
    keys = seed_rows[["dataset", "experiment_name", "method", "seed", "paper_method", "dataset_label"]].drop_duplicates()
    frame = diag.merge(keys, on=["dataset", "experiment_name", "method", "seed"], how="inner")
    rows = []
    for (dataset, method, metric), group in frame.groupby(["dataset_label", "paper_method", "diagnostic_metric"], sort=False):
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "metric": metric,
                "pearson": _finite_mean(group["pearson_r"].to_numpy()),
                "spearman": _finite_mean(group["spearman_r"].to_numpy()),
                "n_seeds": int(group["seed"].nunique()),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["method_order"] = result["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    return result.sort_values(["dataset", "method_order", "metric"]).drop(columns=["method_order"])


def plot_method_diagnostic_heatmaps(diag_by_method: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["nc1", "nc2", "nc3", "nc4", "etf_drift", "angle_drift"]
    for dataset in ["CIFAR-10", "CIFAR-100"]:
        subset = diag_by_method[diag_by_method["dataset"] == dataset]
        if subset.empty:
            continue
        methods = _ordered_methods(subset)
        values = np.full((len(methods), len(metrics)), np.nan)
        for i, method in enumerate(methods):
            method_subset = subset[subset["method"] == method].set_index("metric")
            for j, metric in enumerate(metrics):
                if metric in method_subset.index:
                    values[i, j] = method_subset.loc[metric, "pearson"]
        fig, ax = plt.subplots(figsize=(7.15, max(1.85, 0.35 * len(methods) + 1.0)))
        image = ax.imshow(values, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        ax.set_title(f"{dataset}: diagnostic Pearson r by method")
        ax.set_yticks(np.arange(len(methods)))
        ax.set_yticklabels([SHORT_LABELS[m] for m in methods])
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels(["NC1", "NC2", "NC3", "NC4", "ETF", "Angle"], rotation=25, ha="right")
        for i in range(len(methods)):
            for j in range(len(metrics)):
                value = values[i, j]
                if np.isfinite(value):
                    color = "white" if abs(value) > 0.62 else "black"
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)
        cbar = fig.colorbar(image, ax=ax, shrink=0.85)
        cbar.set_label("Pearson r")
        fig.tight_layout()
        _save(fig, output_dir, f"fig_method_diagnostic_heatmap_{dataset.lower().replace('-', '')}")


def plot_accuracy_matrix_std(matrices: list[dict], output_dir: Path) -> None:
    for dataset in ["CIFAR-10", "CIFAR-100"]:
        methods = _ordered_methods(pd.DataFrame([m for m in matrices if m["dataset"] == dataset]))
        if not methods:
            continue
        fig, axes = plt.subplots(1, len(methods), figsize=(7.15, 1.75), squeeze=False)
        image = None
        for idx, method in enumerate(methods):
            ax = axes[0, idx]
            mats = [m["matrix"].astype(float) for m in matrices if m["dataset"] == dataset and m["method"] == method]
            if not mats:
                ax.axis("off")
                continue
            stack = np.stack(mats)
            std_mat = _nanstd_with_counts(stack, axis=0) if len(mats) > 1 else np.zeros_like(stack[0])
            image = ax.imshow(std_mat, cmap="magma", vmin=0.0, vmax=max(0.001, float(np.nanmax(std_mat))))
            ax.set_title(SHORT_LABELS[method], pad=3)
            ticks = np.arange(std_mat.shape[0])
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks])
            if idx == 0:
                ax.set_ylabel("After task")
            else:
                ax.set_yticklabels([])
            for i in range(std_mat.shape[0]):
                for j in range(std_mat.shape[1]):
                    if np.isfinite(std_mat[i, j]) and (i == j or i == std_mat.shape[0] - 1):
                        ax.text(j, i, f"{std_mat[i, j]:.2f}", ha="center", va="center", fontsize=4.7, color="white")
        if image is not None:
            cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.025)
            cbar.set_label("Std. dev.")
        fig.supxlabel("Evaluated task", y=0.01)
        fig.suptitle(f"{dataset}: accuracy-matrix seed variability", y=1.02, fontsize=9)
        _save(fig, output_dir, f"fig_accuracy_matrix_std_{dataset.lower().replace('-', '')}")


def _plot_nc_trajectories(nc: pd.DataFrame, dataset: str, metrics: list[str], name: str, output_dir: Path) -> None:
    subset = nc[nc["dataset"] == dataset].copy()
    if subset.empty:
        return
    subset["bin"] = np.minimum((subset["progress"] * 80).astype(int), 80)
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.0))
    plotted_methods: set[str] = set()
    for ax, metric in zip(axes.ravel(), metrics):
        metric_subset = subset[pd.notna(subset[metric])].copy()
        if metric_subset.empty:
            ax.axis("off")
            continue
        for method in _ordered_methods(metric_subset):
            group = metric_subset[metric_subset["method"] == method]
            if group.empty:
                continue
            plotted_methods.add(method)
            agg = group.groupby("bin")[metric].agg(["mean", "std"]).reset_index()
            x = agg["bin"].to_numpy(dtype=float) / 80.0
            y = agg["mean"].to_numpy(dtype=float)
            std = agg["std"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x, y, color=COLORS[method], linewidth=1.05, label=SHORT_LABELS[method])
            ax.fill_between(x, y - std, y + std, color=COLORS[method], alpha=0.12, linewidth=0)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("Training progress")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(color="#D7D7D7", linewidth=0.55, alpha=0.78)
    methods = [method for method in METHOD_ORDER if method in plotted_methods]
    handles = [Line2D([0], [0], color=COLORS[method], lw=1.3) for method in methods]
    labels = [SHORT_LABELS[method] for method in methods]
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle(f"{dataset}: task-1 metric trajectories", y=1.02, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, output_dir, name)


def plot_nc_trajectory_figures(nc: pd.DataFrame, output_dir: Path) -> None:
    for dataset in ["CIFAR-10", "CIFAR-100"]:
        suffix = dataset.lower().replace("-", "")
        _plot_nc_trajectories(nc, dataset, ["nc1", "nc2", "nc3", "nc4"], f"fig_nc1_nc4_trajectories_{suffix}", output_dir)
        _plot_nc_trajectories(
            nc,
            dataset,
            ["etf_drift", "angle_drift", "task1_accuracy", "task1_forgetting"],
            f"fig_drift_forgetting_trajectories_{suffix}",
            output_dir,
        )


def plot_pairwise_nc_correlations(nc: pd.DataFrame, output_dir: Path) -> None:
    cols = ["task1_forgetting", "nc1", "nc2", "nc3", "nc4", "etf_drift", "angle_drift"]
    labels = ["Forget", "NC1", "NC2", "NC3", "NC4", "ETF", "Angle"]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65))
    for ax, dataset in zip(axes, ["CIFAR-10", "CIFAR-100"]):
        subset = nc[nc["dataset"] == dataset][cols].copy()
        corr = subset.corr(method="pearson", min_periods=10).to_numpy()
        image = ax.imshow(corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax.set_title(dataset)
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(cols)))
        ax.set_yticklabels(labels)
        for i in range(len(cols)):
            for j in range(len(cols)):
                value = corr[i, j]
                if np.isfinite(value):
                    color = "white" if abs(value) > 0.62 else "black"
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.8, color=color)
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.03)
    cbar.set_label("Pearson r")
    fig.suptitle("Pairwise NC/forgetting correlations", y=1.03, fontsize=9)
    _save(fig, output_dir, "fig_pairwise_nc_forgetting_correlations")


def plot_training_dynamics(training: pd.DataFrame, output_dir: Path) -> None:
    if training.empty:
        return
    subset = training[(training["dataset"] == "CIFAR-100") & (training["seed"] == 42)].copy()
    if subset.empty:
        return
    subset["progress"] = pd.to_numeric(subset["progress"], errors="coerce")
    subset = subset[np.isfinite(subset["progress"])].copy()
    if subset.empty:
        return
    subset["bin"] = np.minimum((subset["progress"] * 120).astype(int), 120)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55))
    specs = [("ce_loss", "Cross-entropy loss"), ("batch_accuracy", "Batch accuracy")]
    for ax, (metric, label) in zip(axes, specs):
        metric_subset = subset[pd.notna(subset.get(metric, np.nan))].copy()
        for method in _ordered_methods(metric_subset):
            group = metric_subset[metric_subset["method"] == method]
            if group.empty:
                continue
            agg = group.groupby("bin")[metric].mean().reset_index()
            ax.plot(
                agg["bin"].to_numpy(dtype=float) / 120.0,
                agg[metric].to_numpy(dtype=float),
                color=COLORS[method],
                linewidth=1.05,
                label=SHORT_LABELS[method],
            )
        for boundary in [0.2, 0.4, 0.6, 0.8]:
            ax.axvline(boundary, color="#999999", linewidth=0.5, linestyle=":", alpha=0.75)
        ax.set_title(f"CIFAR-100 seed 42: {label}")
        ax.set_xlabel("Training progress")
        ax.set_ylabel(label)
        ax.grid(color="#D7D7D7", linewidth=0.55, alpha=0.8)
    handles, labels = _legend_for_methods(_ordered_methods(subset))
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _save(fig, output_dir, "fig_training_dynamics_cifar100_seed42")


def plot_etf_loss_components(training: pd.DataFrame, output_dir: Path) -> None:
    if training.empty or "etf_loss" not in training:
        return
    subset = training[(training["dataset"] == "CIFAR-100") & (training["seed"] == 42)].copy()
    subset = subset[pd.to_numeric(subset.get("etf_loss", np.nan), errors="coerce").fillna(0.0) > 0.0]
    if subset.empty:
        return
    subset["progress"] = pd.to_numeric(subset["progress"], errors="coerce")
    subset = subset[np.isfinite(subset["progress"])].copy()
    if subset.empty:
        return
    subset["bin"] = np.minimum((subset["progress"] * 120).astype(int), 120)
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.45), sharex=True)
    specs = [
        ("etf_loss", "log10(1 + ETF loss)", True),
        ("centroid_loss", "log10(1 + centroid loss)", True),
        ("angle_loss", "Angle loss", False),
    ]
    for ax, (metric, label, log_transform) in zip(axes, specs):
        if metric not in subset:
            ax.axis("off")
            continue
        for method in _ordered_methods(subset):
            group = subset[subset["method"] == method].copy()
            if group.empty or metric not in group:
                continue
            group[metric] = pd.to_numeric(group[metric], errors="coerce")
            agg = group.groupby("bin")[metric].mean().reset_index()
            y = agg[metric].to_numpy(dtype=float)
            if log_transform:
                y = np.log10(1.0 + np.clip(y, a_min=0.0, a_max=None))
            ax.plot(agg["bin"].to_numpy(dtype=float) / 120.0, y, color=COLORS[method], linewidth=1.05, label=SHORT_LABELS[method])
        for boundary in [0.2, 0.4, 0.6, 0.8]:
            ax.axvline(boundary, color="#999999", linewidth=0.5, linestyle=":", alpha=0.75)
        ax.set_title(label)
        ax.set_xlabel("Training progress")
        ax.set_ylabel(label)
        ax.grid(color="#D7D7D7", linewidth=0.55, alpha=0.8)
    handles, labels = _legend_for_methods(_ordered_methods(subset))
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)), frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("CIFAR-100 seed 42: ETF loss components", y=1.02, fontsize=9)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    _save(fig, output_dir, "fig_etf_loss_components_cifar100_seed42")


def plot_etf_corr_seed_distribution(seed_rows: pd.DataFrame, output_dir: Path) -> None:
    subset = seed_rows[pd.notna(seed_rows["etf_forgetting_corr"])].copy()
    if subset.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.45), sharey=True)
    for ax, dataset in zip(axes, ["CIFAR-10", "CIFAR-100"]):
        ds = subset[subset["dataset_label"] == dataset]
        methods = _ordered_paper_methods(ds)
        x = np.arange(len(methods))
        for idx, method in enumerate(methods):
            vals = ds[ds["paper_method"] == method]["etf_forgetting_corr"].astype(float).to_numpy()
            ax.bar(idx, _finite_mean(vals), yerr=_finite_std(vals), capsize=3, color=COLORS[method], edgecolor="#222222", linewidth=0.45)
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(idx + jitter, vals, s=16, color="#222222", alpha=0.72, linewidths=0)
        ax.axhline(0.8, color="#222222", linestyle="--", linewidth=0.75, alpha=0.65)
        ax.axhline(0.0, color="#222222", linewidth=0.7)
        ax.set_title(dataset)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS[m] for m in methods])
        ax.grid(axis="y", color="#D7D7D7", linewidth=0.55, alpha=0.8)
    axes[0].set_ylabel("ETF drift / forgetting Pearson r")
    fig.tight_layout()
    _save(fig, output_dir, "fig_etf_correlation_seed_distribution")


def plot_method_rank_summary(paper: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for dataset, group in paper.groupby("dataset", sort=False):
        acc_rank = group["avg_acc_mean"].rank(ascending=False, method="min")
        forget_rank = group["forgetting_mean"].rank(ascending=True, method="min")
        for idx, row in group.iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "method": row["method"],
                    "accuracy_rank": float(acc_rank.loc[idx]),
                    "forgetting_rank": float(forget_rank.loc[idx]),
                }
            )
    ranks = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.45), sharey=True)
    for ax, dataset in zip(axes, ["CIFAR-10", "CIFAR-100"]):
        subset = ranks[ranks["dataset"] == dataset]
        methods = _ordered_methods(subset)
        x = np.arange(len(methods))
        acc = [subset[subset["method"] == m]["accuracy_rank"].iloc[0] for m in methods]
        forg = [subset[subset["method"] == m]["forgetting_rank"].iloc[0] for m in methods]
        ax.plot(x, acc, marker="o", color="#4C78A8", linewidth=1.1, label="Accuracy rank")
        ax.plot(x, forg, marker="s", color="#E45756", linewidth=1.1, label="Forgetting rank")
        ax.invert_yaxis()
        ax.set_title(dataset)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS[m] for m in methods])
        ax.set_xlabel("Method")
        ax.grid(axis="y", color="#D7D7D7", linewidth=0.55, alpha=0.8)
    axes[0].set_ylabel("Rank (1 is best)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(fig, output_dir, "fig_method_rank_summary")


def _wide_task_table(summary: pd.DataFrame, value_col: str, task_col: str, table_name: str, output_dir: Path) -> pd.DataFrame:
    rows = []
    for (dataset, method), group in summary.groupby(["dataset", "method"], sort=False):
        row = {"dataset": dataset, "method": method}
        for task, task_group in group.groupby(task_col):
            values = task_group[value_col].to_numpy(dtype=float) if value_col in task_group else task_group["mean"].to_numpy(dtype=float)
            if "mean" in task_group and "std" in task_group:
                row[f"task_{int(task)}"] = _mean_std_pair(task_group["mean"].iloc[0], task_group["std"].iloc[0])
            else:
                row[f"task_{int(task)}"] = _mean_std(values)
        rows.append(row)
    frame = pd.DataFrame(rows)
    _write_table(frame, output_dir, table_name)
    return frame


def _mean_std_pair(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def _generate_tables(
    paper: pd.DataFrame,
    seed_rows: pd.DataFrame,
    diag_summary: pd.DataFrame,
    diag_by_method: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    training_availability: pd.DataFrame,
    output_dir: Path,
) -> None:
    seed_table = seed_rows[
        [
            "dataset_label",
            "paper_method",
            "seed",
            "average_accuracy",
            "average_forgetting",
            "backward_transfer",
            "etf_forgetting_corr",
        ]
    ].rename(columns={"dataset_label": "dataset", "paper_method": "method"})
    _write_table(seed_table.round(4), output_dir, "table_seed_level_results")

    final_summary = _summary_by_task(frames["final"], "final_accuracy", "task")
    forgetting_summary = _summary_by_task(frames["forgetting"], "task_forgetting", "task")
    trajectory_summary = _summary_by_task(frames["trajectory"], "seen_task_average_accuracy", "after_task")
    _wide_task_table(final_summary, "final_accuracy", "task", "table_final_task_accuracy_by_task", output_dir)
    _wide_task_table(forgetting_summary, "task_forgetting", "task", "table_task_forgetting_by_task", output_dir)
    _wide_task_table(trajectory_summary, "seen_task_average_accuracy", "after_task", "table_seen_task_accuracy_trajectory", output_dir)

    stability_rows = []
    for (dataset, method), group in frames["stability"].groupby(["dataset", "method"], sort=False):
        stability_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "diagonal_accuracy": _mean_std(group["diagonal_accuracy"].to_numpy()),
                "final_average_accuracy": _mean_std(group["final_average_accuracy"].to_numpy()),
                "previous_task_final_accuracy": _mean_std(group["previous_task_final_accuracy"].to_numpy()),
                "retention_ratio": _mean_std(group["retention_ratio"].to_numpy()),
            }
        )
    _write_table(pd.DataFrame(stability_rows), output_dir, "table_plasticity_retention_summary")

    delta_rows = []
    for dataset, group in paper.groupby("dataset", sort=False):
        lwf = group[group["method"] == "LwF"].iloc[0]
        for _, row in group.iterrows():
            delta_rows.append(
                {
                    "dataset": dataset,
                    "method": row["method"],
                    "delta_avg_acc_vs_lwf": round(float(row["avg_acc_mean"] - lwf["avg_acc_mean"]), 4),
                    "delta_forgetting_vs_lwf": round(float(row["forgetting_mean"] - lwf["forgetting_mean"]), 4),
                    "delta_bwt_vs_lwf": round(float(row["bwt_mean"] - lwf["bwt_mean"]), 4),
                }
            )
    _write_table(pd.DataFrame(delta_rows), output_dir, "table_delta_vs_lwf")

    diag_table = diag_summary.copy()
    for col in diag_table.select_dtypes(include=[np.number]).columns:
        diag_table[col] = diag_table[col].round(4)
    _write_table(diag_table, output_dir, "table_nc_diagnostic_summary")

    method_diag_table = diag_by_method.copy()
    for col in ["pearson", "spearman"]:
        method_diag_table[col] = method_diag_table[col].round(4)
    _write_table(method_diag_table, output_dir, "table_nc_diagnostic_by_method")

    etf_table = seed_rows[pd.notna(seed_rows["etf_forgetting_corr"])][
        ["dataset_label", "paper_method", "seed", "etf_forgetting_corr"]
    ].rename(columns={"dataset_label": "dataset", "paper_method": "method"})
    _write_table(etf_table.round(4), output_dir, "table_etf_correlation_per_seed")

    completion = pd.read_csv(output_dir.parent / "paper_experiment_completion_status.csv")
    _write_table(completion, output_dir, "table_experiment_completion")
    _write_table(training_availability, output_dir, "table_training_log_availability")

    _write_protocol_tables(output_dir)
    _write_table_inventory(output_dir)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_protocol_tables(output_dir: Path) -> None:
    configs = {
        "CIFAR-10 full": ROOT / "config" / "kaggle_cifar10_full.yaml",
        "CIFAR-10 LwF+ETF": ROOT / "config" / "kaggle_cifar10_lwf_etf_strong.yaml",
        "CIFAR-100 full": ROOT / "config" / "kaggle_cifar_full.yaml",
        "CIFAR-100 ETF sparse": ROOT / "config" / "kaggle_etf_sparse.yaml",
        "CIFAR-100 LwF+ETF": ROOT / "config" / "kaggle_lwf_etf_strong.yaml",
    }
    protocol_rows = []
    hyper_rows = []
    for label, path in configs.items():
        cfg = _load_yaml(path)
        protocol_rows.append(
            {
                "config": label,
                "dataset": cfg.get("dataset", "--"),
                "num_tasks": cfg.get("num_tasks", "--"),
                "classes_per_task": cfg.get("classes_per_task", "--"),
                "epochs_per_task": cfg.get("epochs_per_task", "--"),
                "batch_size": cfg.get("batch_size", "--"),
                "architecture": cfg.get("architecture", "--"),
                "feature_dim": cfg.get("feature_dim", "--"),
            }
        )
        hyper_rows.append(
            {
                "config": label,
                "learning_rate": cfg.get("learning_rate", "--"),
                "weight_decay": cfg.get("weight_decay", "--"),
                "ewc_lambda": cfg.get("ewc_lambda", "--"),
                "lwf_alpha": cfg.get("lwf_alpha", "--"),
                "lwf_temperature": cfg.get("lwf_temperature", "--"),
                "etf_alpha": cfg.get("alpha", "--"),
                "lambda_angle": cfg.get("lambda_angle", "--"),
                "anchor_every_n_steps": cfg.get("anchor_every_n_steps", "--"),
            }
        )
    _write_table(pd.DataFrame(protocol_rows), output_dir, "table_experiment_protocol")
    _write_table(pd.DataFrame(hyper_rows), output_dir, "table_key_hyperparameters")


def _write_table_inventory(output_dir: Path) -> None:
    text = """# TNNLS Table Inventory

## Main-paper candidates
- Table I: Experiment protocol.
- Table II: Main 3-seed accuracy/forgetting/BWT results.
- Table III: NC diagnostic correlation summary.
- Table IV: Final task-wise accuracy.
- Table V: Delta versus LwF baseline.
- Table VI: Plasticity-retention summary.

## Appendix candidates
- Per-seed result table.
- Per-task forgetting table.
- Seen-task accuracy trajectory table.
- NC diagnostic by method table.
- ETF correlation per seed table.
- Key hyperparameter table.
- Completion/provenance table.
- Training-log availability table.

This gives 6 realistic main-paper tables and 8 appendix tables without fabricating any values.
"""
    (output_dir / "table_inventory.md").write_text(text, encoding="utf-8")


def _write_asset_inventory(analysis_dir: Path) -> None:
    figures_paper = analysis_dir / "figures_paper"
    figures_tnnls = analysis_dir / "figures_tnnls"
    tables_tnnls = analysis_dir / "tables_tnnls"
    paper_pdfs = sorted(figures_paper.glob("*.pdf"))
    tnnls_pdfs = sorted(figures_tnnls.glob("*.pdf"))
    csv_tables = sorted(tables_tnnls.glob("*.csv"))
    tex_tables = sorted(tables_tnnls.glob("*.tex"))

    lines = [
        "# TNNLS Asset Inventory",
        "",
        "## Counts",
        f"- Core paper figures: {len(paper_pdfs)} PDF figures, with PNG companions.",
        f"- Extended TNNLS figures: {len(tnnls_pdfs)} PDF figures, with PNG companions.",
        f"- Extended tables: {len(csv_tables)} CSV tables and {len(tex_tables)} LaTeX tables.",
        "",
        "## Recommended Main-Paper Figures",
        "- `fig_main_accuracy_forgetting.pdf`: main method comparison.",
        "- `fig_cifar100_core_diagnostics.pdf`: strongest diagnostic evidence.",
        "- `fig_nc_diagnostic_heatmap.pdf`: compact diagnostic summary.",
        "- `fig_diagnostic_correlation_bars.pdf`: Pearson/Spearman diagnostic comparison.",
        "- `fig_accuracy_matrices_cifar10.pdf` and `fig_accuracy_matrices_cifar100.pdf`: continual-learning accuracy matrices.",
        "- `fig_etf_loss_components_cifar100_seed42.pdf`: ETF failure-mode/regularizer behavior evidence.",
        "",
        "## Recommended Appendix Figures",
        "- Seed-level performance.",
        "- Accuracy/forgetting tradeoff.",
        "- Method rank summary.",
        "- Task-wise final accuracy and forgetting profiles.",
        "- Seen-task average accuracy trajectories.",
        "- Accuracy-matrix standard-deviation heatmaps.",
        "- NC1-NC4 trajectories for CIFAR-10 and CIFAR-100.",
        "- Drift/forgetting trajectories for CIFAR-10 and CIFAR-100.",
        "- Pairwise NC/forgetting correlation heatmap.",
        "- Method-specific diagnostic heatmaps.",
        "- Training dynamics for CIFAR-100 seed 42.",
        "",
        "## Main-Paper Table Budget",
        "- Realistic main paper: 4 to 6 tables.",
        "- Strong TNNLS-style package: 6 main tables plus 8 appendix tables.",
        "- Do not use every generated table in the main text; keep overflow for appendix/supplement.",
        "",
        "## Generated Core Figures",
    ]
    lines.extend(f"- `{path.name}`" for path in paper_pdfs)
    lines.extend(["", "## Generated Extended Figures"])
    lines.extend(f"- `{path.name}`" for path in tnnls_pdfs)
    lines.extend(["", "## Generated Extended Tables"])
    lines.extend(f"- `{path.name}`" for path in csv_tables)
    (analysis_dir / "tnnls_asset_inventory.md").write_text("\n".join(lines), encoding="utf-8")


def _make_contact_sheet(paths: list[Path], output_path: Path, title: str, columns: int = 3) -> None:
    if not paths:
        return
    thumb_w, thumb_h = 520, 300
    pad = 24
    title_h = 70
    label_h = 58
    rows = math.ceil(len(paths) / columns)
    sheet_w = columns * thumb_w + (columns + 1) * pad
    sheet_h = title_h + rows * (thumb_h + label_h + pad) + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        title_font = ImageFont.truetype("arial.ttf", 30)
        label_font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    draw.text((pad, 20), title, fill=(20, 20, 20), font=title_font)
    for idx, path in enumerate(paths):
        row = idx // columns
        col = idx % columns
        x = pad + col * (thumb_w + pad)
        y = title_h + row * (thumb_h + label_h + pad)
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            box = Image.new("RGB", (thumb_w, thumb_h), "white")
            ox = (thumb_w - image.width) // 2
            oy = (thumb_h - image.height) // 2
            box.paste(image, (ox, oy))
        draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline=(210, 210, 210), width=1)
        sheet.paste(box, (x, y))
        label = path.stem
        if len(label) > 54:
            label = label[:51] + "..."
        draw.text((x, y + thumb_h + 8), label, fill=(20, 20, 20), font=label_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    pdf_path = output_path.with_suffix(".pdf")
    sheet.save(pdf_path, "PDF", resolution=180.0)


def _write_manifests_and_contact_sheets(analysis_dir: Path) -> None:
    figures_paper = analysis_dir / "figures_paper"
    figures_tnnls = analysis_dir / "figures_tnnls"
    tables_tnnls = analysis_dir / "tables_tnnls"

    figure_rows = []
    for group, directory in [("core", figures_paper), ("extended", figures_tnnls)]:
        for png in sorted(directory.glob("*.png")):
            pdf = png.with_suffix(".pdf")
            figure_rows.append(
                {
                    "group": group,
                    "figure": png.stem,
                    "png_path": str(png.relative_to(analysis_dir)),
                    "pdf_path": str(pdf.relative_to(analysis_dir)) if pdf.exists() else "",
                    "png_bytes": png.stat().st_size,
                    "pdf_bytes": pdf.stat().st_size if pdf.exists() else "",
                }
            )
    pd.DataFrame(figure_rows).to_csv(analysis_dir / "figure_manifest.csv", index=False)

    table_rows = []
    for csv_path in sorted(tables_tnnls.glob("*.csv")):
        tex_path = csv_path.with_suffix(".tex")
        table_rows.append(
            {
                "table": csv_path.stem,
                "csv_path": str(csv_path.relative_to(analysis_dir)),
                "tex_path": str(tex_path.relative_to(analysis_dir)) if tex_path.exists() else "",
                "csv_bytes": csv_path.stat().st_size,
                "tex_bytes": tex_path.stat().st_size if tex_path.exists() else "",
            }
        )
    pd.DataFrame(table_rows).to_csv(analysis_dir / "table_manifest.csv", index=False)

    contact_dir = analysis_dir / "contact_sheets"
    core_pngs = sorted(figures_paper.glob("*.png"))
    extended_pngs = sorted(figures_tnnls.glob("*.png"))
    all_pngs = core_pngs + extended_pngs
    _make_contact_sheet(core_pngs, contact_dir / "core_figures_contact_sheet.png", "Core Paper Figures", columns=2)
    _make_contact_sheet(extended_pngs, contact_dir / "extended_figures_contact_sheet.png", "Extended TNNLS Figures", columns=3)
    _make_contact_sheet(all_pngs, contact_dir / "all_figures_contact_sheet.png", "All Generated Figures", columns=3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "results" / "analysis" / "final_all")
    args = parser.parse_args()

    _configure_matplotlib()
    analysis_dir = args.analysis_dir
    figures_dir = analysis_dir / "figures_tnnls"
    tables_dir = analysis_dir / "tables_tnnls"

    paper = pd.read_csv(analysis_dir / "paper_main_results_3seed.csv")
    raw_seed_rows = pd.read_csv(analysis_dir / "seed_level_results.csv")
    diag_summary = pd.read_csv(analysis_dir / "nc_diagnostic_summary_by_dataset_metric.csv")
    diag = pd.read_csv(analysis_dir / "nc_diagnostic_correlations.csv")

    seed_rows = _paper_seed_rows(raw_seed_rows)
    matrices = _load_matrices(seed_rows)
    matrix_frames = _matrix_records_to_frames(matrices)
    nc = _load_nc_records(seed_rows)
    training, training_availability = _load_training_records(seed_rows)
    diag_by_method = _paper_diag_by_method(diag, seed_rows)

    plot_seed_level_performance(seed_rows, figures_dir)
    plot_accuracy_forgetting_tradeoff(seed_rows, figures_dir)
    plot_method_rank_summary(paper, figures_dir)
    plot_etf_corr_seed_distribution(seed_rows, figures_dir)
    plot_diagnostic_bars(diag_summary, figures_dir)
    plot_method_diagnostic_heatmaps(diag_by_method, figures_dir)
    plot_accuracy_matrix_std(matrices, figures_dir)
    plot_nc_trajectory_figures(nc, figures_dir)
    plot_pairwise_nc_correlations(nc, figures_dir)
    plot_training_dynamics(training, figures_dir)
    plot_etf_loss_components(training, figures_dir)

    final_summary = _summary_by_task(matrix_frames["final"], "final_accuracy", "task")
    forgetting_summary = _summary_by_task(matrix_frames["forgetting"], "task_forgetting", "task")
    trajectory_summary = _summary_by_task(matrix_frames["trajectory"], "seen_task_average_accuracy", "after_task")
    _plot_task_profile(final_summary, "final_accuracy", "task", "Final accuracy", "fig_final_task_accuracy_profiles", figures_dir)
    _plot_task_profile(forgetting_summary, "task_forgetting", "task", "Task forgetting", "fig_task_forgetting_profiles", figures_dir)
    plot_seen_task_trajectory(trajectory_summary, figures_dir)
    plot_plasticity_retention(matrix_frames["stability"], figures_dir)

    _generate_tables(paper, seed_rows, diag_summary, diag_by_method, matrix_frames, training_availability, tables_dir)
    _write_asset_inventory(analysis_dir)
    _write_manifests_and_contact_sheets(analysis_dir)

    print(f"Figures: {figures_dir}")
    print(f"Tables: {tables_dir}")


if __name__ == "__main__":
    main()
