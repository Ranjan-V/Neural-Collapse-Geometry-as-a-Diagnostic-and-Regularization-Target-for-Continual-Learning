"""Generate paper-facing figures and analysis notes from merged results."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CAPSTONE_ROOT = ROOT.parents[1]

METHOD_ORDER = [
    "Finetune",
    "EWC",
    "ETF Anchor",
    "ETF Anchor Sparse",
    "LwF",
    "LwF+ETF Strong",
]

COLORS = {
    "Finetune": "#4C78A8",
    "EWC": "#72B7B2",
    "ETF Anchor": "#E45756",
    "ETF Anchor Sparse": "#F58518",
    "LwF": "#54A24B",
    "LwF+ETF Strong": "#B279A2",
}

SHORT_LABELS = {
    "Finetune": "FT",
    "EWC": "EWC",
    "ETF Anchor": "ETF",
    "ETF Anchor Sparse": "ETF-S",
    "LwF": "LwF",
    "LwF+ETF Strong": "LwF+ETF",
}

PAPER_RUNS = {
    ("CIFAR-10", "Finetune"): ("kaggle_cifar10_full", "finetune"),
    ("CIFAR-10", "EWC"): ("kaggle_cifar10_full", "ewc"),
    ("CIFAR-10", "LwF"): ("kaggle_cifar10_full", "lwf"),
    ("CIFAR-10", "ETF Anchor"): ("kaggle_cifar10_full", "etf_anchor"),
    ("CIFAR-10", "LwF+ETF Strong"): ("kaggle_cifar10_lwf_etf_strong", "lwf_etf"),
    ("CIFAR-100", "Finetune"): ("kaggle_cifar_full", "finetune"),
    ("CIFAR-100", "EWC"): ("kaggle_cifar_full", "ewc"),
    ("CIFAR-100", "LwF"): ("kaggle_cifar_full", "lwf"),
    ("CIFAR-100", "ETF Anchor Sparse"): ("kaggle_etf_sparse", "etf_anchor"),
    ("CIFAR-100", "LwF+ETF Strong"): ("kaggle_lwf_etf_strong", "lwf_etf"),
}


def _source_method_label(method: str, experiment_name: str = "") -> str:
    if method == "finetune":
        return "Finetune"
    if method == "ewc":
        return "EWC"
    if method == "lwf":
        return "LwF"
    if method == "lwf_etf":
        return "LwF+ETF Strong"
    if method == "etf_anchor" and experiment_name == "kaggle_etf_sparse":
        return "ETF Anchor Sparse"
    if method == "etf_anchor":
        return "ETF Anchor"
    return method


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _has_variance(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size <= 2:
        return False
    span = float(np.nanmax(finite) - np.nanmin(finite))
    scale = max(1.0, float(np.nanmax(np.abs(finite))))
    return span > 1e-8 * scale


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _ordered_methods(frame: pd.DataFrame) -> list[str]:
    present = set(frame["method"])
    return [method for method in METHOD_ORDER if method in present]


def _filter_seed_rows_to_paper(seed_rows: pd.DataFrame, paper: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in paper.iterrows():
        run = PAPER_RUNS.get((row["dataset"], row["method"]))
        if run is None:
            continue
        experiment, source_method = run
        dataset_key = row["dataset"].lower().replace("cifar-", "cifar")
        subset = seed_rows[
            (seed_rows["dataset"] == dataset_key)
            & (seed_rows["experiment_name"] == experiment)
            & (seed_rows["method"] == source_method)
        ].copy()
        if not subset.empty:
            rows.append(subset)
    if not rows:
        return seed_rows.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True)


def plot_main_results(paper: pd.DataFrame, output_dir: Path) -> None:
    datasets = list(dict.fromkeys(paper["dataset"]))
    fig, axes = plt.subplots(len(datasets), 2, figsize=(7.15, 4.25), sharex=False)
    if len(datasets) == 1:
        axes = np.array([axes])

    for row_idx, dataset in enumerate(datasets):
        subset = paper[paper["dataset"] == dataset].copy()
        methods = _ordered_methods(subset)
        subset = subset.set_index("method").loc[methods].reset_index()
        x = np.arange(len(subset))
        colors = [COLORS.get(method, "#999999") for method in subset["method"]]

        specs = [
            ("avg_acc_mean", "avg_acc_std", "Average Accuracy", "Accuracy", (0.0, 1.02)),
            ("forgetting_mean", "forgetting_std", "Average Forgetting", "Forgetting", (0.0, None)),
        ]
        for col_idx, (mean_col, std_col, title, ylabel, ylim) in enumerate(specs):
            ax = axes[row_idx, col_idx]
            means = subset[mean_col].astype(float).to_numpy()
            stds = subset[std_col].astype(float).to_numpy()
            ax.bar(x, means, yerr=stds, capsize=3.0, color=colors, edgecolor="#222222", linewidth=0.45)
            for xi, value in zip(x, means):
                ax.text(xi, value + 0.016, f"{value:.3f}", ha="center", va="bottom", fontsize=6.5)
            ax.set_title(f"{dataset}: {title}")
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels([SHORT_LABELS.get(m, m) for m in subset["method"]], rotation=0)
            upper = ylim[1] if ylim[1] is not None else max(0.08, float(np.nanmax(means + stds)) * 1.18)
            ax.set_ylim(ylim[0], upper)
            ax.grid(axis="y", color="#D7D7D7", linewidth=0.6, alpha=0.85)

    fig.tight_layout()
    _save(fig, output_dir, "fig_main_accuracy_forgetting")


def plot_bwt_and_corr(paper: pd.DataFrame, output_dir: Path) -> None:
    datasets = list(dict.fromkeys(paper["dataset"]))
    fig, axes = plt.subplots(len(datasets), 2, figsize=(7.15, 4.25), sharex=False)
    if len(datasets) == 1:
        axes = np.array([axes])

    for row_idx, dataset in enumerate(datasets):
        subset = paper[paper["dataset"] == dataset].copy()
        methods = _ordered_methods(subset)
        subset = subset.set_index("method").loc[methods].reset_index()
        x = np.arange(len(subset))
        colors = [COLORS.get(method, "#999999") for method in subset["method"]]

        ax = axes[row_idx, 0]
        means = subset["bwt_mean"].astype(float).to_numpy()
        stds = subset["bwt_std"].astype(float).to_numpy()
        ax.bar(x, means, yerr=stds, capsize=3.0, color=colors, edgecolor="#222222", linewidth=0.45)
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_title(f"{dataset}: Backward Transfer")
        ax.set_ylabel("BWT")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS.get(m, m) for m in subset["method"]], rotation=0)
        ax.grid(axis="y", color="#D7D7D7", linewidth=0.6, alpha=0.85)

        ax = axes[row_idx, 1]
        corr_subset = subset.dropna(subset=["etf_corr_mean"])
        if corr_subset.empty:
            ax.text(0.5, 0.5, "No ETF drift metric", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
        else:
            cx = np.arange(len(corr_subset))
            cmeans = corr_subset["etf_corr_mean"].astype(float).to_numpy()
            cstds = corr_subset["etf_corr_std"].astype(float).to_numpy()
            ccolors = [COLORS.get(method, "#999999") for method in corr_subset["method"]]
            ax.bar(cx, cmeans, yerr=cstds, capsize=3.0, color=ccolors, edgecolor="#222222", linewidth=0.45)
            for xi, value in zip(cx, cmeans):
                ax.text(xi, value + 0.035, f"{value:.3f}", ha="center", va="bottom", fontsize=6.5)
            ax.set_xticks(cx)
            ax.set_xticklabels([SHORT_LABELS.get(m, m) for m in corr_subset["method"]], rotation=0)
        ax.axhline(0.8, color="#333333", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_ylim(-0.15, 1.05)
        ax.set_title(f"{dataset}: ETF Drift / Forgetting Correlation")
        ax.set_ylabel("Pearson r")
        ax.grid(axis="y", color="#D7D7D7", linewidth=0.6, alpha=0.85)

    fig.tight_layout()
    _save(fig, output_dir, "fig_bwt_and_etf_correlation")


def plot_nc_diagnostic_heatmap(diag_summary: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["nc1", "nc2", "nc3", "nc4", "etf_drift", "angle_drift"]
    datasets = list(dict.fromkeys(diag_summary["dataset"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.15, 2.25), squeeze=False)
    axes = axes[0]
    cmap = plt.get_cmap("RdBu_r")

    for ax, dataset in zip(axes, datasets):
        subset = diag_summary[diag_summary["dataset"] == dataset].set_index("diagnostic_metric")
        values = [subset.loc[m, "pearson_r_mean"] if m in subset.index else np.nan for m in metrics]
        image = ax.imshow([values], cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")
        ax.set_title(dataset.upper().replace("CIFAR", "CIFAR-"))
        ax.set_yticks([])
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels(["NC1", "NC2", "NC3", "NC4", "ETF", "Angle"], rotation=35, ha="right")
        for idx, value in enumerate(values):
            if np.isfinite(value):
                color = "white" if abs(value) > 0.62 else "black"
                ax.text(idx, 0, f"{value:.2f}", ha="center", va="center", fontsize=7.5, color=color)

    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.72, pad=0.03)
    cbar.set_label("Mean Pearson r with task-1 forgetting")
    fig.suptitle("NC Diagnostic Strength Across Datasets", y=1.02, fontsize=11)
    _save(fig, output_dir, "fig_nc_diagnostic_heatmap")


def _matrix_path(row: pd.Series) -> Path:
    seed_dir = f"seed_{int(row['seed'])}"
    return (
        CAPSTONE_ROOT
        / str(row["source_folder"])
        / "logs"
        / str(row["experiment_name"])
        / str(row["method"])
        / seed_dir
        / "accuracy_matrix.npy"
    )


def plot_accuracy_heatmaps(seed_rows: pd.DataFrame, paper: pd.DataFrame, output_dir: Path) -> None:
    for dataset in sorted(paper["dataset"].unique()):
        subset_paper = paper[paper["dataset"] == dataset].copy()
        methods = _ordered_methods(subset_paper)
        fig, axes = plt.subplots(1, len(methods), figsize=(7.15, 1.72), squeeze=False)
        image = None
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("white")

        for idx, label in enumerate(methods):
            ax = axes[0, idx]
            # Match paper label to source experiment/method rows.
            candidates = seed_rows[seed_rows["dataset"].str.upper().str.replace("CIFAR", "CIFAR-") == dataset]
            if label == "ETF Anchor":
                candidates = candidates[
                    (candidates["experiment_name"] == "kaggle_cifar10_full")
                    & (candidates["method"] == "etf_anchor")
                ]
            elif label == "ETF Anchor Sparse":
                candidates = candidates[
                    (candidates["experiment_name"] == "kaggle_etf_sparse")
                    & (candidates["method"] == "etf_anchor")
                ]
            elif label == "LwF+ETF Strong":
                experiment = "kaggle_cifar10_lwf_etf_strong" if dataset == "CIFAR-10" else "kaggle_lwf_etf_strong"
                candidates = candidates[
                    (candidates["experiment_name"] == experiment)
                    & (candidates["method"] == "lwf_etf")
                ]
            else:
                candidates = candidates[candidates["method"].str.lower() == label.lower()]

            mats = []
            for _, seed_row in candidates.iterrows():
                path = _matrix_path(seed_row)
                if path.exists():
                    mats.append(np.load(path))
            if not mats:
                ax.axis("off")
                ax.set_title(f"{label}\nmissing matrix")
                continue
            stacked = np.stack(mats)
            valid_counts = np.isfinite(stacked).sum(axis=0)
            mean_matrix = np.divide(
                np.nansum(stacked, axis=0),
                valid_counts,
                out=np.full(stacked.shape[1:], np.nan, dtype=float),
                where=valid_counts > 0,
            )
            masked = np.ma.masked_invalid(mean_matrix)
            image = ax.imshow(masked, vmin=0.0, vmax=1.0, cmap=cmap)
            ax.set_title(SHORT_LABELS.get(label, label), pad=3)
            ticks = np.arange(mean_matrix.shape[0])
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks])
            if idx == 0:
                ax.set_ylabel("After task")
            else:
                ax.set_yticklabels([])
            for i in range(mean_matrix.shape[0]):
                for j in range(mean_matrix.shape[1]):
                    value = mean_matrix[i, j]
                    if np.isfinite(value) and (i == j or i == mean_matrix.shape[0] - 1):
                        ax.text(
                            j,
                            i,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            fontsize=4.7,
                            color="white" if value < 0.55 else "black",
                        )

        if image is not None:
            cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.025)
            cbar.set_label("Accuracy")
        fig.supxlabel("Evaluated task", y=0.01)
        fig.suptitle(f"{dataset}: mean accuracy matrices", y=1.02, fontsize=9)
        _save(fig, output_dir, f"fig_accuracy_matrices_{dataset.lower().replace('-', '')}")


def plot_nc_scatter(seed_rows: pd.DataFrame, output_dir: Path) -> None:
    data = _load_nc_forgetting_records(seed_rows)
    if data.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.55))
    specs = [
        ("CIFAR-10", "nc3", "NC3"),
        ("CIFAR-10", "nc4", "NC4"),
        ("CIFAR-10", "angle_drift", "Angle Drift"),
        ("CIFAR-100", "nc3", "NC3"),
        ("CIFAR-100", "nc4", "NC4"),
        ("CIFAR-100", "angle_drift", "Angle Drift"),
    ]
    for ax, (dataset, metric, label) in zip(axes.ravel(), specs):
        subset = data[(data["dataset"] == dataset) & pd.notna(data[metric])].copy()
        if subset.empty:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{dataset}: {label}")
            continue
        sample = subset.sample(n=min(len(subset), 1200), random_state=7) if len(subset) > 1200 else subset
        for method, group in sample.groupby("method_label"):
            color = COLORS.get(method, "#777777")
            ax.scatter(group[metric], group["task1_forgetting"], s=7, alpha=0.32, linewidths=0, label=method)
        x = pd.to_numeric(subset[metric], errors="coerce").to_numpy()
        y = pd.to_numeric(subset["task1_forgetting"], errors="coerce").to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() > 2 and _has_variance(x[mask]):
            coeff = np.polyfit(x[mask], y[mask], deg=1)
            xs = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 80)
            ax.plot(xs, coeff[0] * xs + coeff[1], color="#222222", linewidth=1.0)
            r = np.corrcoef(x[mask], y[mask])[0, 1]
            ax.text(0.04, 0.92, f"r={r:.2f}", transform=ax.transAxes, fontsize=8)
        elif mask.sum() > 2:
            ax.text(0.04, 0.92, "near-constant", transform=ax.transAxes, fontsize=7)
        ax.set_title(f"{dataset}: {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("Task-1 forgetting")
        ax.grid(color="#D7D7D7", linewidth=0.5, alpha=0.75)
    fig.tight_layout()
    _save(fig, output_dir, "fig_nc_forgetting_scatter")


def _load_nc_forgetting_records(seed_rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in seed_rows.iterrows():
        nc_path = (
            CAPSTONE_ROOT
            / str(row["source_folder"])
            / "logs"
            / str(row["experiment_name"])
            / str(row["method"])
            / f"seed_{int(row['seed'])}"
            / "nc_metrics.csv"
        )
        if not nc_path.exists():
            continue
        frame = pd.read_csv(nc_path)
        if "task1_forgetting" not in frame:
            continue
        frame = frame[frame.get("task_id", 0) == 0].copy() if "task_id" in frame else frame
        for _, metric_row in frame.iterrows():
            if pd.notna(metric_row.get("task1_forgetting")):
                records.append(
                    {
                        "dataset": row["dataset"].upper().replace("CIFAR", "CIFAR-"),
                        "method": row["method"],
                        "method_label": _source_method_label(row["method"], row["experiment_name"]),
                        "experiment_name": row["experiment_name"],
                        "seed": int(row["seed"]),
                        "task1_forgetting": metric_row.get("task1_forgetting"),
                        "nc3": metric_row.get("nc3"),
                        "nc4": metric_row.get("nc4"),
                        "angle_drift": metric_row.get("angle_drift"),
                    }
                )
    return pd.DataFrame(records)


def plot_cifar100_core_diagnostics(seed_rows: pd.DataFrame, output_dir: Path) -> None:
    data = _load_nc_forgetting_records(seed_rows)
    data = data[data["dataset"] == "CIFAR-100"].copy()
    if data.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.2), sharey=True)
    specs = [
        ("nc3", "NC3", "Weight-mean mismatch"),
        ("nc4", "NC4", "Softmax/NCC agreement"),
        ("angle_drift", "Angle drift", "ETF angular distortion"),
    ]

    for ax, (metric, title, xlabel) in zip(axes, specs):
        subset = data[pd.notna(data[metric])].copy()
        if subset.empty:
            ax.axis("off")
            continue
        sample = subset.sample(n=min(len(subset), 1500), random_state=11) if len(subset) > 1500 else subset
        for label, group in sample.groupby("method_label"):
            ax.scatter(
                group[metric],
                group["task1_forgetting"],
                s=7,
                alpha=0.25,
                linewidths=0,
                color=COLORS.get(label, "#777777"),
                label=SHORT_LABELS.get(label, label),
            )

        x = pd.to_numeric(subset[metric], errors="coerce").to_numpy()
        y = pd.to_numeric(subset["task1_forgetting"], errors="coerce").to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() > 2 and _has_variance(x[mask]):
            coeff = np.polyfit(x[mask], y[mask], deg=1)
            xs = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
            ax.plot(xs, coeff[0] * xs + coeff[1], color="#222222", linewidth=1.05)
            r = np.corrcoef(x[mask], y[mask])[0, 1]
            ax.text(0.04, 0.91, f"r={r:.2f}", transform=ax.transAxes, fontsize=8)
        ax.set_title(f"CIFAR-100: {title}")
        ax.set_xlabel(xlabel)
        ax.grid(color="#D7D7D7", linewidth=0.5, alpha=0.75)

    axes[0].set_ylabel("Task-1 forgetting")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False, bbox_to_anchor=(0.5, -0.045))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, output_dir, "fig_cifar100_core_diagnostics")


def write_analysis_report(paper: pd.DataFrame, diag_summary: pd.DataFrame, output_dir: Path) -> None:
    report = output_dir.parent / "paper_analysis_report.md"
    lines = ["# Paper Analysis Report", ""]
    lines.append("## Experiment Coverage")
    lines.append("- Complete 3-seed results are available for CIFAR-10 and CIFAR-100.")
    lines.append("- Seeds: 42, 43, and 44 for all paper-table methods.")
    lines.append("- Methods: Finetune, EWC, LwF, ETF Anchor/ETF Anchor Sparse, and LwF+ETF Strong.")
    lines.append("")
    lines.append("## Main Empirical Read")
    for dataset in paper["dataset"].unique():
        subset = paper[paper["dataset"] == dataset]
        best_acc = subset.loc[subset["avg_acc_mean"].idxmax()]
        best_forget = subset.loc[subset["forgetting_mean"].idxmin()]
        lines.append(
            f"- {dataset}: best average accuracy is {best_acc['method']} "
            f"({best_acc['avg_acc_mean']:.4f} +/- {best_acc['avg_acc_std']:.4f}); "
            f"lowest forgetting is {best_forget['method']} "
            f"({best_forget['forgetting_mean']:.4f} +/- {best_forget['forgetting_std']:.4f})."
        )
    lines.append("")
    lines.append("## Main Results Table")
    lines.append("| Dataset | Method | Avg. Acc. | Forgetting | BWT | ETF-F Corr. |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, row in paper.iterrows():
        corr = (
            "--"
            if not np.isfinite(row["etf_corr_mean"])
            else f"{row['etf_corr_mean']:.4f} +/- {row['etf_corr_std']:.4f}"
        )
        lines.append(
            f"| {row['dataset']} | {row['method']} | "
            f"{row['avg_acc_mean']:.4f} +/- {row['avg_acc_std']:.4f} | "
            f"{row['forgetting_mean']:.4f} +/- {row['forgetting_std']:.4f} | "
            f"{row['bwt_mean']:.4f} +/- {row['bwt_std']:.4f} | {corr} |"
        )
    lines += [
        "",
        "## Diagnostic Read",
        "- CIFAR-100 shows the strongest neural-collapse diagnostic signal: NC3 and NC4 are strongly associated with task-1 forgetting.",
        "- CIFAR-100 angle drift reaches the target diagnostic threshold: mean Pearson r is above 0.8 and mean Spearman r is above 0.9.",
        "- CIFAR-10 does not show the same ETF/angle-drift signal; NC3 is the most useful CIFAR-10 diagnostic but remains moderate.",
        "- LwF remains the strongest method by accuracy and forgetting on both datasets.",
        "- ETF anchoring alone and LwF+ETF do not beat LwF in these runs, so the safest claim is diagnostic-first rather than method-SOTA.",
        "",
        "## Recommended Paper Claim",
        "Neural collapse geometry is a useful diagnostic lens for continual-learning forgetting, especially on CIFAR-100. ETF anchoring is a plausible auxiliary regularization target, but the current implementation should be presented as exploratory rather than a replacement for LwF.",
        "",
        "## What Not To Claim Yet",
        "- Do not claim that ETF Anchor beats LwF; the 3-seed results do not support that.",
        "- Do not claim universal ETF-drift correlation across datasets; CIFAR-10 is weak.",
        "- Do not frame this as a TNNLS-ready method paper without additional method improvement or broader datasets.",
        "",
        "## Generated Figures",
        "- `fig_main_accuracy_forgetting.pdf`: main accuracy and forgetting comparison.",
        "- `fig_bwt_and_etf_correlation.pdf`: backward transfer and ETF-drift correlation.",
        "- `fig_nc_diagnostic_heatmap.pdf`: diagnostic correlation heatmap.",
        "- `fig_accuracy_matrices_cifar10.pdf` and `fig_accuracy_matrices_cifar100.pdf`: mean task accuracy matrices.",
        "- `fig_nc_forgetting_scatter.pdf`: NC metric scatter plots against task-1 forgetting.",
        "- `fig_cifar100_core_diagnostics.pdf`: focused CIFAR-100 diagnostic evidence for NC3, NC4, and angle drift.",
        "",
        "## Suggested Venue Direction",
        "- Strongest current direction: a workshop or conference submission framed around diagnostics and representation geometry.",
        "- For a higher-impact journal/conference target, add one more dataset and improve ETF regularization so it is competitive with LwF or clearly complementary.",
        "",
    ]
    lines.append("## NC Diagnostic Summary")
    for _, row in diag_summary.iterrows():
        lines.append(
            f"- {row['dataset']} {row['diagnostic_metric']}: "
            f"Pearson {row['pearson_r_mean']:.3f} +/- {row['pearson_r_std']:.3f}, "
            f"Spearman {row['spearman_r_mean']:.3f} +/- {row['spearman_r_std']:.3f}."
        )
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "results" / "analysis" / "final_all")
    args = parser.parse_args()

    _configure_matplotlib()
    analysis_dir = args.analysis_dir
    output_dir = analysis_dir / "figures_paper"
    paper = pd.read_csv(analysis_dir / "paper_main_results_3seed.csv")
    seed_rows = pd.read_csv(analysis_dir / "seed_level_results.csv")
    diag_summary = pd.read_csv(analysis_dir / "nc_diagnostic_summary_by_dataset_metric.csv")
    paper_seed_rows = _filter_seed_rows_to_paper(seed_rows, paper)

    plot_main_results(paper, output_dir)
    plot_bwt_and_corr(paper, output_dir)
    plot_nc_diagnostic_heatmap(diag_summary, output_dir)
    plot_accuracy_heatmaps(paper_seed_rows, paper, output_dir)
    plot_nc_scatter(paper_seed_rows, output_dir)
    plot_cifar100_core_diagnostics(paper_seed_rows, output_dir)
    write_analysis_report(paper, diag_summary, output_dir)

    print(f"Figures: {output_dir}")
    print(f"Report: {analysis_dir / 'paper_analysis_report.md'}")


if __name__ == "__main__":
    main()
