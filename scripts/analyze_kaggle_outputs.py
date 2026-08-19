"""Merge Kaggle output folders and compute paper diagnostics.

Example:
    python scripts/analyze_kaggle_outputs.py ^
      --roots "D:\\Capstone\\Capstone Outputs 1" "D:\\Capstone\\Capstone Outputs 2" ^
              "D:\\Capstone\\Capstone Output 3" ^
      --output-dir results/analysis/final
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ["average_accuracy", "average_forgetting", "backward_transfer", "etf_forgetting_corr"]
DIAGNOSTIC_COLUMNS = ["nc1", "nc2", "nc3", "nc4", "etf_drift", "angle_drift"]
REQUIRED_PAPER_SEEDS = (42, 43, 44)

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

PAPER_METHOD_ORDER = [
    "Finetune",
    "EWC",
    "LwF",
    "ETF Anchor",
    "ETF Anchor Sparse",
    "LwF+ETF Strong",
]


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x_rank = pd.Series(x[mask]).rank(method="average").to_numpy()
    y_rank = pd.Series(y[mask]).rank(method="average").to_numpy()
    return _pearson(x_rank, y_rank)


def _read_result_rows(output_roots: list[Path], include_smoke: bool) -> pd.DataFrame:
    rows: list[dict] = []
    for root in output_roots:
        for csv_path in root.glob("**/logs/aggregate/all_results.csv"):
            frame = pd.read_csv(csv_path)
            frame["source_folder"] = root.name
            rows.extend(frame.to_dict(orient="records"))
        direct_csv = root / "logs" / "aggregate" / "all_results.csv"
        if direct_csv.exists():
            frame = pd.read_csv(direct_csv)
            frame["source_folder"] = root.name
            rows.extend(frame.to_dict(orient="records"))

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if not include_smoke and "dataset" in frame:
        frame = frame[frame["dataset"].fillna("") != "synthetic"].copy()
    for metric in METRICS:
        if metric in frame:
            frame[metric] = frame[metric].map(_safe_float)

    dedup_cols = ["dataset", "experiment_name", "method", "seed"]
    existing = [col for col in dedup_cols if col in frame]
    return frame.drop_duplicates(subset=existing, keep="last").sort_values(existing)


def _aggregate_results(seed_rows: pd.DataFrame) -> pd.DataFrame:
    if seed_rows.empty:
        return pd.DataFrame()
    group_cols = ["dataset", "experiment_name", "method"]
    available_metrics = [metric for metric in METRICS if metric in seed_rows]
    grouped = seed_rows.groupby(group_cols, dropna=False)[available_metrics].agg(["mean", "std", "count"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def _format_mean_std(mean: float, std: float) -> str:
    if math.isnan(mean):
        return "--"
    if math.isnan(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def _latex_table(aggregate: pd.DataFrame) -> str:
    if aggregate.empty:
        return "% No aggregate results found.\n"
    lines = [
        "\\begin{tabular}{lllcccc}",
        "\\hline",
        "Dataset & Experiment & Method & Avg. Acc. & Forgetting & BWT & ETF--F Corr. \\\\",
        "\\hline",
    ]
    for _, row in aggregate.iterrows():
        cells = [
            str(row["dataset"]),
            str(row["experiment_name"]),
            str(row["method"]),
        ]
        for metric in METRICS:
            cells.append(
                _format_mean_std(
                    float(row.get(f"{metric}_mean", float("nan"))),
                    float(row.get(f"{metric}_std", float("nan"))),
                )
            )
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


def _paper_mapping() -> pd.DataFrame:
    return pd.DataFrame(
        PAPER_RUNS,
        columns=["dataset", "experiment_name", "method", "paper_method"],
    )


def _paper_seed_rows(seed_rows: pd.DataFrame) -> pd.DataFrame:
    if seed_rows.empty:
        return pd.DataFrame()
    mapping = _paper_mapping()
    frame = seed_rows.merge(mapping, on=["dataset", "experiment_name", "method"], how="inner")
    frame = frame[frame["seed"].astype(int).isin(REQUIRED_PAPER_SEEDS)].copy()
    return frame.sort_values(["dataset", "paper_method", "seed"])


def _paper_completion_status(seed_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, experiment, method, paper_method in PAPER_RUNS:
        subset = seed_rows[
            (seed_rows["dataset"] == dataset)
            & (seed_rows["experiment_name"] == experiment)
            & (seed_rows["method"] == method)
        ]
        present = sorted(int(seed) for seed in subset["seed"].dropna().unique())
        missing = [seed for seed in REQUIRED_PAPER_SEEDS if seed not in present]
        rows.append(
            {
                "dataset": dataset,
                "experiment_name": experiment,
                "method": method,
                "paper_method": paper_method,
                "present_seeds": " ".join(str(seed) for seed in present),
                "missing_seeds": " ".join(str(seed) for seed in missing),
                "complete": not missing,
            }
        )
    return pd.DataFrame(rows)


def _paper_aggregate(seed_rows: pd.DataFrame) -> pd.DataFrame:
    paper_rows = _paper_seed_rows(seed_rows)
    if paper_rows.empty:
        return pd.DataFrame()
    rows = []
    for (dataset, paper_method), group in paper_rows.groupby(["dataset", "paper_method"], sort=False):
        if set(group["seed"].astype(int)) != set(REQUIRED_PAPER_SEEDS):
            continue
        row = {
            "dataset": dataset.upper().replace("CIFAR", "CIFAR-"),
            "method": paper_method,
        }
        for source, target in [
            ("average_accuracy", "avg_acc"),
            ("average_forgetting", "forgetting"),
            ("backward_transfer", "bwt"),
            ("etf_forgetting_corr", "etf_corr"),
        ]:
            values = group[source].map(_safe_float).to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{target}_mean"] = float(finite.mean()) if finite.size else float("nan")
            row[f"{target}_std"] = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["method_order"] = frame["method"].map({name: idx for idx, name in enumerate(PAPER_METHOD_ORDER)})
    frame["dataset_order"] = frame["dataset"].map({"CIFAR-10": 0, "CIFAR-100": 1})
    return frame.sort_values(["dataset_order", "method_order"]).drop(columns=["dataset_order", "method_order"])


def _paper_latex_table(paper: pd.DataFrame) -> str:
    if paper.empty:
        return "% No complete 3-seed paper results found.\n"
    lines = [
        "\\begin{tabular}{llcccc}",
        "\\hline",
        "Dataset & Method & Avg. Acc. & Forgetting & BWT & ETF--F Corr. \\\\",
        "\\hline",
    ]
    for _, row in paper.iterrows():
        cells = [str(row["dataset"]), str(row["method"])]
        for metric in ["avg_acc", "forgetting", "bwt", "etf_corr"]:
            cells.append(_format_mean_std(float(row[f"{metric}_mean"]), float(row[f"{metric}_std"])))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


def _iter_nc_metric_files(output_roots: list[Path]):
    for root in output_roots:
        for path in root.glob("**/logs/*/*/seed_*/nc_metrics.csv"):
            parts = path.parts
            try:
                logs_index = parts.index("logs")
            except ValueError:
                continue
            if logs_index + 4 >= len(parts):
                continue
            experiment = parts[logs_index + 1]
            method = parts[logs_index + 2]
            seed_text = parts[logs_index + 3]
            if not seed_text.startswith("seed_"):
                continue
            yield root.name, experiment, method, int(seed_text.removeprefix("seed_")), path


def _nc_diagnostics(output_roots: list[Path], seed_rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    dataset_lookup = {}
    if not seed_rows.empty:
        for _, row in seed_rows.iterrows():
            dataset_lookup[(row["experiment_name"], row["method"], int(row["seed"]))] = row.get("dataset", "")

    for source, experiment, method, seed, path in _iter_nc_metric_files(output_roots):
        frame = pd.read_csv(path)
        if "task1_forgetting" not in frame:
            continue
        if "task_id" in frame:
            task0 = frame[frame["task_id"] == 0].copy()
        else:
            task0 = frame.copy()
        if task0.empty:
            continue
        y = task0["task1_forgetting"].map(_safe_float).to_numpy(dtype=float)
        for metric in DIAGNOSTIC_COLUMNS:
            if metric not in task0:
                continue
            x = task0[metric].map(_safe_float).to_numpy(dtype=float)
            records.append(
                {
                    "source_folder": source,
                    "dataset": dataset_lookup.get((experiment, method, seed), ""),
                    "experiment_name": experiment,
                    "method": method,
                    "seed": seed,
                    "diagnostic_metric": metric,
                    "n_points": int((np.isfinite(x) & np.isfinite(y)).sum()),
                    "pearson_r": _pearson(x, y),
                    "spearman_r": _spearman(x, y),
                }
            )
    return pd.DataFrame(records)


def _paper_diagnostic_summary(diagnostics: pd.DataFrame, seed_rows: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    paper_keys = _paper_seed_rows(seed_rows)[["dataset", "experiment_name", "method", "seed"]].drop_duplicates()
    filtered = diagnostics.merge(paper_keys, on=["dataset", "experiment_name", "method", "seed"], how="inner")
    if filtered.empty:
        return pd.DataFrame()
    grouped = filtered.groupby(["dataset", "diagnostic_metric"], dropna=False)[["pearson_r", "spearman_r"]].agg(
        ["mean", "std", "count"]
    )
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    frame = grouped.reset_index()
    frame["dataset"] = frame["dataset"].str.upper().str.replace("CIFAR", "CIFAR-", regex=False)
    return frame.sort_values(["dataset", "diagnostic_metric"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "analysis")
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = [path.resolve() for path in args.roots]

    seed_rows = _read_result_rows(roots, include_smoke=args.include_smoke)
    aggregate = _aggregate_results(seed_rows)
    diagnostics = _nc_diagnostics(roots, seed_rows)
    completion = _paper_completion_status(seed_rows)
    paper = _paper_aggregate(seed_rows)
    paper_diag = _paper_diagnostic_summary(diagnostics, seed_rows)

    seed_rows_path = output_dir / "seed_level_results.csv"
    aggregate_path = output_dir / "aggregate_mean_std_results.csv"
    diagnostics_path = output_dir / "nc_diagnostic_correlations.csv"
    table_path = output_dir / "aggregate_mean_std_table.tex"
    completion_path = output_dir / "paper_experiment_completion_status.csv"
    paper_path = output_dir / "paper_main_results_3seed.csv"
    paper_table_path = output_dir / "paper_main_results_3seed.tex"
    paper_diag_path = output_dir / "nc_diagnostic_summary_by_dataset_metric.csv"

    seed_rows.to_csv(seed_rows_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    table_path.write_text(_latex_table(aggregate), encoding="utf-8")
    completion.to_csv(completion_path, index=False)
    paper.to_csv(paper_path, index=False)
    paper_table_path.write_text(_paper_latex_table(paper), encoding="utf-8")
    paper_diag.to_csv(paper_diag_path, index=False)

    print(f"Seed rows: {seed_rows_path}")
    print(f"Aggregate: {aggregate_path}")
    print(f"Diagnostics: {diagnostics_path}")
    print(f"LaTeX table: {table_path}")
    print(f"Paper completion: {completion_path}")
    print(f"Paper main table: {paper_path}")
    print(f"Paper diagnostics: {paper_diag_path}")
    if not aggregate.empty:
        print("\n=== Aggregate Preview ===")
        preview_cols = ["dataset", "experiment_name", "method", "average_accuracy_mean", "average_forgetting_mean"]
        print(aggregate[preview_cols].to_string(index=False))
    if not diagnostics.empty:
        print("\n=== Diagnostic Preview ===")
        print(
            diagnostics.sort_values("pearson_r", ascending=False)
            .head(12)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
