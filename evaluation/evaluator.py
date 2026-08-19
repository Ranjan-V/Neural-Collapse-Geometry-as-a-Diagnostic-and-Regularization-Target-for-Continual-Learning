"""Evaluation metrics, summaries, and paper-ready tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _as_array(accuracy_matrix) -> np.ndarray:
    matrix = np.asarray(accuracy_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("accuracy_matrix must be 2D.")
    return matrix


def compute_accuracy_matrix(model, all_test_loaders, trainer) -> np.ndarray:
    """Evaluate a model on all task test loaders using the supplied trainer."""
    num_tasks = len(all_test_loaders)
    matrix = np.full((num_tasks, num_tasks), np.nan, dtype=float)
    for task_id, loader in enumerate(all_test_loaders):
        metrics = trainer.evaluate(model, loader, task_id)
        matrix[-1, task_id] = metrics["top1"]
    return matrix


def compute_average_accuracy(accuracy_matrix) -> float:
    """Mean final accuracy across all learned tasks."""
    matrix = _as_array(accuracy_matrix)
    final_row = matrix[-1]
    return float(np.nanmean(final_row))


def compute_forgetting(accuracy_matrix) -> dict[str, Any]:
    """Compute per-task and average forgetting from an accuracy matrix."""
    matrix = _as_array(accuracy_matrix)
    num_tasks = matrix.shape[0]
    forgetting: list[float] = []
    for task_id in range(num_tasks - 1):
        history = matrix[: num_tasks - 1, task_id]
        if np.all(np.isnan(history)) or np.isnan(matrix[-1, task_id]):
            forgetting.append(float("nan"))
            continue
        best_previous = np.nanmax(history)
        forgetting.append(float(best_previous - matrix[-1, task_id]))
    return {
        "per_task": forgetting,
        "average_forgetting": float(np.nanmean(forgetting)) if forgetting else 0.0,
    }


def compute_backward_transfer(accuracy_matrix) -> float:
    """Backward transfer: final old-task accuracy minus accuracy after learning that task."""
    matrix = _as_array(accuracy_matrix)
    num_tasks = matrix.shape[0]
    values: list[float] = []
    for task_id in range(num_tasks - 1):
        immediate = matrix[task_id, task_id]
        final = matrix[-1, task_id]
        if not np.isnan(immediate) and not np.isnan(final):
            values.append(float(final - immediate))
    return float(np.nanmean(values)) if values else 0.0


def compute_etf_forgetting_correlation(etf_drift_log, accuracy_matrix) -> float:
    """Pearson r between ETF drift and task-1 accuracy drop over checkpoints."""
    if isinstance(etf_drift_log, pd.DataFrame):
        drift_values = etf_drift_log.get("etf_drift", pd.Series(dtype=float)).to_numpy(dtype=float)
        if "task1_forgetting" in etf_drift_log:
            forgetting_values = etf_drift_log["task1_forgetting"].to_numpy(dtype=float)
        else:
            forgetting_values = None
    else:
        drift_values = np.asarray(etf_drift_log, dtype=float)
        forgetting_values = None

    if forgetting_values is None:
        matrix = _as_array(accuracy_matrix)
        task1_accuracy = matrix[:, 0]
        if np.all(np.isnan(task1_accuracy)):
            return float("nan")
        baseline = task1_accuracy[0]
        forgetting = baseline - task1_accuracy
    else:
        forgetting = forgetting_values

    length = min(len(drift_values), len(forgetting))
    x = drift_values[:length]
    y = forgetting[:length]
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    if np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def summarize_accuracy_matrix(accuracy_matrix) -> dict[str, float]:
    """Return the main scalar metrics for a completed run."""
    forgetting = compute_forgetting(accuracy_matrix)
    return {
        "average_accuracy": compute_average_accuracy(accuracy_matrix),
        "average_forgetting": float(forgetting["average_forgetting"]),
        "backward_transfer": compute_backward_transfer(accuracy_matrix),
    }


def aggregate_method_results(result_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate seed-level results into mean/std rows by method."""
    if not result_rows:
        return pd.DataFrame()
    frame = pd.DataFrame(result_rows)
    metrics = ["average_accuracy", "average_forgetting", "backward_transfer", "etf_forgetting_corr"]
    existing = [metric for metric in metrics if metric in frame.columns]
    grouped = frame.groupby("method")[existing].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def _format_mean_std(mean: float, std: float) -> str:
    if np.isnan(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def generate_results_table(all_method_results) -> str:
    """Generate a LaTeX table string ready for the paper."""
    if isinstance(all_method_results, pd.DataFrame):
        aggregated = all_method_results
    else:
        aggregated = aggregate_method_results(all_method_results)
    if aggregated.empty:
        return "% No results available yet.\n"

    metric_specs = [
        ("average_accuracy", "Avg. Acc. $\\uparrow$"),
        ("average_forgetting", "Forgetting $\\downarrow$"),
        ("backward_transfer", "BWT $\\uparrow$"),
        ("etf_forgetting_corr", "$r(\\Delta_{ETF},F)$ $\\uparrow$"),
    ]
    columns = ["Method"] + [label for metric, label in metric_specs if f"{metric}_mean" in aggregated]
    lines = [
        "\\begin{tabular}{l" + "c" * (len(columns) - 1) + "}",
        "\\hline",
        " & ".join(columns) + " \\\\",
        "\\hline",
    ]
    for _, row in aggregated.iterrows():
        cells = [str(row["method"])]
        for metric, _label in metric_specs:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            if mean_key in row:
                cells.append(_format_mean_std(float(row[mean_key]), float(row.get(std_key, np.nan))))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


class ResultsLogger:
    """Persist run summaries and generate CSV/LaTeX artifacts."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        existing_csv = self.output_dir / "all_results.csv"
        if existing_csv.exists():
            existing = pd.read_csv(existing_csv)
            self.rows = existing.replace({np.nan: None}).to_dict(orient="records")

    def add_run(
        self,
        method: str,
        seed: int,
        accuracy_matrix,
        nc_metrics_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        summary = summarize_accuracy_matrix(accuracy_matrix)
        summary.update({"method": method, "seed": int(seed)})
        if metadata:
            summary.update(metadata)
        if nc_metrics_path is not None and Path(nc_metrics_path).exists():
            nc_frame = pd.read_csv(nc_metrics_path)
            task0 = nc_frame[nc_frame["task_id"] == 0] if "task_id" in nc_frame else nc_frame
            summary["etf_forgetting_corr"] = compute_etf_forgetting_correlation(task0, accuracy_matrix)
        else:
            summary["etf_forgetting_corr"] = float("nan")
        self.rows = [
            row
            for row in self.rows
            if not (
                row.get("method") == method
                and int(row.get("seed", -1)) == int(seed)
                and (row.get("experiment_name") in {None, summary.get("experiment_name")})
                and (row.get("dataset") in {None, summary.get("dataset")})
            )
        ]
        self.rows.append(summary)
        return summary

    def save(self) -> dict[str, str]:
        json_path = self.output_dir / "all_results.json"
        csv_path = self.output_dir / "all_results.csv"
        table_path = self.output_dir / "main_table.tex"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(self.rows, handle, indent=2)
        frame = pd.DataFrame(self.rows)
        frame.to_csv(csv_path, index=False)
        table = generate_results_table(self.rows)
        table_path.write_text(table, encoding="utf-8")
        return {
            "json": str(json_path),
            "csv": str(csv_path),
            "table": str(table_path),
        }
