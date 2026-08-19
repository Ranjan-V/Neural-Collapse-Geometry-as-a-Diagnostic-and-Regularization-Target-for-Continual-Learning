"""Run the full Geometric Memory experiment suite.

Example:
    python experiments/run_all.py --config config/config.yaml --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluator import ResultsLogger, aggregate_method_results
from training.continual_trainer import ContinualTrainer
from utils import load_config
from visualization.plots import (
    plot_accuracy_heatmaps,
    plot_average_accuracy_comparison,
    plot_drift_vs_forgetting,
    plot_forgetting_curves,
    plot_nc_metrics,
)


DEFAULT_METHODS = ["finetune", "ewc", "lwf", "etf_anchor"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run continual-learning experiments.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--datasets", nargs="+", default=None, help="Defaults to the dataset in config.")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    datasets = args.datasets or [base_config["dataset"]]
    results_logger = ResultsLogger(Path(base_config.get("log_dir", "results/logs")) / "aggregate")
    accuracy_by_method: dict[str, np.ndarray] = {}
    nc_frames: list[pd.DataFrame] = []

    for dataset_name in datasets:
        for seed in args.seeds:
            for method_name in args.methods:
                config = copy.deepcopy(base_config)
                config["dataset"] = dataset_name
                config["seed"] = int(seed)
                print(f"[run_all] dataset={dataset_name} seed={seed} method={method_name}")
                trainer = ContinualTrainer(config)
                summary = trainer.run_experiment(method_name)
                accuracy = np.asarray(summary["accuracy_matrix"], dtype=float)
                key = method_name if len(datasets) == 1 else f"{dataset_name}:{method_name}"
                accuracy_by_method[key] = accuracy
                results_logger.add_run(
                    method=key,
                    seed=seed,
                    accuracy_matrix=accuracy,
                    nc_metrics_path=summary.get("nc_metrics_path"),
                    metadata={
                        "dataset": dataset_name,
                        "experiment_name": config.get("experiment_name", ""),
                    },
                )
                nc_path = Path(summary["nc_metrics_path"])
                if nc_path.exists():
                    frame = pd.read_csv(nc_path)
                    frame["method"] = key
                    frame["seed"] = seed
                    frame["dataset"] = dataset_name
                    nc_frames.append(frame)

    saved = results_logger.save()
    print(f"[run_all] saved aggregate files: {saved}")

    if not args.skip_plots:
        figure_dir = Path(base_config.get("log_dir", "results/logs")).parents[0] / "figures"
        if accuracy_by_method:
            plot_accuracy_heatmaps(accuracy_by_method, figure_dir)
            plot_forgetting_curves(accuracy_by_method, figure_dir)
        summary_frame = aggregate_method_results(results_logger.rows)
        if not summary_frame.empty:
            plot_average_accuracy_comparison(summary_frame, figure_dir)
        if nc_frames:
            nc_frame = pd.concat(nc_frames, ignore_index=True)
            plot_nc_metrics(nc_frame, figure_dir)
            if "etf_drift" in nc_frame:
                plot_drift_vs_forgetting(nc_frame[nc_frame["task_id"] == 0], accuracy_by_method, figure_dir)


if __name__ == "__main__":
    main()
