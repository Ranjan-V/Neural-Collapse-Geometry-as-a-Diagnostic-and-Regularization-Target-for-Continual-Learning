"""Run ETF Anchor ablations.

Examples:
    python experiments/run_ablation.py --vary alpha --values 0.1 0.5 1.0 5.0 10.0
    python experiments/run_ablation.py --vary lambda_angle --values 0.0 0.1 0.5 1.0 2.0
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

from evaluation.evaluator import ResultsLogger
from training.continual_trainer import ContinualTrainer
from utils import load_config
from visualization.plots import plot_alpha_sensitivity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ETF Anchor sensitivity studies.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--vary", required=True, choices=["alpha", "lambda_angle"])
    parser.add_argument("--values", nargs="+", type=float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    output_dir = Path(base_config.get("log_dir", "results/logs")) / "ablation" / args.vary
    logger = ResultsLogger(output_dir)
    rows = []

    for value in args.values:
        for seed in args.seeds:
            config = copy.deepcopy(base_config)
            config[args.vary] = float(value)
            config["seed"] = int(seed)
            variant = f"{args.vary}_{value:g}"
            config["experiment_name"] = f"{base_config.get('experiment_name', 'geometric_memory')}_{variant}"
            config["log_dir"] = str(output_dir / "runs" / variant / "logs")
            config["checkpoint_dir"] = str(output_dir / "runs" / variant / "checkpoints")
            print(f"[run_ablation] {args.vary}={value} seed={seed}")
            trainer = ContinualTrainer(config)
            summary = trainer.run_experiment("etf_anchor")
            accuracy = np.asarray(summary["accuracy_matrix"], dtype=float)
            run_row = logger.add_run(
                method=f"etf_anchor_{args.vary}_{value:g}",
                seed=seed,
                accuracy_matrix=accuracy,
                nc_metrics_path=summary.get("nc_metrics_path"),
            )
            run_row[args.vary] = float(value)
            rows.append(run_row)

    saved = logger.save()
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "ablation_results.csv", index=False)

    if not args.skip_plots and args.vary == "alpha" and not frame.empty:
        grouped = (
            frame.groupby("alpha")[["average_accuracy", "average_forgetting"]]
            .mean()
            .reset_index()
        )
        figure_dir = Path(base_config.get("log_dir", "results/logs")).parents[0] / "figures"
        plot_alpha_sensitivity(grouped, figure_dir)

    print(f"[run_ablation] saved aggregate files: {saved}")


if __name__ == "__main__":
    main()
