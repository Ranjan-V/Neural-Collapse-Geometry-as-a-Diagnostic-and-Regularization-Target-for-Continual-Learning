"""Lightweight project smoke check.

This script intentionally avoids importing torch-heavy modules. Use it before
installing GPU dependencies to verify the config and evaluator layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluator import summarize_accuracy_matrix
from utils import load_config


def main() -> None:
    config = load_config(ROOT / "config" / "config.yaml")
    matrix = np.array(
        [
            [0.90, np.nan, np.nan],
            [0.82, 0.86, np.nan],
            [0.76, 0.80, 0.84],
        ]
    )
    summary = summarize_accuracy_matrix(matrix)
    print(f"config: {config['experiment_name']} ({config['dataset']})")
    print(
        "summary: "
        f"avg_acc={summary['average_accuracy']:.3f}, "
        f"forgetting={summary['average_forgetting']:.3f}, "
        f"bwt={summary['backward_transfer']:.3f}"
    )


if __name__ == "__main__":
    main()

