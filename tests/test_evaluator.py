import math

import numpy as np
import pandas as pd

from evaluation.evaluator import (
    compute_average_accuracy,
    compute_backward_transfer,
    compute_etf_forgetting_correlation,
    compute_forgetting,
    generate_results_table,
)


def test_accuracy_and_forgetting_metrics():
    matrix = np.array(
        [
            [0.90, np.nan, np.nan],
            [0.80, 0.85, np.nan],
            [0.70, 0.75, 0.82],
        ]
    )

    assert compute_average_accuracy(matrix) == np.mean([0.70, 0.75, 0.82])

    forgetting = compute_forgetting(matrix)
    assert math.isclose(forgetting["per_task"][0], 0.20)
    assert math.isclose(forgetting["per_task"][1], 0.10)
    assert math.isclose(forgetting["average_forgetting"], 0.15)

    bwt = compute_backward_transfer(matrix)
    assert math.isclose(bwt, np.mean([-0.20, -0.10]))


def test_etf_forgetting_correlation_uses_checkpoint_forgetting_when_available():
    frame = pd.DataFrame(
        {
            "etf_drift": [0.0, 0.5, 1.0],
            "task1_forgetting": [0.0, 0.25, 0.5],
        }
    )
    matrix = np.array([[0.9], [0.8], [0.7]])

    assert math.isclose(compute_etf_forgetting_correlation(frame, matrix), 1.0)


def test_generate_results_table_contains_mean_std_cells():
    rows = [
        {"method": "finetune", "average_accuracy": 0.50, "average_forgetting": 0.30, "backward_transfer": -0.30},
        {"method": "finetune", "average_accuracy": 0.60, "average_forgetting": 0.20, "backward_transfer": -0.20},
    ]

    table = generate_results_table(rows)

    assert "\\begin{tabular}" in table
    assert "finetune" in table
    assert "$\\pm$" in table
