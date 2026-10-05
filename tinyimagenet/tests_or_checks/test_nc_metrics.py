import math

import torch

from metrics.nc_metrics import (
    centered_normalized_means,
    compute_class_means,
    compute_etf_drift,
    compute_nc1,
    compute_nc2,
    compute_nc3,
)


def test_nc1_is_zero_for_perfect_within_class_collapse():
    features = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    nc1 = compute_nc1(features, labels)

    assert torch.allclose(nc1, torch.tensor(0.0, dtype=torch.float64), atol=1e-10)


def test_nc2_is_zero_for_three_class_simplex_etf():
    sqrt3 = math.sqrt(3.0)
    features = torch.tensor(
        [
            [1.0, 0.0],
            [-0.5, sqrt3 / 2.0],
            [-0.5, -sqrt3 / 2.0],
        ],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 1, 2])
    class_means = compute_class_means(features, labels)

    nc2 = compute_nc2(class_means)
    _, normalized, _, _ = centered_normalized_means(class_means)
    cosines = normalized @ normalized.T
    off_diag = cosines[torch.triu_indices(3, 3, offset=1).unbind()]

    assert torch.allclose(nc2, torch.tensor(0.0, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(off_diag, torch.full((3,), -0.5, dtype=torch.float64), atol=1e-10)


def test_nc3_is_zero_when_weights_align_with_centered_means():
    sqrt3 = math.sqrt(3.0)
    features = torch.tensor(
        [
            [1.0, 0.0],
            [-0.5, sqrt3 / 2.0],
            [-0.5, -sqrt3 / 2.0],
        ],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 1, 2])
    class_means = compute_class_means(features, labels)
    _, _, centered, _ = centered_normalized_means(class_means)

    nc3 = compute_nc3(class_means, centered)

    assert torch.allclose(nc3, torch.tensor(0.0, dtype=torch.float64), atol=1e-10)


def test_etf_drift_zero_and_translation_epsilon_squared():
    frozen_means = {
        0: torch.tensor([0.0, 0.0], dtype=torch.float64),
        1: torch.tensor([1.0, 0.0], dtype=torch.float64),
        2: torch.tensor([0.0, 1.0], dtype=torch.float64),
    }
    current_means = {class_id: mean.clone() for class_id, mean in frozen_means.items()}

    zero_drift = compute_etf_drift(current_means, frozen_means)

    epsilon = 0.25
    translated = {
        class_id: mean + torch.tensor([epsilon, 0.0], dtype=torch.float64)
        for class_id, mean in frozen_means.items()
    }
    translated_drift = compute_etf_drift(translated, frozen_means)

    assert torch.allclose(zero_drift, torch.tensor(0.0, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(
        translated_drift,
        torch.tensor(epsilon**2, dtype=torch.float64),
        atol=1e-10,
    )

