"""Shared utilities for configuration and reproducibility."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path!s}.")
    return config


def set_random_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_output_dirs(config: dict[str, Any]) -> None:
    """Create configured log and checkpoint directories."""
    for key in ("log_dir", "checkpoint_dir"):
        if key in config:
            Path(config[key]).mkdir(parents=True, exist_ok=True)
