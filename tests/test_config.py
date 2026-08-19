from pathlib import Path

from utils import load_config


def test_config_contains_required_top_level_keys():
    config = load_config(Path("config") / "config.yaml")
    required = {
        "seed",
        "dataset",
        "num_tasks",
        "classes_per_task",
        "batch_size",
        "architecture",
        "feature_dim",
        "epochs_per_task",
        "learning_rate",
        "alpha",
        "lambda_angle",
        "buffer_size",
        "eval_every_n_steps",
    }

    assert required.issubset(config)

