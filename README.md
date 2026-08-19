# Neural Collapse Geometry as a Diagnostic and Regularization Target for Continual Learning

This repository contains the code, experiment configuration, Kaggle execution scripts, tests, and curated non-submission analysis artifacts for the project:

**Neural Collapse Geometry as a Diagnostic and Regularization Target for Continual Learning**

The project studies whether neural-collapse (NC) geometry can be used to diagnose catastrophic forgetting in continual learning, and whether ETF-based geometric anchoring can act as a regularization target. The strongest empirical finding is diagnostic: on split CIFAR-100, NC3, NC4, and angular drift provide strong representation-level signals of task-1 forgetting. ETF anchoring is included as a geometry-aware regularization probe, while LwF remains the strongest regularizer in the reported experiments.

## Repository Layout

```text
.
├── config/                 # YAML experiment configs
├── data/                   # Dataset/task split loaders
├── evaluation/             # Accuracy, forgetting, BWT, table utilities
├── experiments/            # run_all.py and ablation runner
├── methods/                # Finetune, EWC, LwF, ETF Anchor, LwF+ETF
├── metrics/                # NC1-NC4, ETF drift, angle drift
├── models/                 # ResNet-18 feature extractor and classifier
├── scripts/                # Local, Kaggle, and analysis scripts
├── tests/                  # Unit tests
├── training/               # Single-task and continual trainers
├── visualization/          # Figure generation utilities
└── results/analysis/       # Curated analysis tables and selected PNG artifacts
```

Large runtime artifacts are intentionally excluded from git, including virtual environments, raw datasets, training logs, model checkpoints, transfer archives, PDFs, Word documents, mathematical notes, and manuscript submission files.

## Installation

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For Kaggle, use:

```bash
python -m pip install -q -r requirements-kaggle.txt
```

Kaggle usually already provides CUDA-enabled PyTorch, so do not reinstall PyTorch there unless CUDA is unavailable.

## Verification

Run unit tests:

```powershell
pytest
```

or on Windows:

```powershell
.\scripts\run_tests.ps1 -Install
```

Run a quick synthetic smoke check:

```powershell
python experiments\run_all.py --config config\smoke.yaml --seeds 42 --methods finetune etf_anchor --skip-plots
```

## Main Experiment Commands

Single-seed CIFAR-100 run:

```powershell
python experiments\run_all.py --config config\config_3050.yaml --seeds 42 --methods finetune ewc lwf etf_anchor
```

Kaggle paper runs are documented in `KAGGLE_README.md`. Typical commands:

```bash
python scripts/kaggle_run.py --plan cifar10_all_3seed --skip-tests --zip-minimal
python scripts/kaggle_run.py --plan cifar100_missing_baselines_seeds43_44 --skip-tests --zip-minimal
```

The scripts write outputs under `results/` locally or `/kaggle/working/capstone_outputs/` on Kaggle.

## Notes

- CIFAR-10 and CIFAR-100 are downloaded through `torchvision`.
- Random seeds are controlled in the experiment scripts and configs.
- Results in `results/analysis/` are curated outputs derived from completed local and Kaggle runs.
- Manuscript files, title pages, author-identifying documents, and detailed mathematical notes are kept outside this public code repository.
