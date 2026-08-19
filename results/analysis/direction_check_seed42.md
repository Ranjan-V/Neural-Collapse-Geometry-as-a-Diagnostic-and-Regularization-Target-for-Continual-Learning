# Direction Check: CIFAR-100 Seed 42

This file merges the two Kaggle output downloads and ignores synthetic smoke rows.

| Experiment | Method | Avg Acc | Forgetting | BWT | ETF-Forget Corr | Delta Acc vs LwF | Delta Forget vs LwF |
|---|---:|---:|---:|---:|---:|---:|---:|
| kaggle_cifar_full | finetune | 0.4047 | 0.4059 | -0.4059 | -- | -0.2645 | +0.3851 |
| kaggle_cifar_full | ewc | 0.4638 | 0.2940 | -0.2940 | -- | -0.2054 | +0.2733 |
| kaggle_cifar_full | lwf | 0.6692 | 0.0207 | -0.0189 | -- | +0.0000 | +0.0000 |
| kaggle_cifar_full | etf_anchor | 0.2134 | 0.1371 | -0.1116 | 0.4207 | -0.4558 | +0.1164 |
| kaggle_etf_sparse | etf_anchor | 0.4133 | 0.3615 | -0.3615 | 0.5906 | -0.2559 | +0.3408 |
| kaggle_lwf_etf | lwf_etf | 0.6478 | 0.0333 | -0.0329 | 0.8409 | -0.0214 | +0.0125 |
| kaggle_lwf_etf_strong | lwf_etf | 0.6537 | 0.0322 | -0.0322 | 0.8823 | -0.0155 | +0.0115 |

## Read

- Standalone ETF Anchor is not the winning method yet. The sparse version recovers average accuracy versus naive finetuning but still forgets heavily.
- LwF is the strongest baseline on accuracy and forgetting for seed 42.
- LwF+ETF strong is close to LwF: -0.0155 average accuracy and +0.0115 forgetting, while producing a strong ETF drift/forgetting correlation of 0.8823.
- Best current paper direction: present ETF Anchor as a geometry-preserving auxiliary regularizer and diagnostic layered on a replay-free continual learner, not as a standalone replacement for distillation.
