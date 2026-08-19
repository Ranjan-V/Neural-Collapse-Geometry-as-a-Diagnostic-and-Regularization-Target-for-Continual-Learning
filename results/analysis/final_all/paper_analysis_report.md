# Paper Analysis Report

## Experiment Coverage
- Complete 3-seed results are available for CIFAR-10 and CIFAR-100.
- Seeds: 42, 43, and 44 for all paper-table methods.
- Methods: Finetune, EWC, LwF, ETF Anchor/ETF Anchor Sparse, and LwF+ETF Strong.

## Main Empirical Read
- CIFAR-10: best average accuracy is LwF (0.9563 +/- 0.0076); lowest forgetting is LwF (0.0029 +/- 0.0011).
- CIFAR-100: best average accuracy is LwF (0.6801 +/- 0.0107); lowest forgetting is LwF (0.0117 +/- 0.0100).

## Main Results Table
| Dataset | Method | Avg. Acc. | Forgetting | BWT | ETF-F Corr. |
|---|---|---:|---:|---:|---:|
| CIFAR-10 | Finetune | 0.6623 +/- 0.1185 | 0.3808 +/- 0.1443 | -0.3808 +/- 0.1443 | -- |
| CIFAR-10 | EWC | 0.6525 +/- 0.1152 | 0.3805 +/- 0.1373 | -0.3805 +/- 0.1373 | -- |
| CIFAR-10 | LwF | 0.9563 +/- 0.0076 | 0.0029 +/- 0.0011 | -0.0025 +/- 0.0011 | -- |
| CIFAR-10 | ETF Anchor | 0.6668 +/- 0.1009 | 0.3330 +/- 0.1013 | -0.3330 +/- 0.1013 | 0.1860 +/- 0.1068 |
| CIFAR-10 | LwF+ETF Strong | 0.9135 +/- 0.0048 | 0.0517 +/- 0.0116 | -0.0515 +/- 0.0113 | 0.1121 +/- 0.1970 |
| CIFAR-100 | Finetune | 0.4284 +/- 0.0229 | 0.3857 +/- 0.0177 | -0.3857 +/- 0.0177 | -- |
| CIFAR-100 | EWC | 0.4839 +/- 0.0191 | 0.2731 +/- 0.0182 | -0.2731 +/- 0.0182 | -- |
| CIFAR-100 | LwF | 0.6801 +/- 0.0107 | 0.0117 +/- 0.0100 | -0.0107 +/- 0.0091 | -- |
| CIFAR-100 | ETF Anchor Sparse | 0.2918 +/- 0.2081 | 0.4675 +/- 0.1806 | -0.4675 +/- 0.1806 | 0.5980 +/- 0.0436 |
| CIFAR-100 | LwF+ETF Strong | 0.6639 +/- 0.0100 | 0.0304 +/- 0.0017 | -0.0300 +/- 0.0022 | 0.6118 +/- 0.2695 |

## Diagnostic Read
- CIFAR-100 shows the strongest neural-collapse diagnostic signal: NC3 and NC4 are strongly associated with task-1 forgetting.
- CIFAR-100 angle drift reaches the target diagnostic threshold: mean Pearson r is above 0.8 and mean Spearman r is above 0.9.
- CIFAR-10 does not show the same ETF/angle-drift signal; NC3 is the most useful CIFAR-10 diagnostic but remains moderate.
- LwF remains the strongest method by accuracy and forgetting on both datasets.
- ETF anchoring alone and LwF+ETF do not beat LwF in these runs, so the safest claim is diagnostic-first rather than method-SOTA.

## Recommended Paper Claim
Neural collapse geometry is a useful diagnostic lens for continual-learning forgetting, especially on CIFAR-100. ETF anchoring is a plausible auxiliary regularization target, but the current implementation should be presented as exploratory rather than a replacement for LwF.

## What Not To Claim Yet
- Do not claim that ETF Anchor beats LwF; the 3-seed results do not support that.
- Do not claim universal ETF-drift correlation across datasets; CIFAR-10 is weak.
- Do not frame this as a TNNLS-ready method paper without additional method improvement or broader datasets.

## Generated Figures
- `fig_main_accuracy_forgetting.pdf`: main accuracy and forgetting comparison.
- `fig_bwt_and_etf_correlation.pdf`: backward transfer and ETF-drift correlation.
- `fig_nc_diagnostic_heatmap.pdf`: diagnostic correlation heatmap.
- `fig_accuracy_matrices_cifar10.pdf` and `fig_accuracy_matrices_cifar100.pdf`: mean task accuracy matrices.
- `fig_nc_forgetting_scatter.pdf`: NC metric scatter plots against task-1 forgetting.
- `fig_cifar100_core_diagnostics.pdf`: focused CIFAR-100 diagnostic evidence for NC3, NC4, and angle drift.

## Suggested Venue Direction
- Strongest current direction: a workshop or conference submission framed around diagnostics and representation geometry.
- For a higher-impact journal/conference target, add one more dataset and improve ETF regularization so it is competitive with LwF or clearly complementary.

## NC Diagnostic Summary
- CIFAR-10 angle_drift: Pearson -0.008 +/- 0.037, Spearman -0.006 +/- 0.011.
- CIFAR-10 etf_drift: Pearson 0.149 +/- 0.147, Spearman 0.487 +/- 0.243.
- CIFAR-10 nc1: Pearson 0.250 +/- 0.256, Spearman 0.319 +/- 0.384.
- CIFAR-10 nc2: Pearson -0.001 +/- 0.046, Spearman 0.002 +/- 0.037.
- CIFAR-10 nc3: Pearson 0.445 +/- 0.282, Spearman 0.469 +/- 0.350.
- CIFAR-10 nc4: Pearson -0.293 +/- 0.412, Spearman -0.183 +/- 0.451.
- CIFAR-100 angle_drift: Pearson 0.822 +/- 0.091, Spearman 0.910 +/- 0.049.
- CIFAR-100 etf_drift: Pearson 0.605 +/- 0.173, Spearman 0.869 +/- 0.088.
- CIFAR-100 nc1: Pearson 0.646 +/- 0.265, Spearman 0.674 +/- 0.281.
- CIFAR-100 nc2: Pearson 0.562 +/- 0.324, Spearman 0.637 +/- 0.298.
- CIFAR-100 nc3: Pearson 0.857 +/- 0.142, Spearman 0.859 +/- 0.120.
- CIFAR-100 nc4: Pearson -0.918 +/- 0.098, Spearman -0.874 +/- 0.120.