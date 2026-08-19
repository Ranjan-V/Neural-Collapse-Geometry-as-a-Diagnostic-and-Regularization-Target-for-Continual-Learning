# TNNLS Asset Inventory

## Counts
- Core paper figures: 7 PDF figures, with PNG companions.
- Extended TNNLS figures: 20 PDF figures, with PNG companions.
- Extended tables: 13 CSV tables and 13 LaTeX tables.

## Recommended Main-Paper Figures
- `fig_main_accuracy_forgetting.pdf`: main method comparison.
- `fig_cifar100_core_diagnostics.pdf`: strongest diagnostic evidence.
- `fig_nc_diagnostic_heatmap.pdf`: compact diagnostic summary.
- `fig_diagnostic_correlation_bars.pdf`: Pearson/Spearman diagnostic comparison.
- `fig_accuracy_matrices_cifar10.pdf` and `fig_accuracy_matrices_cifar100.pdf`: continual-learning accuracy matrices.
- `fig_etf_loss_components_cifar100_seed42.pdf`: ETF failure-mode/regularizer behavior evidence.

## Recommended Appendix Figures
- Seed-level performance.
- Accuracy/forgetting tradeoff.
- Method rank summary.
- Task-wise final accuracy and forgetting profiles.
- Seen-task average accuracy trajectories.
- Accuracy-matrix standard-deviation heatmaps.
- NC1-NC4 trajectories for CIFAR-10 and CIFAR-100.
- Drift/forgetting trajectories for CIFAR-10 and CIFAR-100.
- Pairwise NC/forgetting correlation heatmap.
- Method-specific diagnostic heatmaps.
- Training dynamics for CIFAR-100 seed 42.

## Main-Paper Table Budget
- Realistic main paper: 4 to 6 tables.
- Strong TNNLS-style package: 6 main tables plus 8 appendix tables.
- Do not use every generated table in the main text; keep overflow for appendix/supplement.

## Generated Core Figures
- `fig_accuracy_matrices_cifar10.pdf`
- `fig_accuracy_matrices_cifar100.pdf`
- `fig_bwt_and_etf_correlation.pdf`
- `fig_cifar100_core_diagnostics.pdf`
- `fig_main_accuracy_forgetting.pdf`
- `fig_nc_diagnostic_heatmap.pdf`
- `fig_nc_forgetting_scatter.pdf`

## Generated Extended Figures
- `fig_accuracy_forgetting_tradeoff.pdf`
- `fig_accuracy_matrix_std_cifar10.pdf`
- `fig_accuracy_matrix_std_cifar100.pdf`
- `fig_diagnostic_correlation_bars.pdf`
- `fig_drift_forgetting_trajectories_cifar10.pdf`
- `fig_drift_forgetting_trajectories_cifar100.pdf`
- `fig_etf_correlation_seed_distribution.pdf`
- `fig_etf_loss_components_cifar100_seed42.pdf`
- `fig_final_task_accuracy_profiles.pdf`
- `fig_method_diagnostic_heatmap_cifar10.pdf`
- `fig_method_diagnostic_heatmap_cifar100.pdf`
- `fig_method_rank_summary.pdf`
- `fig_nc1_nc4_trajectories_cifar10.pdf`
- `fig_nc1_nc4_trajectories_cifar100.pdf`
- `fig_pairwise_nc_forgetting_correlations.pdf`
- `fig_plasticity_retention_gap.pdf`
- `fig_seed_level_performance.pdf`
- `fig_seen_task_average_trajectory.pdf`
- `fig_task_forgetting_profiles.pdf`
- `fig_training_dynamics_cifar100_seed42.pdf`

## Generated Extended Tables
- `table_delta_vs_lwf.csv`
- `table_etf_correlation_per_seed.csv`
- `table_experiment_completion.csv`
- `table_experiment_protocol.csv`
- `table_final_task_accuracy_by_task.csv`
- `table_key_hyperparameters.csv`
- `table_nc_diagnostic_by_method.csv`
- `table_nc_diagnostic_summary.csv`
- `table_plasticity_retention_summary.csv`
- `table_seed_level_results.csv`
- `table_seen_task_accuracy_trajectory.csv`
- `table_task_forgetting_by_task.csv`
- `table_training_log_availability.csv`