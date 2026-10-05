# Frozen Tiny ImageNet protocol v1

Primary question: Does degradation of neural-collapse geometry provide a
robust, temporally informative representation-level diagnostic of forgetting?
ETF regularization is a controlled intervention, not a superiority claim.

No hyperparameter will be changed in response to the observed Tiny ImageNet scientific results. Engineering corrections that do not use hypothesis outcomes must be documented separately.

## Design

- Tiny ImageNet-200: 200 classes; 500 training and 50 labeled validation
  images per class; 64x64 RGB. Unlabeled test images are not evaluated.
- Ten class-disjoint tasks, 20 classes each; scientific seeds 42, 43, 44.
- Sort original training class IDs, then numpy.default_rng(seed).permutation.
  Bind WordNet IDs/class mappings once on Kaggle, before any method runs.
  All methods with a given seed receive the identical split.
- Exactly FT, LwF, ETF-S and LwF+ETF: 12 principal runs. No Tiny ImageNet EWC.
- Task-aware training AND evaluation with known task identity, local labels
  0..19, and an expandable biased linear classifier. All rows trainable.
- Standard torchvision ResNet-18 7x7/stride-2/maxpool stem, 512-dimensional
  features, no pretrained weights; identical architecture across methods.

## Optimization and memory

SGD, LR 0.1, momentum 0.9, weight decay 1e-4, 100 epochs/task, CE batch 128,
gradient clip norm 1.0, mixed precision. CosineAnnealingLR resets each task
and advances per successful optimizer update, with T_max = epochs * batches.
No effective-batch enlargement for dual GPUs. Two loader workers/run and
two CPU threads/run are engineering settings, not method-specific tuning.

LwF: frozen pre-expansion teacher, temperature 2, coefficient 1, batchmean
KL divergence with T-squared scaling. ETF-S: alpha=1. LwF+ETF: alpha=0.25.
Both: lambda_angle=1, centroid weight=1, raw centroid residuals, q=4 global
steps, zero warmup, BatchNorm eval on anchors. Memory is task-1-only: 200
image exemplars/class, sampled by the existing balanced buffer (actual
anchor batch 120 for 20 classes). No replay CE is added. Full training-pass
means define frozen anchors, matching the source implementation.

Train augmentation: RandomCrop(64,padding=8), horizontal flip p=0.5,
ToTensor and fixed mean [0.485,0.456,0.406], std [0.229,0.224,0.225].
Validation: ToTensor and the same normalization only. No future-task
training images are used before their task. Validation is never optimized.

## Diagnostics and outcomes

Establish reference means/accuracy after task 1 using ALL task-1 validation
images (1000). Every later task: epoch 0, 5, 10, ..., 100. Total 190 rows
including the task-1 reference. Epoch denotes completed epochs; tasks in
files are zero-based. Validation mode, float32 features, identical fixed
images across checkpoints. No stochastic diagnostic augmentation.

Measure NC1, NC2, NC3, NC4 agreement, centroid drift and angular drift, plus
task-1 accuracy, F1=post-task-1 accuracy minus current accuracy, current and
old-task accuracies. NC2/angular drift duplicate the existing definition.
Raw losses, update indicators, buffer counts and gradient norms are logged.

Accuracy matrix R[t,s] is measured after task t on task s. Report mean
final accuracy, per-old-task max-previous-minus-final forgetting, its mean,
and BWT=mean(final-old minus immediate-old). Negative forgetting is retained.
Task-1 diagnostic forgetting always uses the fixed task-1 baseline rather
than the running maximum used in the final forgetting summary.

## Prespecified analysis

Each dataset/method/config/seed is analyzed separately. Minimum eight
pairs; insufficient sample size or constant variables produce NA + reason.
No pooling of methods or incompatible configurations. Three independent
seeds are the main unit of replication; arithmetic mean, sample SD and
number of estimable seeds accompany seed-level values.

1. Raw Pearson and Spearman against task-1 forgetting for all six metrics.
2. Partial Pearson: residualize both variables against intercept, normalized
   global progress, and categorical task fixed effects. Partial Spearman:
   rank variables and continuous progress before the same residualization;
   task remains a categorical control.
3. Detrending: fixed linear progress trend only; correlate residuals with
   Pearson and Spearman. No search over trend families.
4. Within-task associations: levels and first differences; retain every
   later task and summarize per seed before between-seed summaries.
5. Lag horizons: EXACT +5 epochs, +10 epochs, and current task end. Pair
   within a task only; never interpolate, pair backward, or cross boundaries.
   Report raw and partial Pearson/Spearman for all prespecified horizons.
6. EXPLORATORY change-to-subsequent-change: g[t]-g[t-5] versus F[t+5]-F[t]
   only when all three actual checkpoints exist in the same task.
7. Paired moving-block bootstrap: 2000 replicates, fixed length 4 ordered
   checkpoints, stratified by task, RNG seed 20260903, 95% percentile CI for
   raw/partial/detrended Pearson. Refit controls for each resample. Require
   at least two blocks/task and >=80% valid replicates; no IID fallback.

Bootstrap intervals are descriptive uncertainty under this block scheme,
not a guarantee of coverage for arbitrary nonstationary learning curves.
No checkpoint-level p-values are used as independent-experiment evidence.
End-of-task lag targets repeat within task; controls and this dependence
must accompany interpretation. Levels alone are not incremental prediction
beyond current forgetting. NC4's expected directional association is negative.

Existing CIFAR logs are imported separately using their observed timing.
Named-config snapshots and known 79 batches/epoch permit declared fractional
epoch reconstruction. Exact lag pairs may be absent. No old model is rerun,
no missing centroid references are invented, and historical reference
differences remain visible. CIFAR-10 NC2 can be structurally degenerate for
two centered class means; NA correlations must not be dropped from reporting.

## Stopping and completion

Fixed 100 epochs/task, no scientific early stop, no tuning after results.
Default session budget 9.25 hours. Stop launching/request epoch-boundary
pause 15 minutes before that deadline; preserve the last committed epoch
if a process exceeds the grace. The scheduler archives after worker exit.
Archiving consumes additional wall time; the budget is not a guarantee of
completion before a platform shutdown. Start archive/download before timeout.

AMP overflow skips are explicit events with scaler reduction, preserving
the old training recipe; nonfinite loss/features/metrics or non-AMP gradients
fail the run. OOM causes a failure report, never an automatic batch change.
Microbatch fallback is intentionally not enabled because BatchNorm would
make it scientifically non-equivalent without further design work.

Complete means 12/12 full runs and all requested analysis tables, with
unavailable estimates explicitly marked. Negative results remain in every
report. No target correlation threshold is used to decide success.
