# Existing implementation audit

This extension was built by static inspection on 2026-09-03. No scientific
execution, package installation or dataset download was performed locally.

The `src/models`, `src/methods`, `src/metrics` and `src/evaluation` modules
were copied from the existing geometric_memory project. Their original file
hashes are in `evidence/reused_source_hashes.json`. Their loss/metric functions
remain unchanged. The new engine calls these functions directly.

## Evidence for recipe selection

| Method | Existing named configuration | Archived run evidence |
|---|---|---|
| FT, LwF | kaggle_cifar_full.yaml | Capstone Outputs 1; LwF seeds 43/44 in Capstone Output 3 |
| ETF-S | kaggle_etf_sparse.yaml | seed 42 in Capstone Outputs 1; seeds 43/44 in Missing Baselines outputs |
| LwF+ETF | kaggle_lwf_etf_strong.yaml | seed 42 in Capstone Outputs 2; seeds 43/44 in Capstone Output 3 |

The current manuscript's configuration table agrees with the named sparse
and strong-hybrid run families. The named YAMLs are copied into evidence/.
The dense ETF config has lambda_angle=0.5 and q=1; it is NOT the ETF-S
recipe. The selected sparse and strong-hybrid configs both have lambda=1,
q=4, raw centroids, no warmup, LwF coefficient 1 and temperature 2.

Historical output folders store experiment names and summaries, but not
complete immutable runtime configurations or original source commit hashes.
Therefore we can establish recipe identity from these consistent sources,
but cannot retrospectively certify byte-identical historical execution.
The legacy catalog labels this provenance limitation instead of inventing
original config hashes. New runs record actual resolved configs and hashes.

## Scientific details that must remain explicit

- ResNet-18 uses torchvision's standard 7x7/stride-2 convolution and maxpool.
  No CIFAR-style stem has been substituted for the Tiny ImageNet extension.
- Both training CE and evaluation use task-local logits. This is task-aware
  continual learning, not task-agnostic 200-way classification.
- LwF uses batchmean KL(teacher || student) across all old output columns,
  temperature 2, and T-squared scaling. Teacher capture precedes expansion.
- Anchor memory is task 1 only. `anchor_all_seen_tasks=false` is intentional.
  The existing option for all-seen tasks does not accumulate global labels;
  this extension does not enable that option.
- The actual buffer contains 200 normalized, augmented image tensors per
  task-1 class (4000 total). Frozen training means are computed over the
  full post-task-1 training pass, not merely those retained FIFO examples.
- Balanced sampling takes floor(128/20)=6 exemplars per class, hence 120
  anchor images per update. The CE batch remains 128 (last batch smaller).
- BatchNorm running statistics are frozen on anchor forwards. Student BN
  statistics still update on current-task forwards.
- Classifier has bias, expands preserving old rows, and all rows remain
  trainable. No old-row freezing or class-incremental calibration is added.
- NC1 uses the existing default torch.linalg.pinv and float32 features.
  Center normalization uses eps=1e-12; NC3 uses cosine_similarity defaults.
- NC4 is agreement. Angular drift equals NC2 for the same class set;
  duplicate measurements are not independent corroboration.
- New diagnostic centroids are computed on fixed task-1 validation images
  for EVERY method. They are separate from training-anchor references.
  Historical ETF drift often used a training-anchor reference and is
  labeled accordingly; missing baseline drift cannot be reconstructed.
- Original CIFAR data order used random.Random(seed).shuffle. Tiny ImageNet
  has a new, declared numpy.default_rng(seed) permutation on sorted class
  IDs. This is a new dataset split, not a claimed replay of a CIFAR split.

No unresolved choice of alpha/lambda/LwF coefficient is being deferred to
observed Tiny ImageNet outcomes. The remaining provenance limitation is
historical, not a reason to tune the new protocol.
