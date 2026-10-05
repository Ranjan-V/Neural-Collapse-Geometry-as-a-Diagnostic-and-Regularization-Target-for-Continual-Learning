# Reproduction and recovery

Scientific execution is restricted to Kaggle by entry-point guards.

Each worker sees a single T4 as cuda:0. GPU 0 and GPU 1 execute independent
method/seed jobs. The batch size stays 128. No DataParallel or DDP is used.

Python, NumPy, torch CPU and CUDA RNG states are checkpointed. cuDNN uses
deterministic=true and benchmark=false. TF32 is disabled. Deterministic
algorithms run with warn_only=true, preserving visibility of unsupported
operations rather than claiming bitwise determinism on every CUDA release.
Reference: [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

DataLoader shuffling and augmentations use a deterministic per-task/per-epoch
generator plus worker_init_fn. Workers restart each epoch; their future
states do not have to be guessed on resume. Evaluation uses its own loader
generator. Model construction on resume is followed by RNG restoration.
The smoke suite checks resumed and uninterrupted weights and diagnostics at
rtol=1e-6/atol=1e-7 on the actual Kaggle environment.

Full checkpoint state includes student/classifier, SGD, cosine scheduler,
AMP scaler, method/seed/task/epoch/global step, RNG states, frozen teacher,
FIFO image and feature buffers, anchor/reference means, class mappings,
accuracy matrix, temporal diagnostics, forgetting history, resource history,
training-log byte offset and protocol/config/split/source hashes.

Writes go to a temporary file, are fsynced, checksummed, renamed, and then
published through an atomic latest.json pointer. Two generations are kept.
Uncommitted log tails are truncated to the restored checkpoint's byte offset.
Corrupt latest states are explicitly reported before previous-state recovery;
if both fail, training stops. Incompatible hashes never trigger silent fallback.

A full checkpoint is committed at every completed epoch, at the final
epoch before task finalization, after finalization, and after teacher/head
transition and task-start diagnostics. Historical transition states roll
with the two-generation retention; task-end model snapshots remain for all
tasks. At completion, final model and reports are retained, then redundant
full optimizer/teacher/buffer generations are removed to fit Kaggle storage.
Resume archives retain all historical task snapshots by default, plus the
two full states for every unfinished job and the final model of completed jobs.

Original package versions, CUDA/cuDNN and GPU model are compared at resume.
A change is rejected rather than silently mixing environments. Obtain the
matching Kaggle image/environment when necessary; cross-release exact
continuation is not promised. Any engineering correction needs a new source
version and a documented decision about new versus continued experiments.

Dataset identity is based on class mappings, filenames and image byte sizes.
It detects layout/class/count changes but is not an all-pixels checksum.
Archive SHA256 verification protects the serialized resume payload. Keep
the SAME version of the attached Kaggle Tiny ImageNet dataset across sessions.

Only restore archives from this trusted project. Full torch checkpoints use
weights_only=False because RNG state contains NumPy objects; arbitrary third
party pickle files are not supported. ZIP paths and member checksums are
checked before restore, and different existing outputs are never overwritten.
