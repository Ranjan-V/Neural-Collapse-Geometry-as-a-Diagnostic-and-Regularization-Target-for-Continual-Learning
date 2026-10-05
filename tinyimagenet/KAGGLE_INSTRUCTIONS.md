# Kaggle execution, in order

Use the delivered notebook; it contains every cell below without Windows
paths. Import the separate `.ipynb`, attach the source ZIP as input, and
attach Tiny ImageNet-200. Select GPU T4 x2. File persistence can help retain
working files during supported restarts, but both external ZIP downloads
remain required before ending a session. Do not rely on variables in memory.

| Cell | Action |
|---|---|
| 1 | Verify Python, torch, CUDA, both T4 GPUs, disk space. |
| 2 | Detect dataset; record actual Kaggle URL/version. |
| 3 | Find ZIP or auto-extracted source and copy safely into working/. |
| 4 | Install missing dependencies; restore prior bundle; bind dataset/splits. |
| 5 | Run tests and all four smoke/resume comparisons. |
| 6 | Print protocol and completed/pending jobs. |
| 7 | Start/resume both independent GPU workers, budget 9.25 hours. |
| 8 | Display statuses and failure files. |
| 9 | Optional final analysis; default off to leave archive/download time. |
| 10 | Create/refresh results and resume archives, final archive if analyzed. |
| 11 | List names, sizes and SHA256; download both through Output sidebar. |
| 12 | Re-list download files without retraining or re-zipping. |

The scheduler queues FT seeds 42/43/44, LwF seeds 42/43/44, ETF-S seeds
42/43/44, then LwF+ETF seeds 42/43/44. Assignment is greedy to the next
available GPU and is recorded per attempt. At most two processes train.
Each run sees one GPU as cuda:0. Sessions can stop in the middle of a task.

If your Kaggle interface auto-extracts input ZIPs, do not search only for a
literal ZIP filename. Cell 3 recognizes the source by its required files;
restore_session.py recognizes unpacked resume contents by archive_integrity.
Missing or multiple sources produce actionable errors, never a guessed path.

## Session restart

1. Keep the two downloaded session ZIPs and their checksums on your machine.
2. Create a fresh notebook/session with the SAME source ZIP and dataset.
3. Upload the latest session_resume_bundle.zip as input.
4. Run setup/smoke cells again, then the scheduler. It detects completed,
   resumable and failed jobs. No split or buffer is regenerated on resume.
5. If a job failed, read outputs/failures and worker_logs. Engineering fixes
   must be documented. `--retry-failed` only retries the unchanged recipe.

Restoring into different existing outputs is refused; use a fresh notebook
instead of deleting unknown files. The source package intentionally contains
no personal manuscript/title-page/math files and no laptop venv.

## CPU analysis after 12/12 runs

Use a new Kaggle notebook with accelerator None. Attach the same source and
the final session_results_bundle.zip (small metadata/log archive is enough).
The dual-GPU notebook's Cell 1 is for training; skip that GPU assertion in
this dedicated analysis notebook. Copy the source using Cell 3, then run:

```python
from pathlib import Path
import os, sys, subprocess
PROJECT = Path('/kaggle/working/neurocomputing_tinyimagenet_nc')
os.chdir(PROJECT)
subprocess.run([sys.executable, 'scripts/install_missing.py'], check=True)
# ZIP or auto-extracted root, found in the Kaggle Input sidebar:
RESULTS_SOURCE = '/kaggle/input/YOUR-RESULTS/session_results_bundle.zip'
subprocess.run([sys.executable, 'scripts/restore_session.py', '--archive', RESULTS_SOURCE], check=True)
subprocess.run([sys.executable, 'analyze_temporal_diagnostics.py', '--include-legacy'], check=True)
subprocess.run([sys.executable, 'create_archives.py', '--final', '--results-only'], check=True)
```

In this CPU session no resume training is possible without full checkpoints;
the original session_resume_bundle.zip remains your training backup. The
new results-only archive produced here is for scientific reporting. Never
replace a checkpoint backup with it. Preserve the originals with dates.

## Storage, timing and download recovery

The queue represents 12 x 10 x 100 = 12,000 training epochs, plus validation,
checkpointing and analysis. There is no measured timing estimate yet.
Use system/resource_usage.csv after the first sessions to estimate remaining
runtime. Epoch training throughput and wall time including diagnostics are
recorded separately. Two T4 GPUs improve throughput across runs, not the
speed of a single run. The archive step consumes additional time and disk.

Default resume ZIPs include every task model snapshot. When disk is tight,
`python create_archives.py --omit-task-models` makes a smaller recovery
archive that still has full unfinished states and completed final models,
but excludes historical task snapshots; document this choice and preserve
any prior snapshots already downloaded. No scientific settings change.

Prefer the Output sidebar download action. A printed file link does not
guarantee a browser download. Refresh the file list, wait for ZIP creation
and verification to finish, and download both files before closing the
session. Save a version with outputs as another copy when the platform
allows it. No notebook cell can confirm that a browser finished saving a
file to your laptop.
