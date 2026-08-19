# Kaggle Run Guide

Upload `kaggle-upload.zip` to Kaggle as a dataset or upload it into a notebook session.

Recommended notebook settings:

- Accelerator: GPU
- GPU: T4 x2 or P100. The code uses one GPU by default; this is fine.
- Internet: ON so torchvision can download CIFAR-100/CIFAR-10 automatically.
- Output path: `/kaggle/working/capstone_outputs`

## Cell 1: Copy or Unzip Code

```python
import os
import shutil
import zipfile
from pathlib import Path

target = Path("/kaggle/working/geometric_memory")
if target.exists():
    shutil.rmtree(target)

zip_candidates = list(Path("/kaggle/input").glob("**/kaggle-upload.zip"))
zip_candidates += list(Path("/kaggle/input").glob("**/capstone-kaggle-upload.zip"))
zip_candidates += list(Path("/kaggle/working").glob("kaggle-upload.zip"))
zip_candidates += list(Path("/kaggle/working").glob("capstone-kaggle-upload.zip"))

if zip_candidates:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_candidates[0], "r") as zf:
        zf.extractall(target)
    print("Extracted:", zip_candidates[0])
else:
    folder_candidates = [
        p.parents[1]
        for p in Path("/kaggle/input").glob("**/scripts/kaggle_run.py")
    ]
    if not folder_candidates:
        raise FileNotFoundError("Could not find kaggle-upload.zip or a folder containing scripts/kaggle_run.py under /kaggle/input")
    shutil.copytree(folder_candidates[0], target)
    print("Copied:", folder_candidates[0])

print("Ready at:", target)
print("Top-level files:", sorted(p.name for p in target.iterdir())[:20])
```

## Cell 2: Install Small Missing Dependencies

```bash
%cd /kaggle/working/geometric_memory
!python -m pip install -q -r requirements-kaggle.txt
```

Do not reinstall PyTorch on Kaggle unless CUDA is missing. Kaggle GPU notebooks usually include CUDA PyTorch already.

## Cell 3: Verify GPU and Tests

```bash
%cd /kaggle/working/geometric_memory
!python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
!python -m pytest
```

## Cell 4: Smoke Run

```bash
!python scripts/kaggle_run.py --plan smoke --skip-tests --zip-minimal
```

## Cell 5: Missing CIFAR-100 Baselines

Run this after the existing seed-42 and LwF/LwF+ETF seed-43/44 downloads. It completes 3-seed baselines for Finetune, EWC, and ETF Sparse.

```bash
!python scripts/kaggle_run.py --plan cifar100_missing_baselines_seeds43_44 --skip-tests --zip-minimal
```

This single cell runs:

- CIFAR-100 Finetune seeds 43 and 44
- CIFAR-100 EWC seeds 43 and 44
- CIFAR-100 ETF Sparse seeds 43 and 44

## Cell 6: Second Dataset, Split CIFAR-10

This gives the paper a second dataset with all methods over three seeds.

```bash
!python scripts/kaggle_run.py --plan cifar10_all_3seed --skip-tests --zip-minimal
```

This single cell runs:

- CIFAR-10 Finetune, EWC, LwF, ETF Anchor seeds 42/43/44
- CIFAR-10 LwF+ETF Strong seeds 42/43/44

## Optional: One-Cell Core Paper Run

```bash
!python scripts/kaggle_run.py --plan paper_next_core --skip-tests --zip-minimal
```

This runs Cell 5 and Cell 6 together. Prefer the split cells if your Kaggle quota is tight.

## Optional Older Plans

```bash
!python scripts/kaggle_run.py --plan all_seed42 --skip-tests --zip-minimal
!python scripts/kaggle_run.py --plan lwf_vs_lwf_etf_strong_seeds43_44 --skip-tests --zip-minimal
!python scripts/kaggle_run.py --plan lwf_vs_lwf_etf_strong_3seed --skip-tests --zip-minimal
!python scripts/kaggle_run.py --plan baselines_seed42 --skip-tests --zip
!python scripts/kaggle_run.py --plan etf_seed42 --skip-tests --zip
!python scripts/kaggle_run.py --plan etf_sparse_seed42 --skip-tests --zip
!python scripts/kaggle_run.py --plan lwf_etf_seed42 --skip-tests --zip
```

## Outputs

Everything important is saved under:

```text
/kaggle/working/capstone_outputs/
```

Important files:

```text
/kaggle/working/capstone_outputs/logs/aggregate/all_results.csv
/kaggle/working/capstone_outputs/logs/aggregate/main_table.tex
/kaggle/working/capstone_outputs/checkpoints/**/seed_*_task_*.pt
/kaggle/working/capstone_outputs/manifest.json
/kaggle/working/capstone_outputs_minimal.zip
```

Download `capstone_outputs_minimal.zip` after each run for tables/figures/metrics. Use `--zip` instead of `--zip-minimal` only when you need to download all checkpoint files; full zips can be several GB.

## Local Merge After Downloads

After extracting downloaded Kaggle outputs on your machine, merge them with:

```powershell
cd D:\Capstone\code\geometric_memory
.\.venv\Scripts\Activate.ps1
python scripts\analyze_kaggle_outputs.py --roots "D:\Capstone\Capstone Outputs 1" "D:\Capstone\Capstone Outputs 2" "D:\Capstone\Capstone Output 3" "D:\Capstone\Capstone Output 4" --output-dir results\analysis\paper_next
```

Adjust folder names to match the downloaded output folders. The script writes seed-level CSVs, mean/std tables, LaTeX tables, and NC diagnostic correlations.
