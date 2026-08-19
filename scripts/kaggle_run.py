"""Kaggle orchestration for Geometric Memory.

Examples:
    python scripts/kaggle_run.py --plan smoke
    python scripts/kaggle_run.py --plan baselines_seed42
    python scripts/kaggle_run.py --plan etf_seed42
    python scripts/kaggle_run.py --plan all_seed42
    python scripts/kaggle_run.py --plan lwf_vs_lwf_etf_strong_seeds43_44
    python scripts/kaggle_run.py --plan cifar100_missing_baselines_seeds43_44
    python scripts/kaggle_run.py --plan cifar10_all_3seed
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path("/kaggle/working/capstone_outputs")


PLANS = {
    "smoke": [
        ("config/kaggle_smoke.yaml", ["finetune", "etf_anchor"], [42]),
    ],
    "baselines_seed42": [
        ("config/kaggle_cifar_full.yaml", ["finetune", "ewc", "lwf"], [42]),
    ],
    "etf_seed42": [
        ("config/kaggle_cifar_full.yaml", ["etf_anchor"], [42]),
    ],
    "etf_sparse_seed42": [
        ("config/kaggle_etf_sparse.yaml", ["etf_anchor"], [42]),
    ],
    "lwf_etf_seed42": [
        ("config/kaggle_lwf_etf.yaml", ["lwf_etf"], [42]),
    ],
    "lwf_etf_strong_seed42": [
        ("config/kaggle_lwf_etf_strong.yaml", ["lwf_etf"], [42]),
    ],
    "lwf_vs_lwf_etf_strong_seeds43_44": [
        ("config/kaggle_cifar_full.yaml", ["lwf"], [43, 44]),
        ("config/kaggle_lwf_etf_strong.yaml", ["lwf_etf"], [43, 44]),
    ],
    "lwf_vs_lwf_etf_strong_3seed": [
        ("config/kaggle_cifar_full.yaml", ["lwf"], [42, 43, 44]),
        ("config/kaggle_lwf_etf_strong.yaml", ["lwf_etf"], [42, 43, 44]),
    ],
    "cifar100_missing_baselines_seeds43_44": [
        ("config/kaggle_cifar_full.yaml", ["finetune", "ewc"], [43, 44]),
        ("config/kaggle_etf_sparse.yaml", ["etf_anchor"], [43, 44]),
    ],
    "cifar10_baselines_3seed": [
        ("config/kaggle_cifar10_full.yaml", ["finetune", "ewc", "lwf", "etf_anchor"], [42, 43, 44]),
    ],
    "cifar10_lwf_etf_strong_3seed": [
        ("config/kaggle_cifar10_lwf_etf_strong.yaml", ["lwf_etf"], [42, 43, 44]),
    ],
    "cifar10_all_3seed": [
        ("config/kaggle_cifar10_full.yaml", ["finetune", "ewc", "lwf", "etf_anchor"], [42, 43, 44]),
        ("config/kaggle_cifar10_lwf_etf_strong.yaml", ["lwf_etf"], [42, 43, 44]),
    ],
    "paper_next_core": [
        ("config/kaggle_cifar_full.yaml", ["finetune", "ewc"], [43, 44]),
        ("config/kaggle_etf_sparse.yaml", ["etf_anchor"], [43, 44]),
        ("config/kaggle_cifar10_full.yaml", ["finetune", "ewc", "lwf", "etf_anchor"], [42, 43, 44]),
        ("config/kaggle_cifar10_lwf_etf_strong.yaml", ["lwf_etf"], [42, 43, 44]),
    ],
    "all_seed42": [
        ("config/kaggle_cifar_full.yaml", ["finetune", "ewc", "lwf", "etf_anchor"], [42]),
    ],
    "all_3seed": [
        ("config/kaggle_cifar_full.yaml", ["finetune", "ewc", "lwf", "etf_anchor"], [42, 43, 44]),
    ],
}


def run(command: list[str]) -> None:
    print("\n[kaggle_run]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def summarize_outputs() -> None:
    aggregate = OUTPUT_ROOT / "logs" / "aggregate" / "all_results.csv"
    if aggregate.exists():
        print("\n=== Aggregate Results ===")
        print(aggregate.read_text(encoding="utf-8"))

    manifest = {
        "output_root": str(OUTPUT_ROOT),
        "aggregate_csv": str(aggregate) if aggregate.exists() else None,
        "checkpoints": sorted(str(path) for path in (OUTPUT_ROOT / "checkpoints").glob("**/*.pt")),
        "summaries": sorted(str(path) for path in (OUTPUT_ROOT / "logs").glob("**/summary.json")),
    }
    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved manifest: {manifest_path}")


def zip_outputs() -> None:
    archive = Path("/kaggle/working/capstone_outputs")
    zip_path = Path("/kaggle/working/capstone_outputs.zip")
    if zip_path.exists():
        zip_path.unlink()
    if archive.exists():
        shutil.make_archive(str(archive), "zip", archive)
        print(f"Zipped outputs: {zip_path}")


def zip_minimal_outputs() -> None:
    """Zip logs/figures/tables without multi-GB checkpoints."""
    import zipfile

    archive = Path("/kaggle/working/capstone_outputs")
    zip_path = Path("/kaggle/working/capstone_outputs_minimal.zip")
    keep_suffixes = {".csv", ".json", ".npy", ".tex", ".png", ".pdf"}
    skip_names = {"training_log.csv"}

    if zip_path.exists():
        zip_path.unlink()
    if not archive.exists():
        return

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in archive.rglob("*"):
            if not path.is_file():
                continue
            relative_parts = path.relative_to(archive).parts
            if "checkpoints" in relative_parts:
                continue
            if path.name in skip_names:
                continue
            if path.suffix.lower() not in keep_suffixes:
                continue
            zf.write(path, path.relative_to(archive))
    print(f"Zipped minimal outputs: {zip_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", choices=sorted(PLANS), default="smoke")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--zip", action="store_true", help="Zip /kaggle/working/capstone_outputs at the end.")
    parser.add_argument(
        "--zip-minimal",
        action="store_true",
        help="Zip logs, figures, matrices, and tables without checkpoints.",
    )
    args = parser.parse_args()

    print("Python:", sys.version)
    run([sys.executable, "-c", "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"])

    if not args.skip_tests:
        run([sys.executable, "-m", "pytest"])

    for config, methods, seeds in PLANS[args.plan]:
        run(
            [
                sys.executable,
                "experiments/run_all.py",
                "--config",
                config,
                "--seeds",
                *[str(seed) for seed in seeds],
                "--methods",
                *methods,
            ]
        )

    summarize_outputs()
    if args.zip_minimal:
        zip_minimal_outputs()
    if args.zip:
        zip_outputs()


if __name__ == "__main__":
    main()
