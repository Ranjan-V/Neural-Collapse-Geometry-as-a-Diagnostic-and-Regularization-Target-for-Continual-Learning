"""Build the Kaggle upload zip for the Geometric Memory project.

The project has a source package named ``data`` and also stores downloaded
datasets under ``data/raw``. This packager includes the source module while
excluding heavy dataset payloads and local training outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT.parents[1] / "kaggle-upload.zip"

EXCLUDED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "env",
    "outputs",
    "results",
    "runs",
    "venv",
    "wandb",
}

EXCLUDED_RELATIVE_DIRS = {
    Path("data/raw"),
    Path("data/processed"),
    Path("datasets"),
}

EXCLUDED_SUFFIXES = {
    ".7z",
    ".ckpt",
    ".gz",
    ".pth",
    ".pt",
    ".pyc",
    ".pyo",
    ".tar",
    ".zip",
}

EXCLUDED_FILE_NAMES = {".DS_Store", "Thumbs.db"}

REQUIRED_FILES = {
    "config/kaggle_cifar10_full.yaml",
    "config/kaggle_cifar10_lwf_etf_strong.yaml",
    "data/dataset.py",
    "data/__init__.py",
    "scripts/analyze_kaggle_outputs.py",
    "scripts/kaggle_run.py",
    "requirements-kaggle.txt",
    "config/kaggle_lwf_etf_strong.yaml",
}


def is_excluded(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
        return True
    if any(rel == blocked or blocked in rel.parents for blocked in EXCLUDED_RELATIVE_DIRS):
        return True
    if path.name in EXCLUDED_FILE_NAMES:
        return True
    if path.is_file() and path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    return False


def build_zip(output: Path) -> list[str]:
    if output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    included: list[str] = []
    with ZipFile(output, "w", ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(ROOT.rglob("*")):
            if is_excluded(path):
                continue
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT).as_posix()
            zf.write(path, rel)
            included.append(rel)

    missing = sorted(REQUIRED_FILES - set(included))
    if missing:
        raise RuntimeError(f"Upload zip is missing required files: {missing}")
    return included


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    included = build_zip(args.output)
    print(args.output)
    print(f"files={len(included)}")
    for rel in sorted(REQUIRED_FILES):
        print(f"required ok: {rel}")


if __name__ == "__main__":
    main()
