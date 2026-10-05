"""Kaggle setup: preserve the preinstalled CUDA torch/torchvision pair."""
import importlib.metadata
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from utils.runtime import require_kaggle


def main():
    require_kaggle()
    missing = []
    for requirement in (ROOT / 'requirements_kaggle.txt').read_text().splitlines():
        requirement = requirement.strip()
        if not requirement or requirement.startswith('#'):
            continue
        name = requirement.split('>=')[0]
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name in {'torch', 'torchvision'}:
                raise RuntimeError('Select a Kaggle GPU image with torch and torchvision preinstalled')
            missing.append(requirement)
        else:
            print(name, installed)
            from packaging.version import Version
            if '>=' in requirement and Version(installed) < Version(requirement.split('>=')[1]):
                raise RuntimeError(f'{name} {installed} is below {requirement}; setup will not silently upgrade an existing scientific environment.')
    if missing:
        subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)
    print('Required versions verified. Existing packages were not upgraded.')


if __name__ == '__main__':
    main()
