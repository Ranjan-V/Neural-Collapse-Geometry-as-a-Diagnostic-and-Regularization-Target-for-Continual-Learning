"""Restore an uploaded ZIP or Kaggle-auto-extracted resume dataset."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from utils.runtime import require_kaggle, read_json, file_hash, safe_extract


def main():
    require_kaggle()
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', help='Optional exact archive or auto-extracted dataset root')
    args = parser.parse_args()
    if args.archive:
        candidates = [Path(args.archive)]
    else:
        candidates = list(Path('/kaggle/input').rglob('session_resume_bundle.zip'))
        # Kaggle may automatically extract the ZIP during dataset upload.
        for integrity in Path('/kaggle/input').rglob('archive_integrity.json'):
            contents = read_json(integrity).get('sha256', {})
            if any(name.startswith('outputs/') and name.endswith('.pt') for name in contents):
                candidates.append(integrity.parent)
    if not candidates:
        print('No resume bundle attached. This is a fresh session.')
        return
    if len(candidates) != 1:
        raise RuntimeError('Multiple resume sources. Pass --archive with the intended source: ' + str(candidates))
    source = candidates[0]
    with tempfile.TemporaryDirectory(dir='/kaggle/working', prefix='resume_stage_') as temp:
        if source.is_file():
            safe_extract(source, temp)
            source = Path(temp)
        integrity = read_json(source / 'archive_integrity.json')['sha256']
        for name, expected in integrity.items():
            path = (source / name).resolve()
            if not path.is_relative_to(source.resolve()) or file_hash(path) != expected:
                raise RuntimeError('Resume archive integrity failure: ' + name)
        for name in ['PROTOCOL.md', 'protocol_manifest.json']:
            if file_hash(source / name) != file_hash(ROOT / name):
                raise RuntimeError('Resume bundle belongs to a different frozen protocol')
        destinations = []
        for name in integrity:
            if not name.startswith(('outputs/', 'analysis/', 'paper_outputs/')):
                continue
            path, dest = source / name, ROOT / name
            if dest.exists() and file_hash(dest) != file_hash(path):
                raise RuntimeError('Restore refuses to overwrite different existing output. Use a fresh notebook: ' + str(dest))
            destinations.append((path, dest))
        for path, dest in destinations:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(path, dest)
    print('Resume state restored to', ROOT / 'outputs')


if __name__ == '__main__':
    main()
