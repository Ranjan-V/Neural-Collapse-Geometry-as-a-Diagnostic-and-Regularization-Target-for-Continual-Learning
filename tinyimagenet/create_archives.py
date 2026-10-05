"""Separate inspectable scientific results from full session recovery state."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from utils.runtime import require_kaggle, file_hash, read_json, atomic_json, protocol, source_version, now


def scientific_files(output):
    for path in sorted(output.rglob('*')):
        if path.is_file() and 'checkpoints' not in path.relative_to(output).parts and path.suffix not in {'.pt', '.pth', '.tmp', '.lock'} and not path.name.startswith('stop_'):
            yield path, 'outputs/' + path.relative_to(output).as_posix()
    for directory in ['configs', 'evidence', 'analysis', 'paper_outputs']:
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and path.suffix not in {'.zip', '.pyc', '.tmp'}:
                yield path, path.relative_to(ROOT).as_posix()
    for name in ['PROTOCOL.md', 'REPRODUCIBILITY.md', 'protocol_manifest.json', 'DATASET_CITATION.md', 'SOURCE_AUDIT.md', 'ENGINEERING_CHANGES.md', 'requirements_kaggle.txt']:
        yield ROOT / name, name


def archive(path, entries):
    entries = list(entries)
    uncompressed = sum(p.stat().st_size for p, _ in entries)
    if shutil.disk_usage(path.parent).free < uncompressed + 256 * 1024**2:
        raise RuntimeError(f'Insufficient disk reserve to safely create {path.name}. Need {uncompressed / 1024**3:.2f} GiB plus margin. Download existing archives first; no files were deleted.')
    temporary = path.with_suffix('.zip.tmp')
    hashes = {}
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for source, name in entries:
            hashes[name] = file_hash(source)
            z.write(source, name)
        import json
        z.writestr('archive_integrity.json', json.dumps({'created': now(), 'sha256': hashes}, indent=2))
    with zipfile.ZipFile(temporary) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError('Archive CRC check failed: ' + bad)
    temporary.replace(path)
    atomic_json(path.with_suffix('.sha256.json'), {'filename': path.name, 'sha256': file_hash(path), 'bytes': path.stat().st_size})
    print('Verified archive:', path, f'({path.stat().st_size / 1024**3:.3f} GiB)', flush=True)


def main():
    require_kaggle()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(ROOT / 'outputs'))
    parser.add_argument('--archive-dir', default='/kaggle/working')
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--results-only', action='store_true', help='CPU analysis session: do not create a replacement training-resume archive')
    parser.add_argument('--omit-task-models', action='store_true', help='Smaller resume bundle; omits historical task snapshots, retains exact unfinished run state and completed final models')
    args = parser.parse_args()
    output, dest = Path(args.input).resolve(), Path(args.archive_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    if not output.is_dir():
        raise FileNotFoundError(output)
    results = list(scientific_files(output))
    archive(dest / 'session_results_bundle.zip', results)
    states = []
    for run in ([] if args.results_only else sorted(output.glob('tinyimagenet_*'))):
        complete = (run / 'completed.json').exists()
        checkpoints = run / 'checkpoints'
        if not complete and (checkpoints / 'latest.json').exists():
            pointer = read_json(checkpoints / 'latest.json')
            states.append((checkpoints / 'latest.json', 'outputs/' + (checkpoints / 'latest.json').relative_to(output).as_posix()))
            for key in ['current', 'previous']:
                entry = pointer.get(key)
                if entry:
                    path = checkpoints / entry['file']
                    if file_hash(path) != entry['sha256']:
                        raise RuntimeError('Checkpoint integrity failure; see recovery instructions: ' + str(path))
                    states.append((path, 'outputs/' + path.relative_to(output).as_posix()))
        for path in checkpoints.glob('*_model.pt'):
            if args.omit_task_models and path.name.startswith('task_'):
                continue
            states.append((path, 'outputs/' + path.relative_to(output).as_posix()))
    # Metadata + logs also belong in resume: either session ZIP can be inspected,
    # and a resume bundle alone is sufficient to restore every completed record.
    if not args.results_only:
        archive(dest / 'session_resume_bundle.zip', results + states)
    if args.final:
        receipt = ROOT / 'analysis/completion_report.json'
        if not receipt.exists():
            raise RuntimeError('Run final analysis before creating final scientific archive')
        report = read_json(receipt)
        if not report['complete'] or report['runs_complete'] != 12:
            raise RuntimeError('Final archive requires all 12 measured runs and completed analysis')
        if report['protocol_hash'] != protocol()[1] or report['source_version'] != source_version():
            raise RuntimeError('Analysis code or protocol changed; regenerate analysis')
        for rel, expected in report['input_hashes'].items():
            if file_hash(output / rel) != expected:
                raise RuntimeError('Analysis is stale: ' + rel)
        archive(dest / 'FINAL_Neurocomputing_TinyImageNet_NC_Results.zip', results)


if __name__ == '__main__':
    main()
