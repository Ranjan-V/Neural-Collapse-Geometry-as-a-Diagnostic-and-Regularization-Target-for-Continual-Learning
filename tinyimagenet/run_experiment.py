"""One single-GPU job. Scientific execution is Kaggle-only."""
import argparse
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from utils.runtime import require_kaggle, atomic_json, read_json, now


def main():
    require_kaggle()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['tinyimagenet'], default='tinyimagenet')
    parser.add_argument('--method', choices=['finetune', 'lwf', 'etfs', 'lwf_etf'], required=True)
    parser.add_argument('--seed', type=int, choices=[42, 43, 44], required=True)
    parser.add_argument('--device', choices=['cuda:0'], default='cuda:0')
    parser.add_argument('--resume', choices=['auto', 'never'], default='auto')
    parser.add_argument('--data-root')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--output', default=str(ROOT / 'outputs'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stop-file')
    parser.add_argument('--stop-after-epochs', type=int, help='Engineering pause; does not shorten the scientific run')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.smoke and Path(args.output) == ROOT / 'outputs':
        args.output = str(ROOT / 'smoke' / 'outputs')
    output = Path(args.output)
    run_id = f'tinyimagenet_{args.method}_seed{args.seed}'
    try:
        import fcntl
        (output / run_id).mkdir(parents=True, exist_ok=True)
        run_lock = (output / run_id / '.run.lock').open('w')
        try:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This run is already active in another process')
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Select GPU T4 x2 in Kaggle.')
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Worker must see one GPU. Set CUDA_VISIBLE_DEVICES=0 or use the dual-GPU scheduler.')
        from data.tinyimagenet import locate_dataset
        args.data_root = str(locate_dataset(args.data_root, args.allow_download))
        from training.engine import train
        train(args)
    except Exception as exc:
        status_file = output / run_id / 'status.json'
        status = read_json(status_file) if status_file.exists() else {}
        atomic_json(output / 'failures' / f'{run_id}_{now().replace(":", "-")}.json', {
            'run_id': run_id, 'error': str(exc), 'traceback': traceback.format_exc(),
            'task': status.get('task'), 'epoch': status.get('epoch'), 'gpu': os.getenv('CUDA_VISIBLE_DEVICES'),
            'last_valid_checkpoint': status.get('checkpoint'), 'timestamp': now(), 'failure_position': 'see traceback; status is last committed epoch'})
        raise


if __name__ == '__main__':
    main()
