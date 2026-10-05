"""Kaggle-only end-to-end and epoch-boundary resume checks for all four methods."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from utils.runtime import require_kaggle, now, atomic_json, read_json, protocol, source_version


def main():
    require_kaggle()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    args = parser.parse_args()
    import pandas as pd
    import torch
    from data.tinyimagenet import prepare
    from training.checkpoint import load_state
    stamp = now().replace(':', '-')
    output = ROOT / 'smoke' / stamp
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for method in ['finetune', 'lwf', 'etfs', 'lwf_etf']:
        straight, resumed = output / method / 'straight', output / method / 'resumed'
        for target in [straight, resumed]:
            prepare(args.data_root, target)
        common = [sys.executable, str(ROOT / 'run_experiment.py'), '--method', method, '--seed', '42', '--smoke', '--resume', 'auto', '--data-root', args.data_root]
        subprocess.run(common + ['--output', str(straight)], env=env, check=True)
        subprocess.run(common + ['--output', str(resumed), '--stop-after-epochs', '3'], env=env, check=True)
        rid = f'tinyimagenet_{method}_seed42'
        saved = load_state(resumed / rid, read_json(resumed / rid / 'identity.json'))
        assert saved['task'] == 1 and saved['epoch'] == 1
        if method in {'lwf', 'lwf_etf'}:
            assert saved['method_state']['old_num_classes'] == 2
            assert saved['method_state']['teacher']
        if method in {'etfs', 'lwf_etf'}:
            assert saved['method_state']['anchor']['images']
        subprocess.run(common + ['--output', str(resumed)], env=env, check=True)
        a = torch.load(straight / rid / 'checkpoints/final_model.pt', map_location='cpu', weights_only=True)
        b = torch.load(resumed / rid / 'checkpoints/final_model.pt', map_location='cpu', weights_only=True)
        for key in a['model_state_dict']:
            torch.testing.assert_close(a['model_state_dict'][key], b['model_state_dict'][key], rtol=1e-6, atol=1e-7)
        columns = ['task', 'epoch', 'nc1', 'nc2', 'nc3', 'nc4', 'etf_drift', 'angle_drift', 'task1_accuracy', 'task1_forgetting']
        df1 = pd.read_csv(straight / rid / 'diagnostics/temporal_diagnostics.csv')[columns]
        df2 = pd.read_csv(resumed / rid / 'diagnostics/temporal_diagnostics.csv')[columns]
        pd.testing.assert_frame_equal(df1, df2, check_exact=False, rtol=1e-6, atol=1e-7)
        atomic_json(output / f'{method}_resume_check.json', {'method': method, 'passed': True, 'tolerance': 'rtol=1e-6, atol=1e-7', 'timestamp': now()})
        print('SMOKE AND RESUME PASSED:', method, flush=True)
    atomic_json(ROOT / 'smoke/latest_pass.json', {'path': str(output), 'all_four_methods': True, 'timestamp': now(), 'protocol_hash': protocol()[1], 'source_version': source_version()})


if __name__ == '__main__':
    main()
