"""Import authentic prior logs, never synthesize missing temporal observations."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from utils.runtime import read_json, file_hash, digest


def load_legacy(root):
    root = Path(root)
    frames, report = [], []
    catalog = read_json(root / 'legacy_inputs/catalog.json')
    for item in catalog:
        directory = root / item['relative_dir']
        path = directory / 'nc_metrics.csv'
        if file_hash(path).lower() != item['nc_sha256'].lower():
            raise ValueError('Legacy log hash mismatch: ' + str(path))
        summary = read_json(directory / 'summary.json')
        raw = pd.read_csv(path)
        if 'task_id' not in raw or 'trained_task_id' not in raw:
            report.append({**item, 'availability': 'unavailable: missing task identifiers'})
            continue
        data = raw[raw.task_id == 0].copy()
        matrix = np.array(summary['accuracy_matrix'], dtype=float)
        rows = []
        for _, row in data.iterrows():
            task = int(row.trained_task_id)
            if task == 0:
                continue
            observed = pd.notna(row.get('task1_accuracy')) and pd.notna(row.get('task1_forgetting'))
            if observed:
                # Legacy callback was called after step with zero-based global_step.
                epoch = (float(row.step) + 1 - task * item['epochs_per_task'] * item['steps_per_epoch']) / item['steps_per_epoch']
                if not 0 <= epoch <= item['epochs_per_task']:
                    raise ValueError('Legacy epoch reconstruction outside task; verify config: ' + str(path))
                accuracy, forgetting = float(row.task1_accuracy), float(row.task1_forgetting)
            else:
                # A real end-of-task NC row can be matched to the stored accuracy matrix.
                if int(row.step) != task:
                    continue
                epoch = item['epochs_per_task']
                accuracy = matrix[task, 0]
                forgetting = matrix[0, 0] - accuracy
            rows.append({**row.to_dict(), 'task': task, 'epoch': epoch,
                         'global_progress': (task + epoch / item['epochs_per_task']) / item['num_tasks'],
                         'task1_accuracy': accuracy, 'task1_forgetting': forgetting,
                         'checkpoint_id': f"legacy_task{task}_step{row.step}",
                         'time_basis': 'reconstructed_from_logged_global_step' if observed else 'observed_task_end'})
        frame = pd.DataFrame(rows)
        method = item['method']
        if method == 'etf_anchor':
            method = 'etfs' if item['experiment_name'] in {'kaggle_etf_sparse', 'kaggle_cifar10_full'} else 'etf_dense_historical'
        group_id = item['experiment_name'] + ':' + method
        config_path = root / item['config_snapshot']
        if not config_path.exists():
            raise FileNotFoundError('Missing audited config snapshot: ' + str(config_path))
        if not frame.empty:
            frame['dataset'] = item['dataset']
            frame['method'] = method
            frame['seed'] = int(summary['seed'])
            frame['config_hash'] = digest({'snapshot_sha256': file_hash(config_path), 'method': method})
            frame['protocol_hash'] = 'historical:' + group_id
            frame['source_version'] = 'historical-version-not-recorded'
            frame['provenance_level'] = item['provenance']
            frame['run_id'] = f"{item['dataset']}_{method}_seed{summary['seed']}"
            frame['smoke'] = False
            frame['reference_definition'] = 'historical training-anchor means when available; not new validation reference'
            frame['checkpoint_index'] = np.arange(len(frame))
            frames.append(frame)
        report.append({**item, 'seed': summary['seed'], 'temporal_rows': len(frame), 'availability': 'actual logs imported; exact +5/+10 pairs may be unavailable; no interpolation', 'reference_warning': 'Historical centroid drift uses training anchors; baseline drift may be absent. NC2 and angle_drift duplicate when both exist.'})
    return frames, report
