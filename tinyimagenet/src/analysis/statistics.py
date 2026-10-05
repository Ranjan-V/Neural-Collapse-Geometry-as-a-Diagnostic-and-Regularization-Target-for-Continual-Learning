"""No checkpoint-level independence claims; controls fixed before new results."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

METRICS = ['nc1', 'nc2', 'nc3', 'nc4', 'etf_drift', 'angle_drift']
KEYS = ['dataset', 'method', 'seed', 'config_hash', 'protocol_hash']


def residual(values, design):
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def correlation(x, y, minimum=8):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < minimum:
        return np.nan
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def estimate(frame, metric, target='task1_forgetting', mode='raw', minimum=8):
    needed = [metric, target]
    if mode != 'raw':
        needed += ['global_progress', 'task']
    if not set(needed).issubset(frame):
        return {'n_pairs': 0, 'pearson': np.nan, 'spearman': np.nan, 'reason': 'missing metric or time controls'}
    data = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
    n = len(data)
    x, y = data[metric].to_numpy(float), data[target].to_numpy(float)
    xr, yr = rankdata(x), rankdata(y)
    if mode != 'raw' and n:
        progress = data.global_progress.to_numpy(float)
        design = np.column_stack([np.ones(n), progress])
        rank_design = np.column_stack([np.ones(n), rankdata(progress)])
        if mode == 'partial':
            effects = pd.get_dummies(data.task.astype(int), drop_first=True).to_numpy(float)
            design = np.column_stack([design, effects])
            rank_design = np.column_stack([rank_design, effects])
        if n - np.linalg.matrix_rank(design) < minimum - 2:
            return {'n_pairs': n, 'pearson': np.nan, 'spearman': np.nan, 'reason': 'insufficient residual degrees of freedom'}
        x, y = residual(x, design), residual(y, design)
        if mode == 'partial':
            xr, yr = residual(xr, rank_design), residual(yr, rank_design)
        else:
            xr, yr = rankdata(x), rankdata(y)
    pearson, spearman = correlation(x, y, minimum), correlation(xr, yr, minimum)
    reason = 'ok' if np.isfinite(pearson) and np.isfinite(spearman) else 'too few pairs or constant/degenerate values'
    return {'n_pairs': n, 'pearson': pearson, 'spearman': spearman, 'reason': reason}


def lag_pairs(frame, horizon, task_epochs=100):
    pairs = []
    for task, block in frame.groupby('task', sort=True):
        block = block.sort_values('epoch')
        for _, row in block.iterrows():
            epoch = task_epochs if horizon == 'task_end' else float(row.epoch) + int(horizon)
            if epoch <= row.epoch:
                continue
            future = block[np.isclose(block.epoch, epoch, atol=1e-8, rtol=0)]
            if len(future) == 1:
                pairs.append({**row.to_dict(), 'future_forgetting': float(future.iloc[0].task1_forgetting), 'future_epoch': epoch})
    return pd.DataFrame(pairs)


def block_bootstrap(frame, metric, mode, spec, target='task1_forgetting', minimum=8):
    if metric not in frame or target not in frame:
        return {'lower': np.nan, 'upper': np.nan, 'valid_replicates': 0, 'reason': 'missing values'}
    data = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric, target]).sort_values(['task', 'epoch'])
    length = spec['block_length_checkpoints']
    groups = [g.reset_index(drop=True) for _, g in data.groupby('task', sort=True)]
    if not groups or any(len(g) < 2 * length for g in groups):
        return {'lower': np.nan, 'upper': np.nan, 'valid_replicates': 0, 'reason': 'fewer than two blocks in at least one task; no IID fallback'}
    rng = np.random.default_rng(spec['seed'])
    data = pd.concat(groups, ignore_index=True)
    x, y = data[metric].to_numpy(float), data[target].to_numpy(float)
    progress = data.global_progress.to_numpy(float)
    design = np.column_stack([np.ones(len(data)), progress])
    if mode == 'partial':
        design = np.column_stack([design, pd.get_dummies(data.task.astype(int), drop_first=True).to_numpy(float)])
    offsets = np.cumsum([0] + [len(g) for g in groups])
    values = []
    for _ in range(spec['replicates']):
        selected = []
        for i, g in enumerate(groups):
            starts = rng.integers(0, len(g) - length + 1, size=int(np.ceil(len(g) / length)))
            indices = np.concatenate([np.arange(s, s + length) for s in starts])[:len(g)]
            selected.append(indices + offsets[i])
        indices = np.concatenate(selected)
        xy = np.column_stack([x[indices], y[indices]])
        if mode != 'raw':
            xy = residual(xy, design[indices])
        value = correlation(xy[:, 0], xy[:, 1], minimum)
        if np.isfinite(value):
            values.append(value)
    if len(values) < spec['replicates'] * 0.8:
        return {'lower': np.nan, 'upper': np.nan, 'valid_replicates': len(values), 'reason': 'too many undefined replicates'}
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {'lower': lower, 'upper': upper, 'valid_replicates': len(values), 'reason': 'ok'}


def all_analyses(frame, protocol):
    output = {name: [] for name in ['raw_correlations', 'partial_correlations', 'detrended_correlations', 'within_task_correlations', 'lagged_correlations', 'delta_correlations', 'bootstrap_intervals']}
    minimum = protocol['minimum_pairs']
    for key, run in frame.groupby(KEYS, dropna=False, sort=True):
        meta = dict(zip(KEYS, key))
        run = run[run.task > 0].sort_values(['task', 'epoch'])
        if run.duplicated(['task', 'epoch']).any():
            raise ValueError('Duplicate temporal checkpoint in ' + str(meta))
        for metric in METRICS:
            identity = {**meta, 'geometry_metric': metric}
            for mode, name in [('raw', 'raw_correlations'), ('partial', 'partial_correlations'), ('detrended', 'detrended_correlations')]:
                estimate_row = estimate(run, metric, mode=mode, minimum=minimum)
                output[name].append({**identity, **estimate_row})
                interval = block_bootstrap(run, metric, mode, protocol['bootstrap'], minimum=minimum)
                output['bootstrap_intervals'].append({**identity, 'analysis': mode, 'estimator': 'Pearson', 'confidence': 0.95, 'block_length': protocol['bootstrap']['block_length_checkpoints'], **interval})
            for task, block in run.groupby('task'):
                output['within_task_correlations'].append({**identity, 'task': task, 'variant': 'levels', **estimate(block, metric, minimum=minimum)})
                if metric in block:
                    change = block.copy()
                    change[[metric, 'task1_forgetting']] = change[[metric, 'task1_forgetting']].diff()
                    output['within_task_correlations'].append({**identity, 'task': task, 'variant': 'first_differences', **estimate(change, metric, minimum=minimum)})
            for horizon in [5, 10, 'task_end']:
                paired = lag_pairs(run, horizon)
                raw = estimate(paired, metric, 'future_forgetting', minimum=minimum)
                partial = estimate(paired, metric, 'future_forgetting', mode='partial', minimum=minimum)
                output['lagged_correlations'].append({**identity, 'horizon': str(horizon), **raw, 'partial_pearson': partial['pearson'], 'partial_spearman': partial['spearman'], 'partial_reason': partial['reason']})
            deltas = []
            if metric in run:
                for _, block in run.groupby('task'):
                    block = block.sort_values('epoch').copy()
                    valid = np.isclose(block.epoch.diff(), 5) & np.isclose(block.epoch.shift(-1) - block.epoch, 5)
                    block['delta_geometry'] = block[metric].diff()
                    block['delta_future_forgetting'] = block.task1_forgetting.shift(-1) - block.task1_forgetting
                    deltas.append(block[valid])
            changed = pd.concat(deltas, ignore_index=True) if deltas else pd.DataFrame()
            output['delta_correlations'].append({**identity, 'label': 'EXPLORATORY', **estimate(changed, 'delta_geometry', 'delta_future_forgetting', minimum=minimum)})
    return {k: pd.DataFrame(v) for k, v in output.items()}
