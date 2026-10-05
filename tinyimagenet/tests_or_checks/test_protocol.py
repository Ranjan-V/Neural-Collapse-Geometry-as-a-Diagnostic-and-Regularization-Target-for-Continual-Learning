import numpy as np
import pandas as pd
import pytest

from utils.runtime import protocol, config_for, jobs, bind_json
from data.tinyimagenet import make_split
from analysis.statistics import lag_pairs, estimate, block_bootstrap


def test_jobs_and_fixed_hyperparameters():
    assert len({j['run_id'] for j in jobs()}) == 12
    assert {j['method'] for j in jobs()} == {'finetune', 'lwf', 'etfs', 'lwf_etf'}
    for method in ['etfs', 'lwf_etf']:
        cfg, _, _ = config_for(method)
        assert (cfg['batch_size'], cfg['lambda_angle'], cfg['anchor_every_n_steps']) == (128, 1.0, 4)
        assert cfg['alpha'] == (1.0 if method == 'etfs' else 0.25)


def test_split_integrity_and_manifest_refusal(tmp_path):
    index = {'classes': [f'class{i:03}' for i in range(200)], 'fingerprint': 'fixture'}
    for seed in [42, 43, 44]:
        split = make_split(index, seed)
        assert len(set(split['class_order'])) == 200
        assert all(len(t['class_ids']) == 20 for t in split['tasks'])
        bind_json(tmp_path / f'{seed}.json', split)
        bind_json(tmp_path / f'{seed}.json', make_split(index, seed))
        with pytest.raises(RuntimeError):
            bind_json(tmp_path / f'{seed}.json', make_split(index, seed + 1))


def test_no_interpolation_or_boundary_crossing():
    frame = pd.DataFrame({'task': [1, 1, 1, 2], 'epoch': [90, 95, 100, 0], 'task1_forgetting': [.1, .2, .3, .4]})
    paired = lag_pairs(frame, 5)
    assert list(paired.epoch) == [90, 95]
    assert list(paired.future_forgetting) == [.2, .3]
    assert len(lag_pairs(frame, 'task_end')) == 2
    assert lag_pairs(frame, 7).empty


def test_time_only_association_is_not_reported_as_partial_signal():
    progress = np.arange(40, dtype=float)
    data = pd.DataFrame({'task': np.repeat([1, 2], 20), 'epoch': np.tile(np.arange(20), 2), 'global_progress': progress / 40, 'g': progress, 'task1_forgetting': progress * 2})
    assert estimate(data, 'g')['pearson'] == pytest.approx(1)
    assert np.isnan(estimate(data, 'g', mode='partial')['pearson'])


def test_constant_and_short_series_are_explicitly_unavailable():
    data = pd.DataFrame({'task': [1] * 5, 'epoch': range(5), 'global_progress': np.arange(5) / 5, 'g': [1] * 5, 'task1_forgetting': range(5)})
    assert estimate(data, 'g')['reason'] != 'ok'
    result = block_bootstrap(data, 'g', 'raw', protocol()[0]['bootstrap'])
    assert result['valid_replicates'] == 0
