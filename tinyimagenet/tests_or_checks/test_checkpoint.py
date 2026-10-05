import torch
import pytest
from training.checkpoint import save_state, load_state
from utils.runtime import read_json


def test_checksum_fallback_keeps_a_valid_previous_generation(tmp_path):
    identity = {'seed': 42, 'protocol_hash': 'fixture'}
    def state(generation):
        return {'identity': identity, 'generation': generation, 'phase': 'training', 'task': 1, 'epoch': generation, 'value': torch.tensor([generation])}
    save_state(tmp_path, state(1))
    save_state(tmp_path, state(2))
    pointer = read_json(tmp_path / 'checkpoints/latest.json')
    (tmp_path / 'checkpoints' / pointer['current']['file']).write_bytes(b'broken test checkpoint')
    recovered = load_state(tmp_path, identity)
    assert recovered['generation'] == 1
    assert (tmp_path / 'recovery_notice.json').exists()
    save_state(tmp_path, state(2))
    pointer = read_json(tmp_path / 'checkpoints/latest.json')
    assert pointer['previous']['generation'] == 1
    assert load_state(tmp_path, identity)['generation'] == 2
    with pytest.raises(RuntimeError, match='mismatch'):
        load_state(tmp_path, {'seed': 43})
