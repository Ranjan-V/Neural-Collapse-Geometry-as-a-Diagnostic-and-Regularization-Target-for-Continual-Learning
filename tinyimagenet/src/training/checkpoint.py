"""Complete, checksummed checkpoint generations with an atomic commit pointer."""
from __future__ import annotations

import os
from pathlib import Path
import random

import numpy as np
import torch

from methods.etf_anchor import ExemplarBuffer, MeanBuffer
from models.resnet import build_resnet18
from utils.runtime import atomic_json, file_hash, read_json, now


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state_all(state['cuda'])


def anchor_of(method):
    return getattr(method, 'anchor', method) if hasattr(method, 'frozen_means') else None


def method_state(method):
    state = {}
    teacher = getattr(method, 'teacher', None)
    if teacher is not None:
        state.update(teacher=teacher.state_dict(), old_num_classes=method.old_num_classes)
    anchor = anchor_of(method)
    if anchor is not None:
        state['anchor'] = {
            'frozen_means': anchor.frozen_means, 'frozen_etf_matrix': anchor.frozen_etf_matrix,
            'images': {k: torch.stack(list(v)) for k, v in anchor.exemplar_buffer.images.items()} if anchor.exemplar_buffer else None,
            'features': {k: torch.stack(list(v)) for k, v in anchor.mean_buffer.storage.items()} if anchor.mean_buffer else None,
        }
    return state


def restore_method(method, state, cfg, device):
    if 'teacher' in state:
        method.old_num_classes = state['old_num_classes']
        method.teacher = build_resnet18(cfg['feature_dim'], method.old_num_classes).to(device)
        method.teacher.load_state_dict(state['teacher'])
        method.teacher.eval().requires_grad_(False)
    a = state.get('anchor')
    if a is not None:
        anchor = anchor_of(method)
        anchor.frozen_means = a['frozen_means']
        anchor.frozen_etf_matrix = a['frozen_etf_matrix']
        if a['images'] is not None:
            anchor.exemplar_buffer = ExemplarBuffer(cfg['buffer_size'])
            for k, images in a['images'].items():
                anchor.exemplar_buffer.add_batch(images, torch.full((len(images),), k, dtype=torch.long))
        if a['features'] is not None:
            anchor.mean_buffer = MeanBuffer(cfg['buffer_size'])
            for k, features in a['features'].items():
                anchor.mean_buffer.update(features, torch.full((len(features),), k, dtype=torch.long))


def save_state(run, state):
    directory = Path(run) / 'checkpoints'
    directory.mkdir(parents=True, exist_ok=True)
    generation = int(state['generation'])
    name = f'state_{generation:06d}.pt'
    tmp, dest = directory / (name + '.tmp'), directory / name
    with tmp.open('wb') as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dest)
    entry = {'file': name, 'sha256': file_hash(dest), 'generation': generation, 'phase': state['phase'], 'task': state['task'], 'epoch': state['epoch'], 'timestamp': now(), 'size_bytes': dest.stat().st_size}
    pointer = directory / 'latest.json'
    old = read_json(pointer) if pointer.exists() else None
    previous = None
    if old:
        previous = next((p for p in [old['current'], old.get('previous')] if p and p['generation'] < generation), None)
    atomic_json(pointer, {'current': entry, 'previous': previous})
    # Keep two committed generations; task snapshots are separate and lightweight.
    keep = {name, previous['file'] if previous else name}
    for p in directory.glob('state_*.pt'):
        if p.name not in keep:
            p.unlink()
    return entry


def load_state(run, expected):
    directory = Path(run) / 'checkpoints'
    pointer = read_json(directory / 'latest.json')
    errors = []
    for entry in [pointer['current'], pointer.get('previous')]:
        if not entry:
            continue
        path = directory / entry['file']
        try:
            if file_hash(path) != entry['sha256']:
                raise ValueError('SHA256 mismatch')
            # Only load self-created/trusted experiment archives; RNG includes NumPy state.
            state = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as exc:
            errors.append({'file': str(path), 'error': str(exc)})
            continue
        for k, v in expected.items():
            if state['identity'].get(k) != v:
                raise RuntimeError(f'Checkpoint {k} mismatch. Refusing incompatible resume.')
        if errors:
            atomic_json(Path(run) / 'recovery_notice.json', {'errors': errors, 'recovered': str(path), 'timestamp': now()})
            print('CHECKPOINT RECOVERY:', errors, 'using', path, flush=True)
        return state
    raise RuntimeError('No valid checkpoint generation: ' + str(errors))


def save_model_snapshot(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        torch.save({'model_state_dict': model.state_dict(), 'num_classes': model.classifier.out_features, **metadata}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
