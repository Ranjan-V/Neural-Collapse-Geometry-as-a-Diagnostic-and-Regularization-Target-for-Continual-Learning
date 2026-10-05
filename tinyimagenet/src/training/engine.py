"""Reuse scientific losses; transact training at epoch and task boundaries."""
from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch.amp import GradScaler, autocast

from data.tinyimagenet import loader, make_split, prepare, dataset_index
from diagnostics.collect import diagnostic, evaluate
from evaluation.evaluator import summarize_accuracy_matrix, compute_forgetting
from methods.finetune import SimpleFinetuner
from methods.lwf import LwF
from methods.etf_anchor import ETFAnchor
from methods.lwf_etf import LwFETFAnchor
from models.resnet import build_resnet18
from training.checkpoint import rng_state, restore_rng, method_state, restore_method, anchor_of, save_state, load_state, save_model_snapshot
from utils.runtime import atomic_json, bind_json, config_for, csv_rows, digest, now, read_json, source_version

METHODS = {'finetune': SimpleFinetuner, 'lwf': LwF, 'etfs': ETFAnchor, 'lwf_etf': LwFETFAnchor}


def environment():
    versions = {p: importlib.metadata.version(p) for p in ['torch', 'torchvision', 'numpy', 'Pillow']}
    return {'packages': versions, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'gpu_name': torch.cuda.get_device_name(0)}


def configure(seed, cfg):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cfg['torch_cpu_threads'])
    torch.backends.cudnn.deterministic = cfg['deterministic']
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = cfg['allow_tf32']
    torch.backends.cudnn.allow_tf32 = cfg['allow_tf32']
    torch.use_deterministic_algorithms(cfg['deterministic'], warn_only=cfg['deterministic_algorithms_warn_only'])


def optimizers(model, cfg, steps):
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['learning_rate'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs_per_task'] * steps)
    return optimizer, scheduler


def fresh_state(identity, seed, cfg, split):
    return {'format_version': 1, 'identity': identity, 'seed': seed, 'task': 0, 'epoch': 0, 'global_step': 0,
            'phase': 'training', 'generation': 0, 'split': split, 'config': cfg,
            'accuracy_matrix': [[None] * cfg['num_tasks'] for _ in range(cfg['num_tasks'])],
            'diagnostics': [], 'resources': [], 'task_metrics': [], 'continual_metrics': [],
            'reference_means': None, 'reference_accuracy': None, 'journal_bytes': 0, 'active_wall_seconds': 0.0,
            'task_wall_seconds': 0.0, 'losses': {}, 'environment': environment()}


def publish(run, state, status, checkpoint=None):
    csv_rows(run / 'diagnostics/temporal_diagnostics.csv', state['diagnostics'])
    csv_rows(run / 'system/resource_usage.csv', state['resources'])
    csv_rows(run / 'metrics/task_metrics.csv', state['task_metrics'])
    csv_rows(run / 'metrics/continual_metrics.csv', state['continual_metrics'])
    rows = [{'trained_task': i, **{f'task_{j}': value for j, value in enumerate(row)}} for i, row in enumerate(state['accuracy_matrix'])]
    csv_rows(run / 'metrics/accuracy_matrix.csv', rows)
    atomic_json(run / 'status.json', {**state['identity'], 'status': status, 'task': state['task'], 'epoch': state['epoch'], 'phase': state['phase'], 'global_step': state['global_step'], 'checkpoint': checkpoint, 'updated': now()})


def train(args):
    process_start = time.monotonic()
    cfg, ph, ch = config_for(args.method, args.smoke)
    configure(args.seed, cfg)
    output = Path(args.output).resolve()
    if args.smoke and 'smoke' not in output.parts:
        raise ValueError('Smoke output must be under smoke/')
    if (output / 'dataset_identity.json').exists():
        index = dataset_index(args.data_root)
        if read_json(output / 'dataset_identity.json')['fingerprint'] != index['fingerprint']:
            raise RuntimeError('Dataset inventory changed since protocol binding')
        if read_json(output / 'protocol_frozen.json')['protocol_hash'] != ph:
            raise RuntimeError('Frozen protocol mismatch')
    else:
        index = prepare(args.data_root, output)
    split = make_split(index, args.seed, cfg['num_tasks'], cfg['classes_per_task'])
    if args.smoke:
        bind_json(output / 'smoke_splits' / f'seed_{args.seed}.json', split)
    elif split != read_json(output / 'splits' / f'seed_{args.seed}.json'):
        raise RuntimeError('Bound split mismatch')
    run_id = f'tinyimagenet_{args.method}_seed{args.seed}'
    run = output / run_id
    identity = {'run_id': run_id, 'dataset': 'tinyimagenet', 'method': args.method, 'seed': args.seed,
                'protocol_hash': ph, 'config_hash': ch, 'split_hash': digest(split),
                'source_version': source_version(), 'smoke': args.smoke}
    completed = run / 'completed.json'
    if completed.exists():
        previous = read_json(completed)
        if any(previous.get(k) != v for k, v in identity.items()):
            raise RuntimeError('Completed run has incompatible provenance')
        print(run_id, 'already completed; unchanged', flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    bind_json(run / 'identity.json', identity)
    bind_json(run / 'config.json', cfg)
    bind_json(run / 'split.json', split)
    resumed = (run / 'checkpoints/latest.json').exists()
    if resumed and args.resume != 'auto':
        raise RuntimeError('Run already exists; use --resume auto')
    state = load_state(run, identity) if resumed else fresh_state(identity, args.seed, cfg, split)
    prior_wall_seconds = state.get('run_wall_seconds', 0.0)
    if state['environment'] != environment():
        raise RuntimeError('Software/CUDA/GPU environment changed. Restore the recorded environment before resuming.')
    atomic_json(run / 'system/environment.json', {**state['environment'], 'gpu_assignment': os.getenv('CUDA_VISIBLE_DEVICES'), 'python': os.sys.version, 'cudnn_deterministic': cfg['deterministic'], 'benchmark': False, 'tf32': cfg['allow_tf32'], 'warning_only_determinism': cfg['deterministic_algorithms_warn_only']})
    device = torch.device(args.device)
    model = build_resnet18(cfg['feature_dim'], (state['task'] + 1) * cfg['classes_per_task']).to(device)
    method = METHODS[args.method]()
    train_loader = loader(index, split, state['task'], cfg, args.seed, state['epoch'], training=True)
    optimizer, scheduler = optimizers(model, cfg, len(train_loader))
    scaler = GradScaler('cuda', enabled=cfg['mixed_precision'])
    if resumed:
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        scaler.load_state_dict(state['scaler_state_dict'])
        restore_method(method, state['method_state'], cfg, device)
        restore_rng(state['rng'])
    else:
        method.before_task(model, 0, cfg)
    journal = run / 'logs/steps.jsonl'
    journal.parent.mkdir(parents=True, exist_ok=True)
    if journal.exists():
        if journal.stat().st_size < state['journal_bytes']:
            raise RuntimeError('Training journal shorter than committed checkpoint; restore full resume bundle')
        with journal.open('r+b') as f:
            f.truncate(state['journal_bytes'])
    elif state['journal_bytes']:
        raise RuntimeError('Missing committed training journal')
    checkpoint_usage = run / 'system/checkpoint_usage.jsonl'
    if resumed and checkpoint_usage.exists():
        committed_usage = [line for line in checkpoint_usage.read_text().splitlines() if json.loads(line)['generation'] <= state['generation']]
        checkpoint_usage.write_text('\n'.join(committed_usage) + ('\n' if committed_usage else ''), encoding='utf-8')
    stop = {'requested': False}
    def request_stop(*_):
        stop['requested'] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    def stopped():
        return stop['requested'] or (args.stop_file and Path(args.stop_file).exists()) or (args.stop_after_epochs is not None and epochs_this_process >= args.stop_after_epochs)
    epochs_this_process = 0
    def commit(status='running'):
        checkpoint_start = time.monotonic()
        state['run_wall_seconds'] = prior_wall_seconds + time.monotonic() - process_start
        if state['resources']:
            state['resources'][-1]['run_wall_seconds'] = state['run_wall_seconds']
        state.update(model_state_dict=model.state_dict(), classifier_state_dict=model.classifier.state_dict(),
                     optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                     scaler_state_dict=scaler.state_dict(), method_state=method_state(method), rng=rng_state())
        state['generation'] += 1
        state['journal_bytes'] = journal.stat().st_size if journal.exists() else 0
        entry = save_state(run, state)
        publish(run, state, status, entry)
        usage = run / 'system/checkpoint_usage.jsonl'
        with usage.open('a', encoding='utf-8') as f:
            f.write(json.dumps({**state['identity'], **entry, 'checkpoint_write_seconds': time.monotonic() - checkpoint_start}) + '\n')
        return entry
    if not resumed:
        commit()
    while True:
        if stopped():
            commit('paused')
            print(run_id, 'paused at committed epoch', state['task'], state['epoch'], flush=True)
            return
        if state['phase'] in {'between_tasks', 'complete'}:
            if state['task'] == cfg['num_tasks'] - 1:
                summary = summarize_accuracy_matrix(np.array(state['accuracy_matrix'], dtype=float))
                save_model_snapshot(run / 'checkpoints/final_model.pt', model, {**identity, 'task': state['task'], 'epoch': state['epoch']})
                state['phase'] = 'complete'
                entry = commit('complete')
                atomic_json(completed, {**identity, **summary, 'tasks_completed': cfg['num_tasks'], 'diagnostic_rows': len(state['diagnostics']), 'finished': now(), 'final_checkpoint': 'checkpoints/final_model.pt', 'active_wall_seconds': state['active_wall_seconds'], 'run_wall_seconds': prior_wall_seconds + time.monotonic() - process_start, 'checkpoint': entry, 'full_state_retained': False})
                # Completed jobs will never resume training; retain task/final models,
                # and release large redundant optimizer/teacher/buffer generations.
                for checkpoint in (run / 'checkpoints').glob('state_*.pt'):
                    checkpoint.unlink()
                (run / 'checkpoints/latest.json').unlink(missing_ok=True)
                print(run_id, 'COMPLETE', summary, flush=True)
                return
            state['task'] += 1
            state['epoch'] = 0
            state['task_wall_seconds'] = 0.0
            # Cached feature tensors are not scientific state and must not be deep-copied.
            model.features = None
            method.before_task(model, state['task'], cfg)
            model.classifier.expand(cfg['classes_per_task'], freeze_old=False)
            model.to(device)
            train_loader = loader(index, split, state['task'], cfg, args.seed, 0, training=True)
            optimizer, scheduler = optimizers(model, cfg, len(train_loader))
            state['phase'] = 'training'
            diagnostic(model, index, split, state, cfg, device, {})
            commit()
        if state['epoch'] == cfg['epochs_per_task']:
            # Full epoch checkpoint already committed before this transition.
            begin = time.monotonic()
            model.features = None
            method.after_task(model, loader(index, split, state['task'], cfg, args.seed, cfg['epochs_per_task'], training=True), state['task'], cfg, device)
            for t in range(state['task'] + 1):
                accuracy = evaluate(model, loader(index, split, t, cfg, args.seed), t, cfg, device)['accuracy']
                state['accuracy_matrix'][state['task']][t] = accuracy
                state['task_metrics'].append({**identity, 'timestamp': now(), 'checkpoint_id': f"task{state['task']}_end", 'trained_task': state['task'], 'evaluated_task': t, 'accuracy': accuracy})
            matrix = np.array(state['accuracy_matrix'], dtype=float)[:state['task'] + 1, :state['task'] + 1]
            summary = summarize_accuracy_matrix(matrix)
            state['continual_metrics'].append({**identity, 'timestamp': now(), 'task': state['task'], **summary, 'per_task_forgetting': json.dumps(compute_forgetting(matrix)['per_task']), 'best_previous_accuracies': json.dumps(np.nanmax(matrix, axis=0).tolist())})
            elapsed = time.monotonic() - begin
            state['active_wall_seconds'] += elapsed
            state['task_wall_seconds'] += elapsed
            state['phase'] = 'between_tasks'
            save_model_snapshot(run / 'checkpoints' / f"task_{state['task']}_model.pt", model, {**identity, 'task': state['task'], 'epoch': state['epoch']})
            commit()
            continue
        model.train()
        train_loader = loader(index, split, state['task'], cfg, args.seed, state['epoch'], training=True)
        begin = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        totals = {}
        seen = 0
        with journal.open('a', encoding='utf-8') as f:
            for batch_index, batch in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)
                method.current_epoch, method.current_batch_index, method.current_global_step = state['epoch'], batch_index, state['global_step']
                with autocast('cuda', enabled=cfg['mixed_precision']):
                    result = method.compute_loss(model, batch, state['task'], cfg, device)
                if not torch.isfinite(result.loss) or not all(math.isfinite(v) for v in result.logs.values()):
                    raise FloatingPointError(f"Nonfinite loss at task {state['task']} epoch {state['epoch']} step {state['global_step']}")
                scaler.scale(result.loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip_max_norm'], error_if_nonfinite=not cfg['mixed_precision'])
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < old_scale
                if not torch.isfinite(grad_norm) and not skipped:
                    raise FloatingPointError('Nonfinite gradient without an AMP skipped update')
                if skipped:
                    print(f"AMP overflow: skipped step {state['global_step']}; scale {old_scale} -> {scaler.get_scale()}", flush=True)
                if not skipped:
                    scheduler.step()
                anchor = anchor_of(method)
                counts = {k: len(v) for k, v in anchor.exemplar_buffer.images.items()} if anchor is not None and anchor.exemplar_buffer else {}
                anchor_due = bool(counts) and state['global_step'] % cfg['anchor_every_n_steps'] == 0
                record = {**identity, 'timestamp': now(), 'checkpoint_id': f"task{state['task']}_epoch{state['epoch'] + 1}",
                          'task': state['task'], 'epoch': state['epoch'], 'global_step': state['global_step'],
                          'grad_norm': float(grad_norm) if torch.isfinite(grad_norm) else None, 'optimizer_skipped': skipped, 'anchor_update': anchor_due,
                          'buffer_class_counts': counts, 'lr': optimizer.param_groups[0]['lr'], **result.logs}
                f.write(json.dumps(record, allow_nan=False) + '\n')
                batch_size = len(batch[1])
                seen += batch_size
                for key, value in result.logs.items():
                    totals[key] = totals.get(key, 0.0) + value * batch_size
                state['global_step'] += 1
            f.flush()
            os.fsync(f.fileno())
        torch.cuda.synchronize(device)
        training_seconds = time.monotonic() - begin
        state['epoch'] += 1
        state['losses'] = {k: v / seen for k, v in totals.items()}
        if (state['task'] > 0 and state['epoch'] % cfg['diagnostic_every_epochs'] == 0) or state['epoch'] == cfg['epochs_per_task']:
            diagnostic(model, index, split, state, cfg, device, state['losses'])
        epoch_seconds = time.monotonic() - begin
        state['active_wall_seconds'] += epoch_seconds
        state['task_wall_seconds'] += epoch_seconds
        props = torch.cuda.get_device_properties(device)
        state['resources'].append({**identity, 'timestamp': now(), 'checkpoint_id': f"task{state['task']}_epoch{state['epoch']}", 'task': state['task'], 'epoch': state['epoch'],
                                   'gpu_name': props.name, 'gpu_id': os.getenv('CUDA_VISIBLE_DEVICES', args.device), 'total_vram_bytes': props.total_memory,
                                   'peak_vram_bytes': torch.cuda.max_memory_allocated(device), 'training_seconds': training_seconds,
                                   'epoch_seconds': epoch_seconds, 'task_wall_seconds': state['task_wall_seconds'], 'run_wall_seconds': state['active_wall_seconds'],
                                   'images_per_second': seen / training_seconds, **state['losses']})
        entry = commit()
        # The committed checkpoint size is independently recorded outside its own payload.
        atomic_json(run / 'system/latest_checkpoint_size.json', entry)
        epochs_this_process += 1
        print(f"{run_id}: task {state['task'] + 1}/{cfg['num_tasks']} epoch {state['epoch']}/{cfg['epochs_per_task']} loss={state['losses']['loss']:.5g} train={training_seconds:.1f}s", flush=True)
