from __future__ import annotations

import json
import torch

from data.tinyimagenet import loader
from metrics.nc_metrics import compute_all_nc_metrics, compute_class_means, compute_etf_drift, compute_angle_drift
from models.classifier import get_task_logits
from utils.runtime import now


@torch.no_grad()
def evaluate(model, data, task, cfg, device, collect=False):
    model.eval()
    correct, count = 0, 0
    features, labels, logits = [], [], []
    for x, y in data:
        x, y = x.to(device), y.to(device)
        scores, h = model(x)
        scores = get_task_logits(scores, task, cfg['classes_per_task'])
        if not torch.isfinite(scores).all() or not torch.isfinite(h).all():
            raise FloatingPointError('Nonfinite evaluation features/logits')
        correct += int((scores.argmax(1) == y).sum())
        count += len(y)
        if collect:
            features.append(h.float().cpu())
            labels.append(y.cpu())
            logits.append(scores.float().cpu())
    if not count:
        raise ValueError('Empty evaluation set')
    result = {'accuracy': correct / count}
    if collect:
        result.update(features=torch.cat(features), labels=torch.cat(labels), logits=torch.cat(logits))
    return result


def diagnostic(model, index, split, state, cfg, device, losses):
    task, epoch = state['task'], state['epoch']
    task1 = evaluate(model, loader(index, split, 0, cfg, state['seed']), 0, cfg, device, collect=True)
    means = compute_class_means(task1['features'], task1['labels'])
    if state['reference_means'] is None:
        if task != 0 or epoch != cfg['epochs_per_task']:
            raise ValueError('Task-1 reference must be established at task-1 completion')
        state['reference_means'] = means
        state['reference_accuracy'] = task1['accuracy']
    metrics = compute_all_nc_metrics(task1['features'], task1['labels'], model.classifier.weight[:cfg['classes_per_task']].detach().cpu(), task1['logits'])
    metrics['etf_drift'] = float(compute_etf_drift(means, state['reference_means']))
    metrics['angle_drift'] = float(compute_angle_drift(means, state['reference_means']))
    if not all(torch.isfinite(torch.tensor(v)) for v in metrics.values()):
        raise FloatingPointError('Nonfinite NC diagnostic')
    accuracies = [task1['accuracy']]
    for t in range(1, task + 1):
        accuracies.append(evaluate(model, loader(index, split, t, cfg, state['seed']), t, cfg, device)['accuracy'])
    row = {
        **state['identity'], 'timestamp': now(), 'checkpoint_id': f"task{task}_epoch{epoch}",
        'task': task, 'epoch': epoch, 'within_task_progress': epoch / cfg['epochs_per_task'],
        'global_progress': (task + epoch / cfg['epochs_per_task']) / cfg['num_tasks'],
        'checkpoint_index': len(state['diagnostics']), 'global_step': state['global_step'],
        'task1_accuracy': task1['accuracy'], 'task1_forgetting': state['reference_accuracy'] - task1['accuracy'],
        'current_task_accuracy': accuracies[-1], 'old_task_accuracies': json.dumps(accuracies[:-1]),
        **metrics, **{k: v for k, v in losses.items() if k.endswith('loss')},
    }
    state['diagnostics'].append(row)
    return accuracies
