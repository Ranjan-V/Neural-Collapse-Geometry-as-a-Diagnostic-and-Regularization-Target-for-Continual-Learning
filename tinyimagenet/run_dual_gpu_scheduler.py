"""Two independent GPU workers, bounded sessions and durable job inventory."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from utils.runtime import require_kaggle, jobs, now, read_json, atomic_json, csv_rows, protocol, source_version, config_for


def inspect_jobs(output):
    inventory = []
    manifest_path = output / 'run_manifest.json'
    previous_attempts = read_json(manifest_path).get('attempts', []) if manifest_path.exists() else []
    latest_attempts = {a['run_id']: a for a in previous_attempts}
    ph, version = protocol()[1], source_version()
    for job in jobs():
        folder = output / job['run_id']
        row = {**job, 'status': 'pending'}
        if (folder / 'status.json').exists():
            saved = read_json(folder / 'status.json')
            row.update({k: saved.get(k) for k in ['task', 'epoch', 'phase', 'checkpoint', 'updated']})
            row['status'] = 'resumable'
        if (folder / 'completed.json').exists():
            completed = read_json(folder / 'completed.json')
            if completed['protocol_hash'] != ph or completed['source_version'] != version or completed['config_hash'] != config_for(job['method'])[2]:
                raise RuntimeError('Completed run provenance incompatible: ' + job['run_id'])
            row['status'] = 'complete'
        elif latest_attempts.get(job['run_id'], {}).get('exit_status') not in (None, 0):
            row['status'] = 'failed'
        inventory.append(row)
    return inventory


def main():
    require_kaggle()
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-wall-hours', type=float, default=9.25)
    parser.add_argument('--output', default=str(ROOT / 'outputs'))
    parser.add_argument('--data-root')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--retry-failed', action='store_true', help='Explicit retry with unchanged protocol; previous reports retained')
    parser.add_argument('--stop-grace-minutes', type=float, default=15)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory = inspect_jobs(output)
    if args.status:
        for row in inventory:
            print(row['run_id'], row['status'], row.get('task'), row.get('epoch'))
        print('Completed:', sum(r['status'] == 'complete' for r in inventory), '/ 12')
        return
    if args.max_wall_hours <= 0 or args.stop_grace_minutes <= 0:
        raise ValueError('Positive wall-time and grace required')
    gate = ROOT / 'smoke/latest_pass.json'
    if not gate.exists() or read_json(gate).get('source_version') != source_version() or read_json(gate).get('protocol_hash') != protocol()[1]:
        raise RuntimeError('Run scripts/run_smoke.py first. The scheduler requires a passing resume check for the current source and protocol.')
    import fcntl
    lock = (output / '.scheduler.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('A scheduler already owns this output directory')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError('Select GPU T4 x2; two visible CUDA devices are required')
    gpu_names = [torch.cuda.get_device_name(i) for i in range(2)]
    if not all('T4' in name for name in gpu_names):
        raise RuntimeError('Frozen hardware target is two T4 GPUs: ' + str(gpu_names))
    from data.tinyimagenet import locate_dataset, prepare
    dataset = locate_dataset(args.data_root, args.allow_download)
    prepare(dataset, output)
    session_id = now().replace(':', '-')
    stop_file = output / ('stop_' + session_id)
    previous = read_json(output / 'run_manifest.json') if (output / 'run_manifest.json').exists() else {'attempts': []}
    attempts = previous.get('attempts', [])
    latest_attempts = {a['run_id']: a for a in attempts}
    failures = {rid for rid, a in latest_attempts.items() if a.get('exit_status') not in (None, 0)}
    queue = [r for r in inventory if r['status'] != 'complete' and (args.retry_failed or r['run_id'] not in failures)]
    children = {}
    stopped = {'value': False}
    def signal_stop(*_):
        stopped['value'] = True
    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGINT, signal_stop)
    start = time.monotonic()
    hard_deadline = start + args.max_wall_hours * 3600
    soft_deadline = max(start, hard_deadline - args.stop_grace_minutes * 60)
    def save_manifest():
        rows = inspect_jobs(output)
        for row in rows:
            active = next((v for v in children.values() if v['job']['run_id'] == row['run_id']), None)
            history = [a for a in attempts if a['run_id'] == row['run_id']]
            if history:
                row.update({k: history[-1].get(k) for k in ['gpu', 'start_time', 'end_time', 'exit_status']})
            if active:
                row['status'] = 'running'
            elif row['status'] != 'complete' and history and history[-1].get('exit_status') not in (None, 0):
                row['status'] = 'failed'
        atomic_json(output / 'run_manifest.json', {'session_id': session_id, 'protocol_hash': protocol()[1], 'source_version': source_version(), 'updated': now(), 'runs': rows, 'attempts': attempts})
        csv_rows(output / 'run_manifest.csv', rows)
    try:
        while queue or children:
            current = time.monotonic()
            if stopped['value'] or current >= soft_deadline:
                stop_file.touch(exist_ok=True)
                queue.clear()
            for gpu, child in list(children.items()):
                code = child['process'].poll()
                if code is None and current >= hard_deadline:
                    child['process'].kill()
                    code = child['process'].wait()
                    atomic_json(output / 'failures' / f"{child['job']['run_id']}_walltime_{session_id}.json", {'run_id': child['job']['run_id'], 'error': 'Epoch exceeded shutdown grace; process stopped at hard wall-time. Restore last committed epoch.', 'gpu': gpu, 'timestamp': now(), 'task': read_json(output / child['job']['run_id'] / 'status.json').get('task') if (output / child['job']['run_id'] / 'status.json').exists() else None})
                if code is not None:
                    child['attempt'].update(end_time=now(), exit_status=code)
                    child['log'].close()
                    if code:
                        status = output / child['job']['run_id'] / 'status.json'
                        atomic_json(output / 'failures' / f"{child['job']['run_id']}_process_{session_id}.json", {'run_id': child['job']['run_id'], 'error': f'Worker exited {code}; see worker log/traceback', 'gpu': gpu, 'timestamp': now(), 'last_committed_status': read_json(status) if status.exists() else None, 'log': str(child['log_path'])})
                    del children[gpu]
            for gpu in range(2):
                if gpu in children or not queue or stop_file.exists():
                    continue
                if shutil.disk_usage(output).free < 4 * 1024**3:
                    stopped['value'] = True
                    raise RuntimeError('Less than 4 GiB free. Archive/download and free space before resuming; no hyperparameters changed.')
                job = queue.pop(0)
                log_path = output / 'worker_logs' / f"{job['run_id']}_{session_id}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = log_path.open('w', encoding='utf-8')
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
                command = [sys.executable, str(ROOT / 'run_experiment.py'), '--method', job['method'], '--seed', str(job['seed']), '--device', 'cuda:0', '--resume', 'auto', '--output', str(output), '--data-root', str(dataset), '--stop-file', str(stop_file)]
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                attempt = {'run_id': job['run_id'], 'gpu': gpu, 'gpu_name': gpu_names[gpu], 'start_time': now(), 'end_time': None, 'exit_status': None, 'session_id': session_id, 'pid': process.pid, 'command': command}
                attempts.append(attempt)
                children[gpu] = {'process': process, 'job': job, 'attempt': attempt, 'log': handle, 'log_path': log_path}
                print('GPU', gpu, 'started', job['run_id'], flush=True)
            save_manifest()
            if children:
                time.sleep(10)
    finally:
        stop_file.touch(exist_ok=True)
        for child in children.values():
            child['process'].terminate()
        for child in children.values():
            try:
                code = child['process'].wait(timeout=max(1, hard_deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                child['process'].kill()
                code = child['process'].wait()
            child['attempt'].update(end_time=now(), exit_status=code)
            child['log'].close()
        children.clear()
        save_manifest()
        subprocess.run([sys.executable, str(ROOT / 'create_archives.py'), '--input', str(output)], cwd=ROOT, check=True)
    print('Session stopped. Download BOTH session ZIP files.', flush=True)


if __name__ == '__main__':
    main()
