"""Filesystem transactions, frozen protocol and Kaggle execution guard."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def require_kaggle():
    if not Path('/kaggle/working').is_dir() or sys.platform != 'linux':
        raise RuntimeError('Execution is restricted to Kaggle. Do not run this project on the laptop.')


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def csv_rows(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def protocol():
    import yaml
    value = read_json(ROOT / 'protocol_manifest.json')
    value['resolved_configs'] = {p.stem: yaml.safe_load(p.read_text()) for p in sorted((ROOT / 'configs').glob('*.yaml'))}
    return value, digest(value)


def config_for(method, smoke=False):
    value, ph = protocol()
    cfg = dict(value['resolved_configs']['base'])
    cfg.update(value['resolved_configs'][method])
    smoke_cfg = cfg.pop('smoke')
    if smoke:
        cfg.update(smoke_cfg)
    cfg['smoke'] = smoke
    return cfg, ph, digest(cfg)


def source_version():
    paths = list((ROOT / 'src').rglob('*.py')) + list(ROOT.glob('*.py')) + list((ROOT / 'scripts').glob('*.py'))
    return digest({p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(paths)})


def jobs():
    p, _ = protocol()
    return [{'run_id': f'tinyimagenet_{m}_seed{s}', 'method': m, 'seed': s} for m in p['methods'] for s in p['seeds']]


def bind_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f'Frozen manifest mismatch: {path}. Resume cannot change the protocol or split.')
    else:
        atomic_json(path, value)


def safe_extract(archive, target):
    import zipfile
    target = Path(target).resolve()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            dest = (target / info.filename).resolve()
            if not dest.is_relative_to(target) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe archive member: ' + info.filename)
        bad = z.testzip()
        if bad:
            raise ValueError('Corrupt archive member: ' + bad)
        z.extractall(target)
