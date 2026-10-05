"""Read original or class-folder validation layout without rewriting the data."""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import re
import urllib.request

from utils.runtime import ROOT, require_kaggle, atomic_json, bind_json, digest, file_hash, protocol, safe_extract

EXTENSIONS = {'.jpeg', '.jpg', '.png'}


def locate_dataset(explicit=None, allow_download=False):
    require_kaggle()
    roots = [Path(explicit)] if explicit else [Path('/kaggle/input'), Path('/kaggle/working/runtime_data')]
    candidates = set()
    for parent in roots:
        if (parent / 'train').is_dir() and (parent / 'val').is_dir():
            candidates.add(parent.resolve())
        if parent.exists():
            for train in parent.rglob('train'):
                if train.is_dir() and (train.parent / 'val').is_dir():
                    classes = [p for p in train.iterdir() if p.is_dir()]
                    if len(classes) == 200:
                        candidates.add(train.parent.resolve())
    if len(candidates) > 1:
        raise RuntimeError('Multiple Tiny ImageNet roots. Set --data-root explicitly: ' + str(sorted(map(str, candidates))))
    if candidates:
        return next(iter(candidates))
    if allow_download:
        dest = Path('/kaggle/working/runtime_data')
        dest.mkdir(parents=True, exist_ok=True)
        archive = dest / 'tiny-imagenet-200.zip'
        url = 'https://cs231n.stanford.edu/tiny-imagenet-200.zip'
        try:
            urllib.request.urlretrieve(url, archive)
            safe_extract(archive, dest)
        except Exception as exc:
            raise RuntimeError('Download failed. Attach the original tiny-imagenet-200 dataset in Kaggle Add Input, including train/, val/ and validation labels.') from exc
        atomic_json(dest / 'download_source.json', {'url': url, 'sha256': file_hash(archive)})
        archive.unlink()
        return locate_dataset(str(dest), False)
    raise FileNotFoundError('Tiny ImageNet not found. Kaggle > Add Input: attach tiny-imagenet-200 with 200 class train folders and val/val_annotations.txt (or labeled val class folders). Internet download is optional with --allow-download.')


def image_files(folder):
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)


def dataset_index(root):
    root = Path(root)
    classes = sorted(p.name for p in (root / 'train').iterdir() if p.is_dir())
    if len(classes) != 200:
        raise ValueError(f'Expected 200 classes, found {len(classes)}')
    train = {c: image_files(root / 'train' / c) for c in classes}
    annotations = root / 'val' / 'val_annotations.txt'
    val = {c: [] for c in classes}
    if annotations.exists():
        seen = set()
        for line in annotations.read_text().splitlines():
            name, label, *_ = line.split()
            if label not in val or name in seen:
                raise ValueError('Invalid/duplicate validation annotation: ' + line)
            seen.add(name)
            path = root / 'val' / 'images' / name
            if not path.exists():
                path = root / 'val' / label / 'images' / name
            if not path.exists():
                path = root / 'val' / label / name
            if not path.is_file():
                raise FileNotFoundError(path)
            val[label].append(path)
    else:
        val = {c: image_files(root / 'val' / c) for c in classes}
    for c in classes:
        val[c] = sorted(val[c])
        if len(train[c]) != 500 or len(val[c]) != 50:
            raise ValueError(f'{c}: expected 500 train / 50 val, got {len(train[c])} / {len(val[c])}. No silent subset.')
    inventory = [(split, c, p.name, p.stat().st_size) for split, source in [('train', train), ('val', val)] for c in classes for p in source[c]]
    return {'classes': classes, 'train': train, 'val': val, 'fingerprint': digest(inventory)}


def make_split(index, seed, num_tasks=10, classes_per_task=20):
    import numpy as np
    classes = index['classes']
    order = np.random.default_rng(seed).permutation(len(classes)).tolist()[:num_tasks * classes_per_task]
    return {
        'seed': seed, 'dataset': 'tinyimagenet', 'dataset_fingerprint': index['fingerprint'],
        'algorithm': 'sorted original IDs + numpy.default_rng(seed).permutation',
        'original_class_ids': classes,
        'wordnet_ids': [c if re.fullmatch(r'n\d{8}', c) else None for c in classes],
        'class_order': order, 'class_order_ids': [classes[i] for i in order],
        'tasks': [{'task': t, 'original_indices': order[t * classes_per_task:(t + 1) * classes_per_task],
                   'class_ids': [classes[i] for i in order[t * classes_per_task:(t + 1) * classes_per_task]]}
                  for t in range(num_tasks)],
    }


def prepare(root, output):
    index = dataset_index(root)
    value, ph = protocol()
    output = Path(output)
    bind_json(output / 'protocol_frozen.json', {'protocol_hash': ph, 'protocol': value})
    bind_json(output / 'dataset_identity.json', {'fingerprint': index['fingerprint'], 'classes': index['classes'], 'train_per_class': 500, 'val_per_class': 50})
    atomic_json(output / 'dataset_source.json', {'mounted_root': str(root), 'fingerprint': index['fingerprint'], 'upstream': 'https://cs231n.stanford.edu/2017/project.html', 'identity_basis': 'split, class, filename and byte size; not pixel hash'})
    for seed in value['seeds']:
        bind_json(output / 'splits' / f'seed_{seed}.json', make_split(index, seed))
    return index


def seed_worker(_):
    import numpy as np
    import torch
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class Images:
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image
        path, label = self.rows[i]
        with Image.open(path) as image:
            if image.size != (64, 64):
                raise ValueError(f'Expected 64x64 image: {path}: {image.size}')
            return self.transform(image.convert('RGB')), label


def loader(index, split, task, cfg, seed, epoch=0, training=False, buffer=False):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    aug = cfg['augmentation']
    transforms = []
    if training:
        transforms.extend([T.RandomCrop(aug['random_crop_size'], padding=aug['padding']), T.RandomHorizontalFlip(aug['horizontal_flip_probability'])])
    transforms.extend([T.ToTensor(), T.Normalize(aug['mean'], aug['std'])])
    source = index['train' if training or buffer else 'val']
    rows = []
    for label, c in enumerate(split['tasks'][task]['class_ids']):
        paths = source[c]
        if cfg['smoke']:
            paths = paths[:cfg['train_per_class'] if training or buffer else cfg['val_per_class']]
        rows.extend((p, label) for p in paths)
    # Independent deterministic epoch loaders make an epoch-boundary resume replayable.
    gen = torch.Generator().manual_seed(seed * 100000 + task * 1000 + epoch * 3 + (1 if training else 2))
    return DataLoader(Images(rows, T.Compose(transforms)), batch_size=cfg['batch_size'] if training or buffer else cfg['eval_batch_size'],
                      shuffle=training, num_workers=cfg['num_workers'], worker_init_fn=seed_worker,
                      generator=gen, pin_memory=True, persistent_workers=False, drop_last=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--output', default=str(ROOT / 'outputs'))
    args = parser.parse_args()
    require_kaggle()
    root = locate_dataset(args.data_root, args.allow_download)
    prepare(root, args.output)
    print('Tiny ImageNet ready:', root, flush=True)


if __name__ == '__main__':
    main()
