"""Task splits and dataloaders for continual-learning experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset, Subset


try:
    from torchvision import datasets, transforms
except Exception as exc:  # pragma: no cover - handled clearly at runtime
    datasets = None
    transforms = None
    _TORCHVISION_IMPORT_ERROR = exc
else:
    _TORCHVISION_IMPORT_ERROR = None


@dataclass(frozen=True)
class TaskSplit:
    """A train/test pair and the global classes assigned to one task."""

    train_loader: DataLoader
    test_loader: DataLoader
    classes: list[int]


class LocalLabelDataset(Dataset):
    """Subset a dataset by classes and remap labels to task-local ids."""

    def __init__(
        self,
        base_dataset: Dataset,
        indices: list[int],
        global_classes: list[int],
        target_getter: Callable[[Dataset, int], int] | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = indices
        self.global_classes = list(global_classes)
        self.class_to_local = {class_id: i for i, class_id in enumerate(global_classes)}
        self.target_getter = target_getter or _default_target_getter

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        base_index = self.indices[index]
        image, _ = self.base_dataset[base_index]
        global_label = int(self.target_getter(self.base_dataset, base_index))
        local_label = self.class_to_local[global_label]
        return image, local_label


class TensorTaskDataset(Dataset):
    """Tensor-backed task dataset with local labels."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        if images.shape[0] != labels.shape[0]:
            raise ValueError("images and labels must have the same first dimension.")
        self.images = images
        self.labels = labels.long()

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


def _require_torchvision() -> None:
    if datasets is None or transforms is None:
        raise ImportError("torchvision is required for dataset loading.") from _TORCHVISION_IMPORT_ERROR


def _default_target_getter(dataset: Dataset, index: int) -> int:
    if hasattr(dataset, "targets"):
        return int(dataset.targets[index])
    if hasattr(dataset, "samples"):
        return int(dataset.samples[index][1])
    raise AttributeError("Dataset must expose targets or samples for class filtering.")


def _build_class_order(total_classes: int, num_tasks: int, classes_per_task: int, seed: int) -> list[int]:
    expected = num_tasks * classes_per_task
    if expected > total_classes:
        raise ValueError(
            f"Requested {expected} classes but dataset only has {total_classes}."
        )
    class_order = list(range(total_classes))
    rng = random.Random(seed)
    rng.shuffle(class_order)
    return class_order[:expected]


def _indices_for_classes(dataset: Dataset, class_ids: set[int]) -> list[int]:
    return [
        index
        for index in range(len(dataset))
        if _default_target_getter(dataset, index) in class_ids
    ]


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def _cifar_transforms(dataset_name: str):
    _require_torchvision()
    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif dataset_name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError(f"Unsupported CIFAR dataset: {dataset_name!r}")

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, test_transform


def _build_cifar_tasks(
    dataset_cls,
    total_classes: int,
    num_tasks: int,
    classes_per_task: int,
    seed: int,
    root: str | Path,
    batch_size: int,
    num_workers: int,
    download: bool,
    dataset_name: str,
) -> list[TaskSplit]:
    train_transform, test_transform = _cifar_transforms(dataset_name)
    root = Path(root)
    train_set = dataset_cls(root=root, train=True, download=download, transform=train_transform)
    test_set = dataset_cls(root=root, train=False, download=download, transform=test_transform)

    class_order = _build_class_order(total_classes, num_tasks, classes_per_task, seed)
    tasks: list[TaskSplit] = []

    for task_id in range(num_tasks):
        start = task_id * classes_per_task
        task_classes = class_order[start : start + classes_per_task]
        class_set = set(task_classes)
        train_indices = _indices_for_classes(train_set, class_set)
        test_indices = _indices_for_classes(test_set, class_set)
        train_task = LocalLabelDataset(train_set, train_indices, task_classes)
        test_task = LocalLabelDataset(test_set, test_indices, task_classes)
        tasks.append(
            TaskSplit(
                train_loader=_make_loader(
                    train_task, batch_size, num_workers, shuffle=True, seed=seed + task_id
                ),
                test_loader=_make_loader(
                    test_task, batch_size, num_workers, shuffle=False, seed=seed
                ),
                classes=task_classes,
            )
        )

    return tasks


def get_cifar10_tasks(
    num_tasks: int,
    classes_per_task: int,
    seed: int,
    root: str | Path = "data/raw",
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
) -> list[TaskSplit]:
    """Return CIFAR-10 task splits with labels remapped to task-local ids."""
    _require_torchvision()
    return _build_cifar_tasks(
        datasets.CIFAR10,
        total_classes=10,
        num_tasks=num_tasks,
        classes_per_task=classes_per_task,
        seed=seed,
        root=root,
        batch_size=batch_size,
        num_workers=num_workers,
        download=download,
        dataset_name="cifar10",
    )


def get_cifar100_tasks(
    num_tasks: int,
    classes_per_task: int,
    seed: int,
    root: str | Path = "data/raw",
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
) -> list[TaskSplit]:
    """Return CIFAR-100 task splits with labels remapped to task-local ids."""
    _require_torchvision()
    return _build_cifar_tasks(
        datasets.CIFAR100,
        total_classes=100,
        num_tasks=num_tasks,
        classes_per_task=classes_per_task,
        seed=seed,
        root=root,
        batch_size=batch_size,
        num_workers=num_workers,
        download=download,
        dataset_name="cifar100",
    )


def _tiny_imagenet_transforms():
    _require_torchvision()
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4802, 0.4481, 0.3975),
                std=(0.2302, 0.2265, 0.2262),
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(72),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4802, 0.4481, 0.3975),
                std=(0.2302, 0.2265, 0.2262),
            ),
        ]
    )
    return train_transform, test_transform


def get_tiny_imagenet_tasks(
    num_tasks: int,
    seed: int,
    root: str | Path = "data/raw/tiny-imagenet-200",
    classes_per_task: int | None = None,
    batch_size: int = 128,
    num_workers: int = 4,
) -> list[TaskSplit]:
    """Return Tiny-ImageNet tasks from local ImageFolder-style directories."""
    _require_torchvision()
    root = Path(root)
    train_dir = root / "train"
    val_dir = root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Tiny-ImageNet must be prepared locally with train/ and val/ directories."
        )

    train_transform, test_transform = _tiny_imagenet_transforms()
    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    test_set = datasets.ImageFolder(val_dir, transform=test_transform)
    total_classes = len(train_set.classes)
    if classes_per_task is None:
        if total_classes % num_tasks != 0:
            raise ValueError("classes_per_task is required when classes do not divide evenly.")
        classes_per_task = total_classes // num_tasks

    class_order = _build_class_order(total_classes, num_tasks, classes_per_task, seed)
    tasks: list[TaskSplit] = []

    for task_id in range(num_tasks):
        start = task_id * classes_per_task
        task_classes = class_order[start : start + classes_per_task]
        class_set = set(task_classes)
        train_indices = _indices_for_classes(train_set, class_set)
        test_indices = _indices_for_classes(test_set, class_set)
        train_task = LocalLabelDataset(train_set, train_indices, task_classes)
        test_task = LocalLabelDataset(test_set, test_indices, task_classes)
        tasks.append(
            TaskSplit(
                train_loader=_make_loader(
                    train_task, batch_size, num_workers, shuffle=True, seed=seed + task_id
                ),
                test_loader=_make_loader(
                    test_task, batch_size, num_workers, shuffle=False, seed=seed
                ),
                classes=task_classes,
            )
        )

    return tasks


def get_synthetic_tasks(
    num_tasks: int,
    classes_per_task: int,
    seed: int,
    batch_size: int = 8,
    num_workers: int = 0,
    image_size: int = 32,
    train_samples_per_class: int = 16,
    test_samples_per_class: int = 8,
) -> list[TaskSplit]:
    """Small deterministic image tasks for end-to-end pipeline smoke tests."""
    total_classes = num_tasks * classes_per_task
    prototype_generator = torch.Generator().manual_seed(seed)
    prototypes = torch.randn(total_classes, 3, image_size, image_size, generator=prototype_generator)

    def build_split(samples_per_class: int, split_seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        generator = torch.Generator().manual_seed(split_seed)
        class_tensors: list[tuple[torch.Tensor, torch.Tensor]] = []
        for task_id in range(num_tasks):
            task_images: list[torch.Tensor] = []
            task_labels: list[torch.Tensor] = []
            for local_label in range(classes_per_task):
                global_class = task_id * classes_per_task + local_label
                noise = 0.15 * torch.randn(
                    samples_per_class,
                    3,
                    image_size,
                    image_size,
                    generator=generator,
                )
                images = prototypes[global_class].unsqueeze(0) + noise
                labels = torch.full((samples_per_class,), local_label, dtype=torch.long)
                task_images.append(images)
                task_labels.append(labels)
            class_tensors.append((torch.cat(task_images, dim=0), torch.cat(task_labels, dim=0)))
        return class_tensors

    train_splits = build_split(train_samples_per_class, seed)
    test_splits = build_split(test_samples_per_class, seed + 10_000)
    tasks: list[TaskSplit] = []
    for task_id in range(num_tasks):
        task_classes = list(range(task_id * classes_per_task, (task_id + 1) * classes_per_task))
        train_dataset = TensorTaskDataset(*train_splits[task_id])
        test_dataset = TensorTaskDataset(*test_splits[task_id])
        tasks.append(
            TaskSplit(
                train_loader=_make_loader(
                    train_dataset, batch_size, num_workers, shuffle=True, seed=seed + task_id
                ),
                test_loader=_make_loader(
                    test_dataset, batch_size, num_workers, shuffle=False, seed=seed
                ),
                classes=task_classes,
            )
        )
    return tasks


class ContinualDataset:
    """Convenience wrapper around task splits."""

    def __init__(self, config: dict) -> None:
        dataset_name = config["dataset"].lower()
        common = dict(
            num_tasks=int(config["num_tasks"]),
            classes_per_task=int(config["classes_per_task"]),
            seed=int(config["seed"]),
            root=config.get("data_root", "data/raw"),
            batch_size=int(config["batch_size"]),
            num_workers=int(config["num_workers"]),
        )
        if dataset_name == "cifar10":
            self.tasks = get_cifar10_tasks(**common)
        elif dataset_name == "cifar100":
            self.tasks = get_cifar100_tasks(**common)
        elif dataset_name == "tiny_imagenet":
            self.tasks = get_tiny_imagenet_tasks(**common)
        elif dataset_name == "synthetic":
            self.tasks = get_synthetic_tasks(
                num_tasks=int(config["num_tasks"]),
                classes_per_task=int(config["classes_per_task"]),
                seed=int(config["seed"]),
                batch_size=int(config["batch_size"]),
                num_workers=int(config["num_workers"]),
                image_size=int(config.get("synthetic_image_size", 32)),
                train_samples_per_class=int(config.get("synthetic_train_samples_per_class", 16)),
                test_samples_per_class=int(config.get("synthetic_test_samples_per_class", 8)),
            )
        else:
            raise ValueError(f"Unsupported dataset: {config['dataset']!r}")

        self.class_to_task_map: dict[int, int] = {}
        for task_id, task in enumerate(self.tasks):
            for class_id in task.classes:
                self.class_to_task_map[class_id] = task_id

    def get_task(self, task_id: int, split: str = "train") -> DataLoader:
        task = self.tasks[task_id]
        if split == "train":
            return task.train_loader
        if split == "test":
            return task.test_loader
        raise ValueError("split must be 'train' or 'test'.")

    def get_all_test_loaders(self) -> list[DataLoader]:
        return [task.test_loader for task in self.tasks]

    def get_task_classes(self, task_id: int) -> list[int]:
        return list(self.tasks[task_id].classes)
