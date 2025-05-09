import os
import random
import zipfile
from dataclasses import dataclass
from enum import Enum, auto
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar

import numpy as np
import ray
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from .config import DATASET_PATH

# ╭──────────────────────────────────────────────────────────╮
# │                       Type Define                        │
# ╰──────────────────────────────────────────────────────────╯

Accuracy = float


class TrainStepFunction(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> None: ...


# ╭──────────────────────────────────────────────────────────╮
# │                          Enums                           │
# ╰──────────────────────────────────────────────────────────╯


class ModelType(Enum):
    RESNET_18 = auto()
    RESNET_50 = auto()

    def __str__(self):
        return self.name


class TrialStatus(Enum):
    RUNNING = auto()
    PENDING = auto()
    TERMINATE = auto()
    PAUSE = auto()
    NEED_MUTATION = auto()
    FAILED = auto()

    def __str__(self):
        return self.name


class WorkerType(Enum):
    CPU = auto()
    GPU = auto()


class DatasetType(Enum):
    CIFAR10 = auto()
    CIFAR100 = auto()
    IMAGENET = auto()


# ╭──────────────────────────────────────────────────────────╮
# │                       Dataclasses                        │
# ╰──────────────────────────────────────────────────────────╯


@dataclass
class WorkerState:
    id: int
    num_cpus: int
    num_gpus: int
    node_name: str
    calculate_ability: float = 0.0
    max_trials: int = 1
    worker_type: WorkerType = WorkerType.CPU


@dataclass
class Hyperparameter:
    lr: float
    momentum: float
    batch_size: int
    model_type: ModelType

    def __str__(self) -> str:
        return f"Hyperparameter(lr:{self.lr:.3f}, momentum:{self.momentum:.3f}, batch_size:{self.batch_size:4d}, model_type:{self.model_type})"

    @classmethod
    def random(cls) -> "Hyperparameter":
        return cls(
            lr=random.uniform(0.001, 1),
            momentum=random.uniform(0.001, 1),
            batch_size=random.choice([64, 128, 256, 512, 1024]),
            model_type=ModelType.RESNET_18,
        )


@dataclass
class Checkpoint:
    model_state_dict: Dict
    optimizer_state_dict: Dict


# ╭──────────────────────────────────────────────────────────╮
# │                        Functions                         │
# ╰──────────────────────────────────────────────────────────╯

T = TypeVar("T")
Composeable = Callable[[T], T]


def compose(*functions: Composeable) -> Composeable:
    def apply(value: T, fn: Composeable[T]) -> T:
        return fn(value)

    return lambda data: reduce(apply, functions[::-1], data)


def pipe(*functions: Composeable) -> Composeable:
    def apply(value: T, fn: Composeable[T]) -> T:
        return fn(value)

    return lambda data: reduce(apply, functions, data)


def get_data_loader(
    batch_size: int = 64,
    train_transform: Optional[transforms.Compose] = None,
    test_transform: Optional[transforms.Compose] = None,
    data_dir: str = DATASET_PATH,
) -> tuple:
    data_dir = os.path.expanduser(data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(os.path.join(data_dir, "cifar-10-batches-py")):
        print(f"{os.path.join(data_dir, 'cifar-10-batches-py')} 不存在")
        torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=None
        )

    if train_transform is None:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    if test_transform is None:
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=test_transform
    )

    np.random.seed(42)
    indices = np.random.permutation(len(test_dataset))
    test_size = len(test_dataset) // 2
    test_indices = indices[:test_size]
    valid_indices = indices[test_size:]

    valid_dataset = Subset(test_dataset, valid_indices.tolist())
    test_dataset = Subset(test_dataset, test_indices.tolist())

    # print(f"{len(train_dataset)=}, {len(valid_dataset)=}, {len(test_dataset)=}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader


def get_model(model_type: ModelType):
    if model_type == ModelType.RESNET_18:
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model

    elif model_type == ModelType.RESNET_50:
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, 100)
        return model


def get_head_node_address() -> str:
    return ray.get_runtime_context().gcs_address.split(":")[0]


def colored_progress_bar(data: List[int], bar_width: int) -> str:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    colors = [GREEN, RED, YELLOW]
    total = sum(data)
    if total == 0:
        return " " * bar_width + " (no data)"

    percentages = [x / total for x in data]
    lengths = [int(p * bar_width) for p in percentages]

    while sum(lengths) < bar_width:
        max_idx = percentages.index(max(percentages))
        lengths[max_idx] += 1

    bar = "".join(colors[i % len(colors)] + "━" * l for i, l in enumerate(lengths))
    bar += RESET

    data_str = "/".join([f"{x:04d}" for x in data])
    perc_str = "/".join([f"{p * 100:.2f}%" for p in percentages])

    return f"{bar}  {data_str}  {perc_str}"


def unzip_file(zip_path: str, extract_to: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
