import os
from dataclasses import dataclass
from enum import Enum, auto
from functools import reduce
from typing import Any, Callable, Protocol, TypeVar

import ray
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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


class TrialStatus(Enum):
    RUNNING = auto()
    PENDING = auto()
    TERMINAL = auto()
    PAUSE = auto()


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


@dataclass
class Hyperparameter:
    lr: float
    momentum: float
    batch_size: int
    model_type: ModelType


@dataclass
class Checkpoint:
    model_state_dict: dict
    optimzer_state_dict: dict
    checkpoint_interval: int


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
    model_type: ModelType,
    batch_size: int = 64,
    train_transform=None,
    test_transform=None,
    data_dir=DATASET_PATH,
) -> tuple:
    data_dir = os.path.expanduser(data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

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

    if not os.path.exists(os.path.join(data_dir, "cifar-10-batches-py")):
        print(f"{os.path.join(data_dir, 'cifar-10-batches-py')} 不存在")
        torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=None
        )

    if model_type == ModelType.RESNET_18:
        train_loader = DataLoader(
            torchvision.datasets.CIFAR10(
                root=data_dir, train=True, download=False, transform=train_transform
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            torchvision.datasets.CIFAR10(
                root=data_dir, train=False, download=False, transform=test_transform
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        return train_loader, test_loader

    elif model_type == ModelType.RESNET_50:
        train_loader = DataLoader(
            torchvision.datasets.CIFAR100(
                root=data_dir, train=True, download=False, transform=train_transform
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        test_loader = DataLoader(
            torchvision.datasets.CIFAR100(
                root=data_dir, train=False, download=False, transform=test_transform
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        return train_loader, test_loader

    raise ValueError("`model_type` must be in ModelType")


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
