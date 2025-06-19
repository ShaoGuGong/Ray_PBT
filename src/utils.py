import random
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from functools import reduce
from typing import Any, Protocol, TypeVar

import ray
from torch import nn, optim
from torch.utils.data import DataLoader

# ╭──────────────────────────────────────────────────────────╮
# │                          Enums                           │
# ╰──────────────────────────────────────────────────────────╯


class ModelType(Enum):
    RESNET_18 = auto()
    RESNET_50 = auto()

    def __str__(self) -> str:
        return self.name


class TrialStatus(Enum):
    RUNNING = auto()
    PENDING = auto()
    TERMINATE = auto()
    PAUSE = auto()
    INTERRUPTED = auto()
    NEED_MUTATION = auto()
    FAILED = auto()

    def __str__(self) -> str:
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
        return (
            f"Hyperparameter(lr:{self.lr:.3f}, momentum:{self.momentum:.3f} "
            f"batch_size:{self.batch_size:4d}, model_type:{self.model_type})"
        )

    @classmethod
    def random(cls) -> "Hyperparameter":
        return cls(
            lr=random.uniform(0.001, 1),
            momentum=random.uniform(0.001, 1),
            batch_size=512,
            model_type=ModelType.RESNET_18,
        )


@dataclass
class Checkpoint:
    model_state_dict: dict
    optimizer_state_dict: dict


# ╭──────────────────────────────────────────────────────────╮
# │                       Type Define                        │
# ╰──────────────────────────────────────────────────────────╯

Accuracy = float


class TrainStepFunction(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> None: ...


class DataloaderFactory(Protocol):
    def __call__(
        self,
        batch_size: int,
    ) -> tuple[DataLoader, DataLoader, DataLoader]: ...


class ModelInitFunction(Protocol):
    def __call__(
        self,
        hyperparameter: Hyperparameter,
    ) -> tuple[nn.Module, optim.Optimizer]: ...


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


def get_head_node_address() -> str:
    return ray.get_runtime_context().gcs_address.split(":")[0]


def colored_progress_bar(data: list[int], bar_width: int) -> str:
    green = "\033[92m"
    red = "\033[91m"
    yellow = "\033[93m"
    reset = "\033[0m"

    colors = [green, red, yellow]
    total = sum(data)
    if total == 0:
        return " " * bar_width + " (no data)"

    percentages = [x / total for x in data]
    lengths = [int(p * bar_width) for p in percentages]

    while sum(lengths) < bar_width:
        max_idx = percentages.index(max(percentages))
        lengths[max_idx] += 1

    bar = "".join(
        colors[i % len(colors)] + "━" * length for i, length in enumerate(lengths)
    )
    bar += reset

    data_str = "/".join([f"{x:04d}" for x in data])
    perc_str = "/".join([f"{p * 100:.2f}%" for p in percentages])

    return f"{bar}  {data_str}  {perc_str}"


def unzip_file(zip_path: str, extract_to: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
