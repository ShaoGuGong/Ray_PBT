import logging
import random
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum, auto
from functools import reduce
from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np
import ray
from numpy._typing import NDArray
from torch import device, nn, optim
from torch.utils.data import DataLoader

# ╭──────────────────────────────────────────────────────────╮
# │                          Enums                           │
# ╰──────────────────────────────────────────────────────────╯


class TunerType(Enum):
    NES = auto()
    PBT = auto()
    GROUP_NES = auto()


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
# │                        DataClass                         │
# ╰──────────────────────────────────────────────────────────╯


@dataclass(slots=True)
class WorkerState:
    id: int
    num_cpus: int
    num_gpus: int
    node_name: str
    calculate_ability: float = 0.0
    max_trials: int = 1
    worker_type: WorkerType = WorkerType.CPU


@dataclass(slots=True)
class Hyperparameter:
    lr: float
    momentum: float
    weight_decay: float
    dampening: float
    batch_size: int
    model_type: ModelType

    def __str__(self) -> str:
        return (
            f"Hyperparameter(lr:{self.lr:.6f}, momentum:{self.momentum:.6f}, "
            f"weight_decay:{self.weight_decay:.6f},"
            f" dampening:{self.dampening:.6f}, "
            f"batch_size:{self.batch_size:4d}, model_type:{self.model_type})"
        )

    def to_ndarray(self) -> NDArray[np.floating]:
        return np.array(
            [
                self.lr,
                self.momentum,
                self.weight_decay,
                self.dampening,
                (self.batch_size - 32) / 480.0,
            ],
        )

    @classmethod
    def get_random_hyper(cls) -> "Hyperparameter":
        return cls(
            lr=random.uniform(1e-4, 1e-1),
            momentum=random.uniform(1e-4, 1e-1),
            weight_decay=random.uniform(1e-7, 1e-4),
            dampening=random.uniform(1e-7, 1e-4),
            batch_size=int(random.uniform(0.0, 1.0) * 480.0 + 32),
            model_type=ModelType.RESNET_18,
        )


@dataclass(slots=True)
class Checkpoint:
    model_state_dict: dict
    optimizer_state_dict: dict


# ╭──────────────────────────────────────────────────────────╮
# │                       Type Define                        │
# ╰──────────────────────────────────────────────────────────╯

Accuracy = float


class TrainStepFunction(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> None: ...  # noqa: ANN401


class DataloaderFactory(Protocol):
    def __call__(
        self,
        batch_size: int,
    ) -> tuple[DataLoader, DataLoader]: ...


class ModelInitFunction(Protocol):
    def __call__(
        self,
        hyperparameter: Hyperparameter,
        device: device | None,
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
        colors[i % len(colors)] + "━" * length
        for i, length in enumerate(lengths)
    )
    bar += reset

    data_str = "/".join([f"{x:04d}" for x in data])
    perc_str = "/".join([f"{p * 100:.2f}%" for p in percentages])

    return f"{bar}  {data_str}  {perc_str}"


def unzip_file(zip_path: str, extract_to: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def get_tuner_logger() -> logging.Logger:
    timestamp = (datetime.now(UTC) + timedelta(hours=8)).strftime(
        "%Y-%m-%d_%H-%M-%S",
    )
    log_dir = Path(Path.cwd()) / "logs" / timestamp
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("Tuner")

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # 或者選擇更合適的級別

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s TUNER -- %(message)s",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)  # 只顯示 INFO 級別以上的訊息
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(Path(log_dir) / "tuner.log")
        file_handler.setLevel(logging.DEBUG)  # 記錄所有級別的日誌
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ╭──────────────────────────────────────────────────────────╮
# │                          String                          │
# ╰──────────────────────────────────────────────────────────╯

LOG_TABLE_HEAD: str = (
    f"┏{'':━^4}┳{'':━^11}┳{'':━^11}┳"
    f"{'':━^37}┳{'':━^3}┳{'':━^7}┳{'':━^7}┓\n"
    f"┃{'':^4}┃{'':^11}┃{'Worker':^11}┃"
    f"{'Hyparameter':^37}┃{'':^3}┃{'':^7}┃{'':^7}┃\n"
    f"┃{'ID':^4}┃{'Status':^11}┣{'':━^4}┳{'':━^6}╋"
    f"{'':━^7}┳{'':━^10}┳{'':━^6}┳{'':━^11}┫{'Ph':^3}┃{'Iter':^7}┃{'Acc':^7}┃\n"
    f"┃{'':^4}┃{'':^11}┃{'ID':^4}┃{'TYPE':^6}┃{'lr':^7}┃"
    f"{'momentum':^10}┃{'bs':^6}┃{'model':^11}┃{'':^3}┃{'':^7}┃{'':^7}┃\n"
    f"┣{'':━^4}╋{'':━^11}╋{'':━^4}╋{'':━^6}╋{'':━^7}╋"
    f"{'':━^10}╋{'':━^6}╋{'':━^11}╋{'':━^3}╋{'':━^7}╋{'':━^7}┫\n"
)

LOG_TABLE_TAIL: str = (
    f"┗{'':━^4}┻{'':━^11}┻{'':━^4}┻{'':━^6}┻{'':━^7}┻"
    f"{'':━^10}┻{'':━^6}┻{'':━^11}┻{'':━^3}┻{'':━^7}┻{'':━^7}┛\n"
)
