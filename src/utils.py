import logging
import math
import random
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from functools import reduce
from pathlib import Path
from typing import Any, NamedTuple, Protocol, TypeVar

import numpy as np
import ray
from numpy._typing import NDArray
from numpy.random import Generator
from scipy.linalg import expm
from torch import nn, optim
from torch.utils.data import DataLoader

# ╭──────────────────────────────────────────────────────────╮
# │                          Enums                           │
# ╰──────────────────────────────────────────────────────────╯


class TunerType(Enum):
    PBT = auto()
    NES = auto()


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


#                         ╭─────────────────────────────╮
#                         │ Dataclasses and NamedTuples │
#                         ╰─────────────────────────────╯


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
    weight_decay: float
    dampening: float
    batch_size: int
    model_type: ModelType

    def __str__(self) -> str:
        return (
            f"Hyperparameter(lr:{self.lr:.3f}, momentum:{self.momentum:.3f} "
            f"batch_size:{self.batch_size:4d}, model_type:{self.model_type})"
        )

    def to_ndarray(self) -> NDArray[np.floating]:
        return np.array([self.lr, self.momentum, self.weight_decay, self.dampening])

    @classmethod
    def random(cls) -> "Hyperparameter":
        return cls(
            lr=random.uniform(0.001, 1),
            momentum=random.uniform(0.001, 1),
            weight_decay=random.uniform(1e-5, 1e-3),
            dampening=random.uniform(1e-5, 1e-3),
            batch_size=512,
            model_type=ModelType.RESNET_18,
        )


@dataclass
class Fitness:
    accuracy: float
    hyperparameter: Hyperparameter


class NaturalGradients(NamedTuple):
    sigma_gradient: np.floating
    delta_gradient: NDArray[np.floating]
    b_gradient: NDArray[np.floating]


@dataclass
class Distribution:
    mean: NDArray[np.floating]
    sigma: float
    b_matrix: NDArray[np.floating]
    random_generator: Generator

    @staticmethod
    def fitness_shaping(fitnesses: list[Fitness]) -> list:
        """
        Fitness Shaping use

        Args:
            fitness: model's fitness.

        Returns:
            List[float]: Returns utility.

        Raises:
            ZeroDivisionError: if length of fitnesses is zero.
        """
        #                            ╭───────────────────────╮
        #                            │ NES'S Fitness Shaping │
        #                            ╰───────────────────────╯
        size = len(fitnesses)
        if size == 0:
            raise ZeroDivisionError
        base = math.log(size / 2 + 1)
        utilities_raw = [max(0.0, base - math.log(i + 1)) for i in range(size)]
        denominator = sum(utilities_raw)
        return [u / denominator - 1.0 / size for u in utilities_raw]

    @staticmethod
    def mapping(x: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Map real number x ∈ ℝ to [0, 1] using tanh

        Args:
            x (float): Real input value

        Returns:
            float: Output in [0, 1]

        Example:
            >>> round(Hyperparameter.mapping(0), 3)
            0.5
        """
        y = np.tanh(x) / 2.0 + 0.5
        assert np.all((y >= 0.0) & (y <= 1.0)), f"Mapping Error output is {y}"
        return y

    @staticmethod
    def inverse_mapping(y: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Map [0, 1] to ℝ using arctanh

        Args:
            y (float): Input in [0, 1]

        Returns:
            float: Real number

        Example:
            >>> x = 0.5
            >>> Hyperparameter.inverse_mapping(x)
            0.0
        """
        assert np.all((y >= 0.0) & (y <= 1.0)), "Input y is contrains invalid number"
        return np.arctanh(2.0 * y - 1.0)

    @classmethod
    def get_random_ditribution(cls) -> "Distribution":
        random_generator = np.random.default_rng()
        init_hyper = np.concatenate(
            [
                random_generator.uniform(
                    1e-4,
                    1e-1,
                    size=1,
                ),
                random_generator.uniform(
                    3e-1,
                    9e-1,
                    size=1,
                ),
                random_generator.uniform(
                    1e-6,
                    2e-5,
                    size=2,
                ),
            ],
        )

        diag_vals = random_generator.uniform(0.8, 1.5, size=len(init_hyper))
        covariance = np.diag(diag_vals)
        square_root_of_covariance = np.linalg.cholesky(covariance).T
        mean = np.array(init_hyper)
        sigma = abs(np.linalg.det(square_root_of_covariance)) ** (1 / len(init_hyper))
        b_matrix = square_root_of_covariance / sigma

        return cls(
            mean=mean,
            sigma=sigma,
            b_matrix=b_matrix,
            random_generator=random_generator,
        )

    def get_new_hyper(self) -> Hyperparameter:
        sample: NDArray[np.floating] = self.random_generator.multivariate_normal(
            mean=np.zeros(self.mean.size),
            cov=np.eye(self.mean.size),
        )
        moddify_sample: np.ndarray = self.mean + self.sigma * (self.b_matrix @ sample)
        parameter: list[float] = Distribution.mapping(moddify_sample).tolist()
        return Hyperparameter(
            *parameter,
            batch_size=512,
            model_type=ModelType.RESNET_18,
        )

    def update_distribution(self, fitnesses: list[Fitness]) -> None:
        gradients: NaturalGradients = self._commpute_gradient(fitnesses=fitnesses)
        dimension = 4
        mean_stride = 1.0
        b_matrix_stride = (9 + 3 * np.log(dimension)) / (
            5 * dimension * np.sqrt(dimension)
        )
        sigma_stride = (9 + 3 * np.log(dimension)) / (
            5 * dimension * np.sqrt(dimension)
        )
        self.mean += (
            mean_stride * self.sigma * (self.b_matrix @ gradients.delta_gradient)
        )
        self.sigma *= np.exp((sigma_stride / 2) * gradients.sigma_gradient)
        self.b_matrix = self.b_matrix @ expm(
            (b_matrix_stride / 2) * gradients.b_gradient,
        )

    def _commpute_gradient(
        self,
        fitnesses: list[Fitness],
    ) -> NaturalGradients:
        """
        Compute the natural gradient from the fitnesses.

        Args:
            fitnesses (list[Fitness]): List of Fitness objects.

        Returns:
            NaturalGradients: The computed natural gradients.
        """
        sorted_fitnesses = sorted(fitnesses, key=lambda x: x.accuracy, reverse=True)
        utilities = Distribution.fitness_shaping(sorted_fitnesses)
        shaping_fitnesses = [
            Fitness(accuracy=utility, hyperparameter=fitness.hyperparameter)
            for utility, fitness in zip(utilities, sorted_fitnesses, strict=True)
        ]
        delta_gradient = np.sum(
            [
                fitness.accuracy
                * Distribution.inverse_mapping(fitness.hyperparameter.to_ndarray())
                for fitness in shaping_fitnesses
            ],
            axis=0,
        )
        m_gradient = np.sum(
            [
                fitness.accuracy
                * (
                    np.outer(
                        Distribution.inverse_mapping(
                            fitness.hyperparameter.to_ndarray(),
                        ),
                        Distribution.inverse_mapping(
                            fitness.hyperparameter.to_ndarray(),
                        ),
                    )
                    - np.eye(self.mean.size)
                )
                for fitness in shaping_fitnesses
            ],
            axis=0,
        )
        sigma_gradient = np.trace(m_gradient) / self.mean.size
        b_gradient = m_gradient - sigma_gradient * np.eye(self.mean.size)

        return NaturalGradients(
            sigma_gradient=sigma_gradient,
            delta_gradient=delta_gradient,
            b_gradient=b_gradient,
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
    ) -> tuple[DataLoader, DataLoader]: ...


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


def get_tuner_logger() -> logging.Logger:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
