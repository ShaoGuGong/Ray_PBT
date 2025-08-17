#! /usr/bin/env python3
import argparse
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from itertools import islice, repeat
from pathlib import Path
from time import perf_counter

import ray
import torch
import torchvision
from torch import device, nn, optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

from src.config import DATASET_PATH, STOP_ITERATION
from src.group_nes_tuner import GroupNESTuner
from src.nes_tuner import NESTuner
from src.pbt_tuner import PBTTuner
from src.trial_state import TrialState
from src.utils import (
    Hyperparameter,
    TunerType,
    get_head_node_address,
    unzip_file,
)
from src.utils_nes import Distribution, DistributionManager


def cifar10_data_loader_factory(
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    data_dir = Path(DATASET_PATH).expanduser()
    if not Path(data_dir).exists():
        Path(data_dir).mkdir(parents=True, exist_ok=True)

    if not (Path(data_dir) / "cifar-10-batches-py").exists():
        print(f"{Path(data_dir) / 'cifar-10-batches-py'} 不存在")  # noqa: T201
        torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=None,
        )

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ],
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ],
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=train_transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader


def resnet18_init_fn(
    hyperparameter: Hyperparameter,
    device: device | None = None,
) -> tuple[nn.Module, optim.Optimizer]:
    model = models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()  # type: ignore[assignment]
    if device is not None:
        model.to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=hyperparameter.lr,
        momentum=hyperparameter.momentum,
        weight_decay=hyperparameter.weight_decay,
        dampening=hyperparameter.dampening,
    )
    return model, optimizer


def generate_trial_states(
    n: int,
    random_fn: Callable | None = None,
    tuner_type: TunerType = TunerType.PBT,
) -> list[TrialState]:
    match tuner_type:
        case TunerType.PBT:
            if random_fn is not None:
                error_message = "PBT tuner does not support random_fn"
                raise ValueError(error_message)
            return [
                TrialState(
                    i,
                    Hyperparameter.get_random_hyper(),
                    model_init_fn=resnet18_init_fn,
                    stop_iteration=STOP_ITERATION,
                )
                for i in range(n)
            ]
        case TunerType.NES:
            if random_fn is None:
                error_message = "NES tuner requires a random_fn"
                raise ValueError(error_message)
            return [
                TrialState(
                    i,
                    random_fn(),
                    model_init_fn=resnet18_init_fn,
                    stop_iteration=STOP_ITERATION,
                )
                for i in range(n)
            ]
        case TunerType.GROUP_NES:
            if random_fn is None:
                error_message = "Group NES tuner requires a random_fn"
                raise ValueError(error_message)
            return [
                TrialState(
                    i,
                    random_fn(i),
                    model_init_fn=resnet18_init_fn,
                    stop_iteration=STOP_ITERATION,
                )
                for i in range(n)
            ]


def train_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    device: device | None = None,
) -> None:
    model.train()
    train_device = device if device is not None else torch.device("cpu")
    criterion = nn.CrossEntropyLoss().to(train_device)

    for raw_inputs, raw_targets in islice(train_loader, 1):
        inputs, targets = (
            raw_inputs.to(train_device),
            raw_targets.to(train_device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


def collect_all_nodes_logs(tuner: ray.ObjectRef) -> None:
    zip_logs_bytes: bytes = ray.get(
        tuner.get_zipped_log.remote(),  # type: ignore[call-arg]
    )
    zip_output_dir = f"./logs/{
        (datetime.now(UTC) + timedelta(hours=8)).strftime('%Y-%m-%d_%H-%M-%S')
    }/"
    Path(zip_output_dir).mkdir(parents=True, exist_ok=True)
    zip_output_path = Path(zip_output_dir) / "logs.zip"
    with Path(zip_output_path).open("wb") as f:
        f.write(zip_logs_bytes)

    unzip_file(zip_output_path, zip_output_dir)  # type: ignore[call-arg]


def hyperparameter_optimize(tuner_type: TunerType, trial_num: int) -> None:
    start_time = perf_counter()
    ray.init(
        runtime_env={
            "working_dir": ".",
            "excludes": [
                ".git",
                "test",
                "logs/*",
                "LICENSE",
                "README.md",
                ".venv",
            ],
        },
    )
    match tuner_type:
        case TunerType.NES:
            distribution: Distribution = Distribution.get_random_distribution()
            trial_states = generate_trial_states(
                trial_num,
                distribution.get_new_hyper,
            )
            tuner = NESTuner.options(  # type: ignore[call-arg]
                max_concurrency=5,
                num_cpus=1,
                resources={f"node:{get_head_node_address()}": 0.01},
            ).remote(
                trial_states,
                train_step,
                cifar10_data_loader_factory,
                distribution,
            )
        case TunerType.PBT:
            trial_states = generate_trial_states(trial_num)
            tuner = PBTTuner.options(  # type: ignore[call-arg]
                max_concurrency=5,
                num_cpus=1,
                resources={f"node:{get_head_node_address()}": 0.01},
            ).remote(trial_states, train_step, cifar10_data_loader_factory)

        case TunerType.GROUP_NES:
            manager: DistributionManager = DistributionManager(
                num_distributions=3,
            )
            trial_states = generate_trial_states(
                trial_num,
                manager.hyper_init,
            )
            tuner = GroupNESTuner.options(  # type: ignore[call-arg]
                max_concurrency=5,
                num_cpus=1,
                resources={f"node:{get_head_node_address()}": 0.01},
            ).remote(
                trial_states,
                train_step,
                cifar10_data_loader_factory,
                manager,
            )

    ray.get(tuner.run.remote())  # type: ignore[call-arg]
    end_time = perf_counter()

    collect_all_nodes_logs(tuner)

    ray.shutdown()

    log_dir = Path("~/Documents/workspace/shaogu/Ray_PBT/log").expanduser()
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "time.log"
    with log_file.open("a") as f:
        f.write(f"{tuner_type} use time: {end_time - start_time}\n")


def resolve_tuners(tuner_type: str) -> list[TunerType]:
    res: list[TunerType] | None = None
    match tuner_type:
        case "COM":
            res = [TunerType.NES, TunerType.PBT]
        case "NES":
            res = [TunerType.NES]
        case "PBT":
            res = [TunerType.PBT]
        case _:
            error_message = f"Unknown tuner type: {tuner_type}"
            raise ValueError(error_message)
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tuner_type",
        "-T",
        type=str,
        choices=["NES", "PBT", "COM"],
        default="COM(comparison PBT and NES)",
        dest="tuner_type",
        help="Select the Tuner type",
    )
    parser.add_argument(
        "--test_num",
        "-N",
        type=int,
        default=1,
        dest="test_num",
        help="How many times to run the test",
    )
    parser.add_argument(
        "--trial_num",
        "-R",
        type=int,
        default=5,
        dest="trial_num",
        help="Number of trails to run for each tuner type",
    )
    args = parser.parse_args()

    tuners = resolve_tuners(args.tuner_type)
    for _ in repeat(None, args.test_num):
        for tuner_type in tuners:
            hyperparameter_optimize(
                tuner_type=tuner_type,
                trial_num=args.trial_num,
            )


if __name__ == "__main__":
    main()
