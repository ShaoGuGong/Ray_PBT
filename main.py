#! /usr/bin/env python3
import argparse
from collections.abc import Callable
from datetime import datetime
from itertools import islice
from pathlib import Path
from time import perf_counter

import ray
import torch
import torchvision
from torch import nn, optim
from torch._prims_common import DeviceLikeType
from torch.utils.data import DataLoader
from torchvision import models, transforms

from src.config import DATASET_PATH, STOP_ITERATION
from src.nes_tuner import NESTuner
from src.pbt_tuner import PBTTuner
from src.trial_state import TrialState
from src.utils import (
    Distribution,
    Hyperparameter,
    TunerType,
    get_head_node_address,
    unzip_file,
)


def cifar10_data_loader_factory(
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    data_dir = Path(DATASET_PATH).expanduser()
    if not Path(data_dir).exists():
        Path(data_dir).mkdir(parents=True, exist_ok=True)

    if not (Path(data_dir) / "cifar-10-batches-py").exists():
        print(f"{Path(data_dir) / 'cifar-10-batches-py'} 不存在")
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
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ],
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
) -> tuple[nn.Module, optim.Optimizer]:
    model = models.resnet18(num_classes=10)
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
) -> list[TrialState]:
    if random_fn is None:
        return [
            TrialState(
                i,
                Hyperparameter.random(),
                model_init_fn=resnet18_init_fn,
                stop_iteration=STOP_ITERATION,
            )
            for i in range(n)
        ]

    return [
        TrialState(
            i,
            random_fn(),
            model_init_fn=resnet18_init_fn,
            stop_iteration=STOP_ITERATION,
        )
        for i in range(n)
    ]


def train_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    batch_size: int,
    device: DeviceLikeType = torch.device("cpu"),
) -> None:
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)

    for raw_inputs, raw_targets in islice(train_loader, 1):
        inputs, targets = raw_inputs.to(device), raw_targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


def hyperparameter_optimize(tuner_type: TunerType) -> None:
    start = perf_counter()
    ray.init(
        runtime_env={
            "working_dir": ".",
            "excludes": [".git", "test", "logs/*", "LICENSE", "README.md", ".venv"],
        },
    )
    match tuner_type:
        case TunerType.NES:
            distribution: Distribution = Distribution.get_random_ditribution()
            trial_states = generate_trial_states(10, distribution.get_new_hyper)
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
            trial_states = generate_trial_states(10)
            tuner = PBTTuner.options(  # type: ignore[call-arg]
                max_concurrency=5,
                num_cpus=1,
                resources={f"node:{get_head_node_address()}": 0.01},
            ).remote(trial_states, train_step, cifar10_data_loader_factory)

    ray.get(tuner.run.remote())  # type: ignore[call-arg]

    zip_logs_bytes: bytes = ray.get(tuner.get_zipped_log.remote())  # type: ignore[call-arg]

    zip_output_dir = f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
    Path(zip_output_dir).mkdir(parents=True, exist_ok=True)
    zip_output_path = Path(zip_output_dir) / "logs.zip"
    with Path(zip_output_path).open("wb") as f:
        f.write(zip_logs_bytes)

    unzip_file(zip_output_path, zip_output_dir)  # type: ignore[call-arg]

    ray.shutdown()
    log_dir = Path("~/Documents/workspace/shaogu/Ray_PBT/log").expanduser()
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "time.log"
    with log_file.open("a") as f:
        f.write(f"{tuner_type} use time: {perf_counter() - start}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tuner_type",
        type=str,
        choices=["NES", "PBT", "COM"],
        default="COM",
        dest="tuner_type",
        help="Select the Tuner type of NES, PBT or COM(comparison PBT and NES)",
    )
    args = parser.parse_args()
    tuner_type: TunerType | None = None
    if args.tuner_type == "COM":
        hyperparameter_optimize(tuner_type=TunerType.PBT)
        hyperparameter_optimize(tuner_type=TunerType.NES)
    elif args.tuner_type == "NES":
        tuner_type = TunerType.NES
        hyperparameter_optimize(tuner_type=tuner_type)
    elif args.tuner_type == "PBT":
        tuner_type = TunerType.PBT
        hyperparameter_optimize(tuner_type=tuner_type)
    else:
        error_message = f"Unknown tuner type: {args.tuner_type}"
        raise ValueError(error_message)


if __name__ == "__main__":
    main()
