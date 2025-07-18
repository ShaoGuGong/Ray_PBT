from datetime import UTC, datetime, timedelta
from itertools import islice
from pathlib import Path

import ray
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

from src.config import DATASET_PATH, STOP_ITERATION
from src.trial_state import TrialState
from src.tuner import Tuner
from src.utils import Checkpoint, Hyperparameter, get_head_node_address, unzip_file

DEFAULT_DEVICE = torch.device("cpu")


def cifar10_data_loader_factory(
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
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

    # np.random.seed(42)
    # indices = np.random.permutation(len(test_dataset))
    # test_size = len(test_dataset) // 2
    # test_indices = indices[:test_size]
    # valid_indices = indices[test_size:]
    #
    # valid_dataset = Subset(test_dataset, valid_indices.tolist())
    # test_dataset = Subset(test_dataset, test_indices.tolist())

    # print(f"{len(train_dataset)=}, {len(valid_dataset)=}, {len(test_dataset)=}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # valid_loader = DataLoader(
    #     valid_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    # )
    #
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    # )

    # return train_loader, valid_loader, test_loader
    return train_loader, valid_loader, None


def resnet18_init_fn(
    hyperparameter: Hyperparameter,
    checkpoint: Checkpoint,
    device: torch.device,
) -> tuple[nn.Module, optim.Optimizer]:
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)

    if checkpoint.is_empty():
        model.to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=hyperparameter.lr,
            momentum=hyperparameter.momentum,
        )

        return model, optimizer

    model.load_state_dict(checkpoint.model_state_dict)
    model = model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=hyperparameter.lr,
        momentum=hyperparameter.momentum,
    )
    optimizer.load_state_dict(checkpoint.optimizer_state_dict)

    for param_group in optimizer.param_groups:
        param_group["lr"] = hyperparameter.lr
        param_group["momentum"] = hyperparameter.momentum

    return model, optimizer


def generate_trial_states(n: int = 1) -> list[TrialState]:
    return [
        TrialState(
            i,
            Hyperparameter.random(),
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
    device: torch.device = DEFAULT_DEVICE,
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


if __name__ == "__main__":
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
                ".ruff_cache",
            ],
        },
    )
    trial_states = generate_trial_states(50)
    tuner = Tuner.options(  # type: ignore[call-arg]
        max_concurrency=13,
        num_cpus=1,
        resources={f"node:{get_head_node_address()}": 0.01},
    ).remote(trial_states, train_step, cifar10_data_loader_factory)
    ray.get(tuner.run.remote())  # type: ignore[call-arg]

    zip_logs_bytes: bytes = ray.get(tuner.get_zipped_log.remote())  # type: ignore[call-arg]

    time_stamp = (datetime.now(UTC) + timedelta(hours=8)).strftime("%Y-%m-%d_%H-%M-%S")
    zip_output_dir = f"./logs/{time_stamp}/"

    Path(zip_output_dir).mkdir(parents=True, exist_ok=True)
    zip_output_path = Path(zip_output_dir) / "logs.zip"
    with Path(zip_output_path).open("wb") as f:
        f.write(zip_logs_bytes)

    unzip_file(zip_output_path, zip_output_dir)  # type: ignore[call-arg]

    ray.shutdown()
