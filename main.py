import os
from datetime import datetime
from itertools import islice
from typing import List, Tuple

import ray
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.config import DATASET_PATH, STOP_ITERATION
from src.trial_state import TrialState
from src.tuner import Tuner
from src.utils import Hyperparameter, get_head_node_address, unzip_file


def cifar10_data_loader_factory(
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_dir = os.path.expanduser(DATASET_PATH)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(os.path.join(data_dir, "cifar-10-batches-py")):
        print(f"{os.path.join(data_dir, 'cifar-10-batches-py')} 不存在")
        torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=None
        )

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=test_transform
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
    )
    valid_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
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


def generate_trial_states(n: int = 1) -> List[TrialState]:
    return [
        TrialState(
            i,
            Hyperparameter.random(),
            stop_iteration=STOP_ITERATION,
        )
        for i in range(n)
    ]


def train_step(model, optimizer, train_loader, batch_size, device=torch.device("cpu")):
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)

    for inputs, targets in islice(train_loader, 1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    ray.init(
        runtime_env={
            "working_dir": ".",
            "excludes": [".git", "test", "logs/*", "LICENSE", "README.md"],
        }
    )
    trial_states = generate_trial_states(50)
    tuner = Tuner.options(  # type: ignore
        max_concurrency=16,
        num_cpus=1,
        resources={f"node:{get_head_node_address()}": 0.01},
    ).remote(trial_states, train_step, cifar10_data_loader_factory)
    ray.get(tuner.run.remote())

    zip_logs_bytes: bytes = ray.get(tuner.get_zipped_log.remote())

    zip_output_dir = f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
    os.makedirs(zip_output_dir, exist_ok=True)
    zip_output_path = os.path.join(zip_output_dir, "logs.zip")
    with open(zip_output_path, "wb") as f:
        f.write(zip_logs_bytes)

    unzip_file(zip_output_path, zip_output_dir)

    ray.shutdown()
