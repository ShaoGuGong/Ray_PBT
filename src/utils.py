import os
from dataclasses import dataclass
from enum import Enum, auto

import torch
import torchvision
import torchvision.transforms as transforms
import numpy

# ╭──────────────────────────────────────────────────────────╮
# │                          Enums                           │
# ╰──────────────────────────────────────────────────────────╯


class ModelType(Enum):
    RESNET_18 = auto()
    RESNET_50 = auto()


# ╭──────────────────────────────────────────────────────────╮
# │                       Dataclasses                        │
# ╰──────────────────────────────────────────────────────────╯


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


def fetch_nodes_resources():
    raise NotImplementedError


def get_data_loader(
    model_type: ModelType, batch_size: int = 64, transform=None, data_dir="../dataset/"
) -> tuple:
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    if model_type == ModelType.RESNET_18:
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root=data_dir, train=True, download=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root=data_dir, train=False, download=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        return train_loader, test_loader

    elif model_type == ModelType.RESNET_50:
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(
                root=data_dir, train=True, download=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(
                root=data_dir, train=False, download=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        return train_loader, test_loader

    raise ValueError("`model_type` must be in ModelType")
