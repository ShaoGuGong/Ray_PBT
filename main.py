import random
from itertools import islice
from typing import List

import ray
import torch
import torch.nn as nn

from src.config import STOP_ITERATION
from src.trial_state import TrialState
from src.tuner import Tuner
from src.utils import Hyperparameter, ModelType, get_head_node_address


def generate_trial_states(n: int = 1) -> List[TrialState]:
    return [
        TrialState(
            i,
            Hyperparameter(
                lr=random.uniform(0.001, 1),
                momentum=random.uniform(0.001, 1),
                batch_size=random.choice([64, 128, 256, 512, 1024]),
                model_type=ModelType.RESNET_18,
            ),
            stop_iteration=STOP_ITERATION,
        )
        for i in range(n)
    ]


def train_step(model, optimizer, train_loader, batch_size, device=torch.device("cpu")):
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)

    for inputs, targets in islice(train_loader, 1024 // batch_size):
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
    trial_states = generate_trial_states(20)
    tuner = Tuner.options(
        max_concurrency=3,
        num_cpus=1,
        resources={f"node:{get_head_node_address()}": 0.01},
    ).remote(trial_states, train_step)
    ray.get(tuner.run.remote())
