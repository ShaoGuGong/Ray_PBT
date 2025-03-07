import os
import sys
import time
from typing import List

import ray
import torch.nn as nn
from numpy import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from trial import Hyperparameter, TrialScheduler, TrialState
from utils import ModelType, get_data_loader
from worker import generate_all_workers


def train_step(model, optimizer, train_loader, device=None):
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        break


@ray.remote
def test():
    return os.listdir(os.getcwd())


def generate_trial_states(n: int = 10) -> List[TrialState]:
    return [
        TrialState(
            i,
            Hyperparameter(
                lr=random.uniform(0.001, 1),
                momentum=random.uniform(0.001, 1),
                batch_size=random.choice([64, 128, 256, 512]),
                model_type=ModelType.RESNET_18,
            ),
        )
        for i in range(n)
    ]


if __name__ == "__main__":
    ray.init(runtime_env={"working_dir": "./src"})

    print(ray.get(test.remote()))

    workers = generate_all_workers()
    print(*workers, sep="\n")
    worker = workers[0]

    trial_states = generate_trial_states()
    print(f"總共{len(trial_states)} 個 Trial")
    print([f"Trial {i.id}" for i in trial_states])

    scheduler = TrialScheduler(trial_states, workers)
    scheduler.run()
