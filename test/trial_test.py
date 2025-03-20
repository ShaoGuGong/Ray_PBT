import os
import random
import sys
import time
from itertools import islice
from typing import List

import numpy as np
import ray
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from trial import TrialScheduler, TrialState
from utils import Hyperparameter, ModelType
from worker import generate_all_workers


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


@ray.remote
def test():
    return os.listdir(os.getcwd())


def generate_trial_states(n: int = 0) -> List[TrialState]:
    return [
        TrialState(
            i,
            Hyperparameter(
                lr=10 ** random.uniform(-2, -4),
                momentum=random.uniform(0.001, 1),
                batch_size=random.choice([64, 128, 256, 512, 1024]),
                model_type=ModelType.RESNET_18,
            ),
            stop_iteration=1,
        )
        for i in range(n)
    ]


@ray.remote
class Test:
    def __init__(self) -> None:
        print("Test Start")
        pass

    def run(self):
        time.sleep(10)


if __name__ == "__main__":
    # ray.init(num_cpus=4, runtime_env={"working_dir": "./src", "exclude": ["logs/"]})
    #
    # trial_states = generate_trial_states()
    # print(f"總共{len(trial_states)} 個 Trial")
    # print(*[t.hyperparameter for t in trial_states], sep="\n")
    #
    # scheduler = TrialScheduler(train_step, trial_states)
    # scheduler.run()
    # scheduler.get_workers_logs()
    head_node_address = ray.get_runtime_context().gcs_address.split(":")[0]

    t = Test.options(num_cpus=1, resources={f"node:{head_node_address}": 0.01}).remote()
    t.run.remote()
    generate_all_workers(train_step)
    t.run.remote()
