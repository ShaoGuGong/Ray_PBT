import os
from datetime import datetime
from itertools import islice
from typing import List

import ray
import torch
import torch.nn as nn

from src.config import STOP_ITERATION
from src.trial_state import TrialState
from src.tuner import Tuner
from src.utils import Hyperparameter, get_head_node_address, unzip_file


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
    trial_states = generate_trial_states(40)
    tuner = Tuner.options(  # type: ignore
        max_concurrency=3,
        num_cpus=1,
        resources={f"node:{get_head_node_address()}": 0.01},
    ).remote(trial_states, train_step)
    ray.get(tuner.run.remote())

    zip_logs_bytes: bytes = ray.get(tuner.get_zipped_log.remote())

    zip_output_dir = f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
    os.makedirs(zip_output_dir, exist_ok=True)
    zip_output_path = os.path.join(zip_output_dir, "logs.zip")
    with open(zip_output_path, "wb") as f:
        f.write(zip_logs_bytes)

    unzip_file(zip_output_path, zip_output_dir)
