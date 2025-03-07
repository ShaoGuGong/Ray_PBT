import random
import os
import time
from dataclasses import dataclass
from typing import Callable, List

import ray
import torch
from ray.actor import ActorHandle

from trial import TrialState
from utils import TrialStatus

# Type Define
Accuracy = float


@dataclass
class WorkerState:
    id: int
    num_cpus: int
    num_gpus: int
    node_name: str
    calculate_ability: float = 0.0
    max_trials: int = 3


@ray.remote
class Worker:
    def __init__(self, worker_state: WorkerState, train_step: Callable) -> None:
        self.worker_state = worker_state
        self.active_trials = {}
        self.train_step = train_step

    def assign_trial(self, trial_state: TrialState) -> TrialState:
        self.active_trials[trial_state.id] = trial_state
        trial_state.worker_id = self.worker_state.id
        return self.train(trial_state)

    def get_active_trials_nums(self) -> int:
        return len(self.active_trials)

    def has_available_slots(self) -> bool:
        # print(f"目前有{len(self.active_trials)}個Trials")
        return len(self.active_trials) < self.worker_state.max_trials

    def train(self, trial_state: TrialState) -> TrialState:
        print(f"Worker {self.worker_state.id} 開始訓練 Trial {trial_state.id}")

        for i in range(10):
            # print(f"[{trial_state.id}]: {i+1} s")
            time.sleep(1)
        trial_state.accuracy = random.randint(10, 90) / 100
        self.active_trials.pop(trial_state.id)
        print(f"Trial {trial_state.id} 訓練結束")

        self.finish_trial(trial_state)

        return trial_state

    def finish_trial(self, trial_state: TrialState) -> None:
        # trial_state.worker_id = -1
        trial_state.status = TrialStatus.TERMINAL


def generate_all_workers() -> List[ActorHandle]:
    visited_address = set()
    worker_states = []
    index = 0

    # Fetch all avaiable resource from Ray cluster.
    for node in ray.nodes():
        if node["Alive"]:
            if node["NodeManagerAddress"] in visited_address:
                continue

            resource = node["Resources"]
            if "CPU" in resource:
                worker_states.append(
                    WorkerState(
                        id=index,
                        num_cpus=resource.get("CPU", 0),
                        num_gpus=0,
                        node_name=f"node:{node['NodeManagerAddress']}",
                    )
                )
                index += 1
            if "GPU" in resource:
                worker_states.append(
                    WorkerState(
                        id=index,
                        num_cpus=0,
                        num_gpus=resource.get("GPU", 0),
                        node_name=f"node:{node['NodeManagerAddress']}",
                    )
                )
                index += 1
            visited_address.add(node["NodeManagerAddress"])

    workers: list[ActorHandle] = []
    print(*worker_states)

    for index, worker_state in enumerate(worker_states):
        workers.append(
            Worker.options(
                max_concurrency=worker_state.max_trials,
                name=f"worker-{index}",
                num_cpus=worker_state.num_cpus,
                num_gpus=worker_state.num_gpus,
                resources={worker_state.node_name: 0.01},
            ).remote(worker_state, train_step=None)
        )

    return workers
