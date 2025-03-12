from dataclasses import dataclass
from typing import Callable, List

import ray
import torch.optim as optim

from utils import ModelType, TrialStatus, get_data_loader, get_model
from worker import WorkerState


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


# TODO: UNIFINISHED
class TrialState:
    def __init__(
        self,
        id: int,
        hyperparameter: Hyperparameter,
        stop_iteration: int,
    ) -> None:
        self.id = id
        self.hyperparameter = hyperparameter
        self.stop_iteration = stop_iteration
        self.status = TrialStatus.PENDING
        self.resource_id = -1
        self.run_time = 0
        self.iteration = 0

        model = get_model(self.hyperparameter.model_type)
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.hyperparameter.lr,
            momentum=self.hyperparameter.momentum,
        )

        self.checkpoint = Checkpoint(
            model.state_dict(),
            optimizer.state_dict(),
            0,
        )
        self.accuracy = 0
        self.stop_accuracy = 0.0

    def update_checkpoint(self, model, optimizer, iteration: int) -> None:
        self.checkpoint.model_state_dict = model.state_dict()
        self.checkpoint.optimzer_state_dict = optimizer.state_dict()
        self.iteration = iteration

    def __str__(self) -> str:
        result = (
            f"Trial: {self.hyperparameter}\n {self.status=}\n {self.stop_iteration=}"
        )
        return result


# TODO: UNIFINISHED
@ray.remote
class Trial:
    def __init__(
        self,
        trial_state: TrialState,
        worker: WorkerState,
        train_fn: Callable,
    ) -> None:
        self.trial_state = trial_state
        self.worker = worker
        self.train_fn = train_fn
        self.mutation_iteration = 10

        hyperparameter = self.trial_state.hyperparameter
        self.model = get_model(hyperparameter.model_type)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=hyperparameter.lr,
            momentum=hyperparameter.momentum,
        )
        self.train_loader, self.test_loader = get_data_loader(
            hyperparameter.model_type, hyperparameter.batch_size
        )

    def train(self) -> float:
        for _ in range(self.mutation_iteration):
            self.train_fn(self.trial_state)
        return 8.7


class TrialScheduler:
    def __init__(
        self,
        trial_states: List[TrialState],
        resources: List[WorkerState],
        stop_interation: int,
        max_iteration: int,
    ) -> None:
        self.available_resources = resources
        self.used_resources = []
        self.max_iteration = max_iteration
        self.trial_states = trial_states
        self.running_trials = []

    def get_remaining_generation(self) -> int:
        result = 0

        for trial_state in filter(
            lambda trial_state: trial_state.status != TrialStatus.TERMINAL,
            self.trial_states,
        ):
            result += self.max_iteration - trial_state.iteration

        return result

    # TODO: Unfinished assign function
    def assign_trail_to_worker(self) -> None:
        if len(self.available_resources) == 0:
            return

        resource = self.available_resources.pop(0)
        self.used_resources.append(resource)

        trial = Trial.options(
            num_cpus=resource.num_cpus,
            num_gpus=resource.num_gpus,
            resources={resource.node_name: 0.001},
        ).remote()

        self.running_trials.append(trial)
        self.used_resources.append(resource)

    def get_pending_trials(self, is_sorted: bool = False) -> List[TrialState]:
        return [
            trial_state
            for trial_state in self.trial_states
            if trial_state.status == TrialStatus.PENDING
        ]

    def is_finished(self):
        return all(trial.status == TrialStatus.TERMINAL for trial in self.trial_states)
