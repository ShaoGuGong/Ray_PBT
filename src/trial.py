from dataclasses import dataclass
from typing import Optional

import ray
import torch.optim as optim

from utils import ModelType, TrialStatus, get_model
from worker import Worker


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
        hyperparameter: Hyperparameter,
        model=get_model(ModelType.RESNET_18),
        stop_iteration:int,
        optimizer=None,
    ) -> None:
        self.hyperparameter = hyperparameter
        self.model = model

        if optimizer is None:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.hyperparameter.lr,
                momentum=self.hyperparameter.momentum,
            )
        self.optimizer = optimizer

        self.stop_iteration = stop_iteration
        self.status = TrialStatus.PENDING
        self.resource_id = -1
        self.run_time = 0
        self.iteration = 0
        self.checkpoint = Checkpoint(
            self.model.state_dict(),
            self.optimizer.state_dict(),
            0,
        )
        self.accuracy = 0
        self.stop_accuracy = 0.0

    def update_checkpoint(self, model, optimizer, iteration: int) -> None:
        self.checkpoint.model_state_dict = model.state_dict()
        self.checkpoint.optimzer_state_dict = optimizer.state_dict()
        self.iteration = iteration


# TODO: UNIFINISHED
@ray.remote
class Trial:
    def __init__(self, trial_state: TrialState, worker: Worker) -> None:
        self.state = trial_state
        self.worker = worker
        raise NotImplementedError

    def update_calculate_ability(self):
        pass

    def update_state(self):
        check = 0
        if self.state.stop_iteration:
            if self.state.iteration < self.state.stop_iteration:
                check += 1
            else:
                check = -9  # Termination condition met

        if self.state.stop_accuracy != 1:
            if accuracys[i] < self.stop_acc:
                check += 1
            else:
                check = -9  # Termination condition met

        # WARN:
        if check > 0:  # Add IDs that still need training to the scheduler
            self.state.status = TrialStatus.PENDING

        elif check < 0:
            self.state.status = TrialStatus.TERMINAL
        else:
            print("No end condition!!")  # No termination condition set
            exit(0)


class TrialRunner:
    def __init__(self):
        pass


class TrialScheduler:
    def __init__(
        self,
        trial_states: list[TrialState],
        resources: list[Worker],
        stop_interation: int,
        max_iteration: int,
    ) -> None:
        self.available_resources = resources
        self.used_resources = []
        self.max_iteration = max_iteration
        self.trial_states = trial_states

    def get_remaining_generation(self) -> int:
        result = 0

        for trial_state in filter(
            lambda trial_state: trial_state.status != TrialStatus.TERMINAL,
            self.trial_states,
        ):
            result += self.max_iteration - trial_state.iteration

        return result

    def assign_trail_to_worker(self) -> Optional[Trial]:
        remaining_generation = self.get_remaining_generation()

        if len(self.available_resources) == 0:
            return None

        resource = self.available_resources.pop(0)
        pending_trials = self.get_pending_trials()

        # TODO: Unfinished assign function

        raise NotImplementedError

    def get_pending_trials(self, is_sorted: bool = False) -> list[TrialState]:
        return [
            trial_state
            for trial_state in self.trial_states
            if trial_state.status == TrialStatus.PENDING
        ]
