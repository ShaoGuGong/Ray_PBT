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

        self.status = TrialStatus.PENDING
        self.resource_id = -1
        self.run_time = 0
        self.iteration = 0
        self.checkpoint = Checkpoint(
            self.model.state_dict(),
            self.optimizer.state_dict(),
            0,
        )

    def update_checkpoint(self, model, optimizer, iteration: int) -> None:
        self.checkpoint.model_state_dict = model.state_dict()
        self.checkpoint.optimzer_state_dict = optimizer.state_dict()
        self.iteration = iteration


# TODO: UNIFINISHED
@ray.remote
class Trial:
    def __init__(self, trial_state: TrialState) -> None:
        self.state = trial_state


class TrialRunner:
    def __init__(self, trial_states: list[TrialState], max_iteration: int) -> None:
        self.trial_states = trial_states
        self.max_iteration = max_iteration

    def get_remaining_generation(self) -> int:
        result = 0
        for trial_state in filter(
            lambda trial_state: trial_state.status != TrialStatus.TERMINAL,
            self.trial_states,
        ):
            result += self.max_iteration - trial_state.iteration

        return result


class TrialScheduler:
    def __init__(self, resources: list[Worker], stop_interval: int) -> None:
        self.available_resources = resources
        self.used_resources = []
        self.stop_interval = stop_interval

    def assign_trail_to_worker(
        self, trial: TrialState, remaining_generation
    ) -> Optional[Trial]:
        if len(self.available_resources) == 0:
            return None

        resource = self.available_resources.pop(0)

        # TODO: Unfinished assign function

        raise NotImplementedError
