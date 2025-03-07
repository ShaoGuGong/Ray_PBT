import asyncio
import logging
from dataclasses import dataclass
from typing import List

import ray
import torch.optim as optim
from ray import ObjectRef
from ray.actor import ActorHandle

from utils import ModelType, TrialStatus, get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,%(msecs)03d %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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


class TrialState:
    def __init__(
        self,
        id: int,
        hyperparameter: Hyperparameter,
        stop_iteration: int = 1000,
    ) -> None:
        self.id = id
        self.hyperparameter = hyperparameter
        self.stop_iteration = stop_iteration
        self.status = TrialStatus.PENDING
        self.worker_id = -1
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
        self.accuracy = 0.0
        self.stop_accuracy = 0.0

    def update_checkpoint(self, model, optimizer, iteration: int) -> None:
        self.checkpoint.model_state_dict = model.state_dict()
        self.checkpoint.optimzer_state_dict = optimizer.state_dict()
        self.iteration = iteration

    def __str__(self) -> str:
        result = f"Trial: {str(self.hyperparameter)}\n {self.status=}\n {self.stop_iteration=}"
        return result


class TrialScheduler:
    def __init__(
        self, trial_states: List[TrialState], workers: List[ActorHandle]
    ) -> None:
        self.workers = workers
        self.trial_states = trial_states
        self.running_futures: List[ObjectRef] = []
        self.completed_trial_state = []

    def assign_trial_to_worker(self) -> List[ObjectRef]:
        available_futures = [
            worker.has_available_slots.remote() for worker in self.workers
        ]

        available_workers = [
            worker
            for worker, is_available in zip(self.workers, ray.get(available_futures))
            if is_available
        ]

        if not available_workers:
            logging.info("No available workers.")
            return self.running_futures

        for worker in available_workers:
            if not self.trial_states:
                break

            trial = self.trial_states.pop(0)
            future = worker.assign_trial.remote(trial)
            self.running_futures.append(future)

        return self.running_futures

    def run(self):
        loop = asyncio.get_event_loop()

        while self.running_futures or self.trial_states:
            self.assign_trial_to_worker()

            if not self.running_futures:
                break

            done_futures, self.running_futures = ray.wait(
                self.running_futures, timeout=1.0
            )

            loop.run_until_complete(self.handle_done_futures(done_futures))

        logging.info("🎉 所有 Trial 訓練完成！")

    async def handle_done_futures(self, done_futures: List[ObjectRef]):
        for future in done_futures:
            trial_state = ray.get(future)
            self.completed_trial_state.append(trial_state)
            logging.info(
                f"✅ Worker {trial_state.worker_id } 完成 Trial {trial_state.id} ，Accuracy: {trial_state.accuracy:.2f}"
            )
