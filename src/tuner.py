import random
from dataclasses import replace
from typing import List

import ray

from trial import TrialScheduler
from trial_state import TrialResult, TrialState
from utils import Hyperparameter, ModelType, TrainStepFunction
from worker import generate_all_workers


@ray.remote
class Tuner:
    def __init__(
        self,
        trial_states: List[TrialState],
        train_step: TrainStepFunction,
    ) -> None:
        self.trial_states = trial_states

        print(f"總共{len(trial_states)} 個 Trial")
        print(*[t.hyperparameter for t in trial_states], sep="\n")

        self.workers = generate_all_workers(
            ray.get_runtime_context().current_actor, train_step=train_step
        )
        self.scheduler: TrialScheduler = TrialScheduler(self.workers, trial_states)
        self.trial_result: TrialResult = TrialResult()

    def run(self) -> None:
        print("Tuner Start")
        self.scheduler.run()
        self.scheduler.get_workers_logs()
        print("Tuner End")

    def update_trial_result(
        self, iteration: int, accuracy: float, hyperparameter: Hyperparameter
    ):
        self.trial_result.update_train_result(iteration, accuracy, hyperparameter)

    def mutation(self) -> Hyperparameter:
        hyperparameter = self.trial_result.get_history_best_result()[1]

        if hyperparameter is None:
            return Hyperparameter(
                lr=random.uniform(0.001, 1),
                momentum=random.uniform(0.001, 1),
                batch_size=random.choice([64, 128, 256, 512, 1024]),
                model_type=ModelType.RESNET_18,
            )

        mutation_options = (
            ("lr", random.uniform(0.001, 1)),
            ("momentum", random.uniform(0.001, 1)),
            ("batch_size", random.choice([64, 128, 256, 512, 1024])),
        )

        return replace(
            hyperparameter,
            **{k: v for k, v in random.sample(mutation_options, 2)},
        )

    def get_mean_accuracy(self, iteration):
        return self.trial_result.get_mean_accuray(iteration)
