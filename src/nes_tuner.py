from pathlib import Path
from time import sleep

import ray

from .trial_state import TrialState
from .tuner import Tuner
from .utils import (
    DataloaderFactory,
    Distribution,
    Fitness,
    TrainStepFunction,
)


@ray.remote
class NESTuner(Tuner):
    __slots__ = {
        "distribution",
    }

    def __init__(
        self,
        trial_states: list[TrialState],
        train_step: TrainStepFunction,
        dataloader_factory: DataloaderFactory,
        distribution: Distribution,
    ) -> None:
        super().__init__(
            trial_states,
            train_step,
            dataloader_factory,
        )
        self.ditribution: Distribution = distribution

    def run(self) -> None:
        super().run()
        log_dir = Path("~/Documents/workspace/shaogu/Ray_PBT/log").expanduser()
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "accuracy.log"
        with log_file.open("a") as f:
            f.write(f"Use NES Accuracy: {self.trial_result.history_best[0]:.6f}\n")

    def should_mutate_trial(self, _: TrialState) -> bool:  # type: ignore[override]
        sleep(4.0)
        return True

    def mutate_trial(self, trial_state: TrialState) -> TrialState:
        self.logger.info(
            "Trial %d: 執行mutation, 原始超參數: %s",
            trial_state.id,
            trial_state.hyperparameter,
        )

        accuracy_increment = (trial_state.accuracy - trial_state.previous_accuracy) / (
            1.0 - trial_state.previous_accuracy
        )
        fitness = Fitness(
            fitness=accuracy_increment,
            hyperparameter=trial_state.hyperparameter,
        )
        self.ditribution.update_distribution(fitness=fitness)
        new_hyper = self.ditribution.get_new_hyper()
        trial_state.hyperparameter = new_hyper

        trial_state.checkpoint = self.trial_result.history_best[2]

        self.logger.info(
            "Trial-%d Iter-%d, 結束mutation, 新超參數: %s",
            trial_state.id,
            trial_state.iteration,
            trial_state.hyperparameter,
        )

        return trial_state
