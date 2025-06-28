import random
from collections import deque
from pathlib import Path

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
        "fitnesses",
        "population_size",
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
        self.population_size = 10
        self.fitnesses: deque[Fitness] = deque(maxlen=self.population_size)

    def run(self) -> None:
        super().run()
        log_dir = Path("~/Documents/workspace/shaogu/Ray_PBT/log").expanduser()
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "accuracy.log"
        with log_file.open("a") as f:
            f.write(f"Use NES Accuracy: {self.trial_result.history_best[0]:.6f}\n")

    def _get_quantile_trial(
        self,
        ratio: float = 0.25,
    ) -> tuple[list[TrialState], list[TrialState]]:
        return self.trial_result.get_quantile(ratio)

    def should_mutate_trial(self, trial_state: TrialState) -> bool:
        return True

    def mutate_trial(self, trial_state: TrialState) -> TrialState:
        self.fitnesses.append(
            Fitness(
                accuracy=trial_state.accuracy,
                hyperparameter=trial_state.hyperparameter,
            ),
        )

        self.logger.info(
            "Trial %d: 執行mutation, 原始超參數: %s",
            trial_state.id,
            trial_state.hyperparameter,
        )

        if len(self.fitnesses) >= self.population_size:
            self.ditribution.update_distribution(list(self.fitnesses))

        new_hyper = self.ditribution.get_new_hyper()
        trial_state.hyperparameter = new_hyper
        _, upper_quantile = self._get_quantile_trial()
        if upper_quantile:
            chose_trial = random.choice(upper_quantile)
            trial_state.checkpoint = chose_trial.checkpoint

        self.logger.info(
            "Trial-%d Iter-%d, 結束mutation, 新超參數: %s",
            trial_state.id,
            trial_state.iteration,
            trial_state.hyperparameter,
        )

        return trial_state
