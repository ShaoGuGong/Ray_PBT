from pathlib import Path

import ray

from .trial_state import TrialState
from .tuner import Tuner
from .utils import DataloaderFactory, TrainStepFunction
from .utils_nes import DistributionManager, Fitness


@ray.remote
class GroupNESTuner(Tuner):
    __slots__ = ("distribution_manager",)

    def __init__(
        self,
        trial_states: list[TrialState],
        train_step: TrainStepFunction,
        dataloader_factory: DataloaderFactory,
        distribution_manager: DistributionManager,
    ) -> None:
        super().__init__(
            trial_states,
            train_step,
            dataloader_factory,
        )
        self.distribution_manager: DistributionManager = distribution_manager

    def run(self) -> None:
        super().run()
        history_best = self.trial_result.get_history_best_result()
        log_dir = Path("~/Documents/workspace/shaogu/Ray_PBT/log").expanduser()
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "accuracy.log"
        with log_file.open("a") as f:
            f.write(f"Use NES Accuracy: {history_best[0]:.6f}\n")

    def should_mutate_trial(  # type: ignore[override]
        self,
        _: TrialState,
    ) -> bool:
        return True

    def mutate_trial(self, trial_state: TrialState) -> TrialState:
        self.logger.info(
            "Trial %d: 執行mutation, 原始超參數: %s",
            trial_state.id,
            trial_state.hyperparameter,
        )

        accuracy_increment = (
            trial_state.accuracy - trial_state.previous_accuracy
        ) / (1.0 - trial_state.previous_accuracy)
        fitness = Fitness(
            fitness=accuracy_increment,
            hyperparameter=trial_state.hyperparameter,
        )
        self.distribution_manager.update_distribution(
            trial_id=trial_state.id,
            fitness=fitness,
        )
        new_hyper = self.distribution_manager.get_new_hyper(
            trial_id=trial_state.id,
        )
        trial_state.hyperparameter = new_hyper

        self.logger.info(
            "Trial-%d Iter-%d, 結束mutation, 新超參數: %s",
            trial_state.id,
            trial_state.iteration,
            trial_state.hyperparameter,
        )

        return trial_state
