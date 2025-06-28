import random
from pathlib import Path

import ray

from .trial_state import TrialState
from .tuner import Tuner


# NOTE:
# model 的建立的時間,
# batch_size 對於 throughput 計算
@ray.remote
class PBTTuner(Tuner):
    def run(self) -> None:
        super().run()
        log_dir = Path("~/Documents/workspace/shaogu/Ray_PBT/log").expanduser()
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "accuracy.log"
        with log_file.open("a") as f:
            f.write(f"Use PBT Accuracy: {self.trial_result.history_best[0]:.6f}\n")

    def _get_quantile_trial(
        self,
        ratio: float = 0.25,
    ) -> tuple[list[TrialState], list[TrialState]]:
        return self.trial_result.get_quantile(ratio)

    def should_mutate_trial(self, trial_state: TrialState) -> bool:
        lower_quantile, _ = self._get_quantile_trial()
        return trial_state in lower_quantile

    def mutate_trail(self, trial_state: TrialState) -> TrialState:
        self.logger.info(
            "Trial %d: 執行mutation, 原始超參數: %s",
            trial_state.id,
            trial_state.hyperparameter,
        )

        _, upper_quantile = self._get_quantile_trial()

        chose_trial = random.choice(upper_quantile)
        hyperparameter = chose_trial.hyperparameter
        hyperparameter.lr *= 0.8
        hyperparameter.momentum *= 1.2

        trial_state.hyperparameter = hyperparameter
        trial_state.checkpoint = chose_trial.checkpoint

        self.logger.info(
            "Trial-%d Iter-%d, 結束mutation, 新超參數: %s",
            trial_state.id,
            trial_state.iteration,
            trial_state.hyperparameter,
        )

        return trial_state
