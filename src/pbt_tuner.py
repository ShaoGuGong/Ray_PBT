import random
from itertools import repeat
from pathlib import Path

import numpy as np
import ray

from .trial_state import TrialState
from .tuner import Tuner
from .utils import Hyperparameter, ModelType


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
        trial_state.checkpoint = chose_trial.checkpoint

        r = random.randint(1, 100)
        if r <= 25:
            trial_state.hyperparameter = Hyperparameter.random()
        else:
            hyper = chose_trial.hyperparameter
            perturbation = np.array(
                random.choice([0.8, 1.2, 1.0, 1.0]) for _ in repeat(None, 4)
            )
            hyperparameter = hyper * perturbation
            trial_state.hyperparameter = Hyperparameter(
                *hyperparameter,
                batch_size=512,
                model_type=ModelType.RESNET_18,
            )

        self.logger.info(
            "Trial-%d Iter-%d, 結束mutation, 新超參數: %s",
            trial_state.id,
            trial_state.iteration,
            trial_state.hyperparameter,
        )

        return trial_state
