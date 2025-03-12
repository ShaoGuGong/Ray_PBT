from typing import List

from trial import Hyperparameter, TrialScheduler, TrialState
from worker import generate_all_workers


class Tuner:
    def __init__(
        self, hyperparameters: List[Hyperparameter], stop_iteration: int = 1000
    ) -> None:
        self.workers = generate_all_workers()
        self.hyperparemeters = hyperparameters
        self.stop_iteration = stop_iteration
        self.num_hyperparameters = len(self.hyperparemeters)
        self.package_size = (len(self.workers) - self.num_hyperparameters) // 3
        self.trial_states = self._generate_trail_states()
        self.max_iteration = 1000
        self.trial_scheduler = TrialScheduler(self.hyperparemeters, self.workers)

    def _generate_trail_states(self) -> List[TrialState]:
        result = []

        for index, hyperparameter in enumerate(self.hyperparemeters):
            result.append(TrialState(index, hyperparameter, self.stop_iteration))

        return result

    def run(self) -> None:

        return
