from trial import Hyperparameter, TrialState
from worker import generate_all_workers


class ModelInitializer:
    pass


class Tuner:
    def __init__(self, hyperparameters: list[Hyperparameter]) -> None:
        self.workers = generate_all_workers()
        self.hyperparemeters = hyperparameters
        self.num_hyperparameters = len(self.hyperparemeters)
        self.package_size = (len(self.workers) - self.num_hyperparameters) // 3
        self.trial_states = self._generate_trail_states()
        self.max_iteration = 1000

    def run() -> None:
        pass

    def _generate_trail_states(self) -> list[TrialState]:
        result = []

        for hyperparameter in self.hyperparemeters:
            result.append(TrialState(hyperparameter))

        return result
