from collections import defaultdict
from typing import List

import ray

from trial import TrialScheduler
from trial_state import TrialState
from utils import TrainStepFunction
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
        self.scheduler = TrialScheduler(self.workers, trial_states)
        self.accuracy_table = defaultdict(float)

    def run(self) -> None:
        print("Tuner Start")
        self.scheduler.run()
        self.scheduler.get_workers_logs()
        print("Tuner End")

    def record_accuracy(self, accuracy: float, iteration: int) -> None:
        self.accuracy_table[iteration] = max(self.accuracy_table[iteration], accuracy)
        self.print_accuracy_table()

    def get_accuracy(self, iteration: int) -> float:
        return self.accuracy_table[iteration]

    def print_accuracy_table(self) -> None:
        print("┏━━━━━━━━━━━┳━━━━━━━━━━┓")
        print("┃ Iteration ┃ Accuracy ┃")
        print("┡━━━━━━━━━━━╇━━━━━━━━━━┩")
        for k, v in self.accuracy_table.items():
            print(f"│ {k:9d} │ {v:8.4f} │")
        print("└───────────┴──────────┘")
