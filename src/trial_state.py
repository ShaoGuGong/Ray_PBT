from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch.optim as optim

from .config import STOP_ACCURACY
from .utils import (Checkpoint, Hyperparameter, TrialStatus, WorkerType,
                    get_model)


class TrialState:
    def __init__(
        self,
        id: int,
        hyperparameter: Hyperparameter,
        stop_iteration: int = 10,
    ) -> None:
        self.id = id
        self.hyperparameter = hyperparameter
        self.stop_iteration = stop_iteration
        self.status = TrialStatus.PENDING
        self.worker_id = -1
        self.worker_type: Optional[WorkerType] = None
        self.run_time = 0
        self.iteration = 0
        self.device_iteration_count = {WorkerType.CPU: 0, WorkerType.GPU: 0}

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
        self.stop_accuracy = STOP_ACCURACY

    def __str__(self) -> str:
        result = f"{'Trial: '+str(self.id):=^40}\n Hyperparameter: {str(self.hyperparameter)}\n status: {self.status}\n iteration: {self.iteration} \n stop_iteration: {self.stop_iteration} \n{'':=^40}"
        return result


class TrialResult:
    def __init__(self, top_k: int = 1, bottom_k: int = 1) -> None:
        self.table: Dict[int, List[Tuple[float, Hyperparameter]]] = defaultdict(list)
        self.top_k: int = top_k
        self.bottom_k: int = bottom_k
        self.history_best: Tuple[
            float, Optional[Hyperparameter], Optional[Checkpoint]
        ] = (0.0, None, None)
        self.trial_state_history: Dict[int, Dict] = {}
        self.trial_progress: Dict[int, TrialState] = {}

    def record_trial_progress(self, trial_state: TrialState) -> None:
        self.trial_progress[trial_state.id] = trial_state

    def update_trial_result(self, trial_state: TrialState) -> None:
        self.record_trial_progress(trial_state)
        self.table[trial_state.iteration].append(
            (trial_state.accuracy, trial_state.hyperparameter)
        )
        self.table[trial_state.iteration].sort(key=lambda x: x[0], reverse=True)
        if trial_state.accuracy > self.history_best[0]:
            self.history_best = (
                trial_state.accuracy,
                trial_state.hyperparameter,
                trial_state.checkpoint,
            )

        self.display_results()

    def get_mean_accuray(self, iteration: int) -> float:
        return sum([i[0] for i in self.table[iteration]]) / len(self.table[iteration])

    def get_top_k_result(self, iteration: int) -> List[Tuple[float, Hyperparameter]]:
        return self.table[iteration][: self.top_k]

    def get_history_best_result(
        self,
    ) -> Tuple[float, Optional[Hyperparameter], Optional[Checkpoint]]:
        return self.history_best

    def display_results(self) -> None:
        if not self.table:
            print("No results available")
            return

        print(f"┏{'':━^13}┳{'':━^15}┳{'':━^35}┓")
        print(f"┃{'Iteration':^13}┃{'Accuracy':^15}┃{'Hyperparameter':^35}┃")
        print(f"┣{'':━^4}┳{'':━^8}╋{'':━^15}╋{'':━^35}┫")

        for iteration, row in self.table.items():
            top = row[: self.top_k]

            bot = row[-self.bottom_k :]
            if len(row) <= self.top_k:
                bot = []

            for data in sorted(top, key=lambda x: x[0] * -1):
                hyper = data[1]
                accuracy = data[0]
                output = f"lr:{hyper.lr:5.2f} momentum:{hyper.momentum:5.2f} batch:{hyper.batch_size:4}"

                print(f"┃{'':^4}┃{'':^8}┃{accuracy:^15.6f}┃{output:^35}┃")

            print(f"\033[s\033[{(len(top)+1)//2}A\033[8C", end="")
            print(f"{'top-'+str(self.top_k):^4}\033[u", end="")

            if self.bottom_k == 0 or len(bot) == 0:
                print(f"\033[s\033[{(len(top)+len(bot)+2)//2}A\033[1C", end="")
                print(f"{iteration:^4}\033[u", end="")
                print(f"┣{'':━^4}╋{'':━^8}╋{'':━^15}╋{'':━^35}┫")
                continue

            print(f"┃{'':^4}┣{'':━^8}╋{'':━^15}╋{'':━^35}┫")
            for data in sorted(bot, key=lambda x: x[0] * -1):
                hyper = data[1]
                accuracy = data[0]
                output = f"lr:{hyper.lr:5.2f} momentum:{hyper.momentum:5.2f} batch:{hyper.batch_size:4}"

                print(f"┃{'':^4}┃{'':^8}┃{accuracy:^15.6f}┃{output:^35}┃")
            print(f"\033[s\033[{(len(bot)+1)//2}A\033[8C", end="")
            print(f"{'bot-'+str(self.bottom_k):^4}\033[u", end="")
            print(f"\033[s\033[{(len(top)+len(bot)+2)//2}A\033[1C", end="")
            print(f"{iteration:^4}\033[u", end="")
            print(f"┣{'':━^4}╋{'':━^8}╋{'':━^15}╋{'':━^35}┫")
        print(f"\033[A┗{'':━^4}┻{'':━^8}┻{'':━^15}┻{'':━^35}┛")

        print(f"History Best: {self.history_best[0]} {self.history_best[1]}")

    def to_json(self) -> str:
        import json

        return json.dumps(self.table)
