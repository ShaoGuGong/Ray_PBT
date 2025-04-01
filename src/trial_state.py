from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch.optim as optim

from utils import Checkpoint, Hyperparameter, TrialStatus, get_model


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
        self.run_time = 0
        self.iteration = 0

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
        self.stop_accuracy = 0.5

    def update_checkpoint(self, model, optimizer, iteration: int) -> None:
        self.checkpoint.model_state_dict = model.state_dict()
        self.checkpoint.optimzer_state_dict = optimizer.state_dict()
        self.iteration = iteration

    def __str__(self) -> str:
        result = f"Trial: {str(self.hyperparameter)}\n {self.status=}\n {self.stop_iteration=}"
        return result


class TrialResult:
    def __init__(self, top_k: int = 1, bottom_k: int = 1) -> None:
        self.table: Dict[int, List[Tuple[float, Hyperparameter]]] = defaultdict(list)
        self.top_k: int = top_k
        self.bottom_k: int = bottom_k
        self.history_best: Tuple[float, Optional[Hyperparameter]] = (0.0, None)

    def update_train_result(
        self, iteration: int, accuracy: float, hyperparameter: Hyperparameter
    ) -> None:
        self.table[iteration].append((accuracy, hyperparameter))
        self.table[iteration].sort(reverse=True)
        if accuracy > self.history_best[0]:
            self.history_best = (accuracy, hyperparameter)

    def get_mean_accuray(self, iteration: int) -> float:
        return sum([i[0] for i in self.table[iteration]]) / len(self.table[iteration])

    def get_top_k_result(self, iteration: int) -> List[Tuple[float, Hyperparameter]]:
        return self.table[iteration][: self.top_k]

    def get_history_best_result(self) -> Tuple[float, Optional[Hyperparameter]]:
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

            for data in sorted(top, key=lambda x: x[0] * -1):
                hyper = data[1]
                accuracy = data[0]
                output = f"lr:{hyper.lr:5.2f} momentum:{hyper.momentum:5.2f} batch:{hyper.batch_size:4}"

                print(f"┃{'':^4}┃{'':^8}┃{accuracy:^15.6f}┃{output:^35}┃")

            print(f"\033[s\033[{(self.top_k+1)//2}A\033[8G", end="")
            print(f"{'top-'+str(self.top_k):^4}\033[u", end="")

            if self.bottom_k == 0:
                print(f"\033[s\033[{(self.top_k+self.bottom_k+2)//2}A\033[2G", end="")
                print(f"{iteration:^4}\033[u", end="")
                print(f"┣{'':━^4}╋{'':━^8}╋{'':━^15}╋{'':━^35}┫")
                continue

            print(f"┃{'':^4}┣{'':━^8}╋{'':━^15}╋{'':━^35}┫")
            for data in sorted(bot, key=lambda x: x[0] * -1):
                hyper = data[1]
                accuracy = data[0]
                output = f"lr:{hyper.lr:5.2f} momentum:{hyper.momentum:5.2f} batch:{hyper.batch_size:4}"

                print(f"┃{'':^4}┃{'':^8}┃{-accuracy:^15.6f}┃{output:^35}┃")
            print(f"\033[s\033[{(self.bottom_k+1)//2}A\033[8G", end="")
            print(f"{'bot':^4}\033[u", end="")
            print(f"\033[s\033[{(self.top_k+self.bottom_k+2)//2}A\033[2G", end="")
            print(f"{iteration:^4}\033[u", end="")
            print(f"┣{'':━^4}╋{'':━^8}╋{'':━^15}╋{'':━^35}┫")
        print(f"\033[A\033[0G┗{'':━^4}┻{'':━^8}┻{'':━^15}┻{'':━^35}┛")

    def to_json(self) -> str:
        import json

        return json.dumps(self.table)
