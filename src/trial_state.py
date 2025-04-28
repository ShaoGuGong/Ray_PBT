from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch.optim as optim

from .config import (STOP_ACCURACY, TRIAL_PROGRESS_OUTPUT_PATH,
                     TRIAL_RESULT_OUTPUT_PATH)
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
    def __init__(self, top_k: int = 5, bottom_k: int = 5) -> None:
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

    def display_results(self, output_path: str = TRIAL_RESULT_OUTPUT_PATH) -> None:
        if not self.table:
            print("No results available")
            return
        try:
            with open(output_path, "w") as f:
                f.write(f"┏{'':━^13}┳{'':━^15}┳{'':━^35}┓\n")
                f.write(
                    f"┃{'Iteration':^13}┃{'Accuracy':^15}┃{'Hyperparameter':^35}┃\n"
                )
                f.write(f"┣{'':━^4}┳{'':━^8}╋{'':━^15}╋{'':━^35}┫\n")

                for iteration, row in self.table.items():
                    top = row[: self.top_k]

                    bot = row[-self.bottom_k :]
                    if len(row) <= self.top_k:
                        bot = []

                    for data in sorted(top, key=lambda x: x[0] * -1):
                        hyper = data[1]
                        accuracy = data[0]
                        output = f"lr:{hyper.lr:5.2f} momentum:{hyper.momentum:5.2f} batch:{hyper.batch_size:4}"

                        f.write(f"┃{'':^4}┃{'':^8}┃{accuracy:^15.6f}┃{output:^35}┃\n")

                    f.write(f"\033[s\033[{(len(top)+1)//2}A\033[8C")
                    f.write(f"{'top-'+str(self.top_k):^4}\033[u")

                    if self.bottom_k == 0 or len(bot) == 0:
                        f.write(f"\033[s\033[{(len(top)+len(bot)+2)//2}A\033[1C")
                        f.write(f"{iteration:^4}\033[u")
                        f.write(f"┣{'':━^4}╋{'':━^8}╋{'':━^15}╋{'':━^35}┫\n")
                        continue

                    f.write(f"┃{'':^4}┣{'':━^8}╋{'':━^15}╋{'':━^35}┫\n")
                    for data in sorted(bot, key=lambda x: x[0] * -1):
                        hyper = data[1]
                        accuracy = data[0]
                        output = f"lr:{hyper.lr:5.2f} momentum:{hyper.momentum:5.2f} batch:{hyper.batch_size:4}"

                        f.write(f"┃{'':^4}┃{'':^8}┃{accuracy:^15.6f}┃{output:^35}┃\n")
                    f.write(f"\033[s\033[{(len(bot)+1)//2}A\033[8C")
                    f.write(f"{'bot-'+str(self.bottom_k):^4}\033[u")
                    f.write(f"\033[s\033[{(len(top)+len(bot)+2)//2}A\033[1C")
                    f.write(f"{iteration:^4}\033[u")
                    f.write(f"┣{'':━^4}╋{'':━^8}╋{'':━^15}╋{'':━^35}┫\n")
                f.write(f"\033[A┗{'':━^4}┻{'':━^8}┻{'':━^15}┻{'':━^35}┛\n")

                f.write(
                    f"History Best: {self.history_best[0]} {self.history_best[1]}\n"
                )
        except Exception as e:
            print(f"{e}")

    def display_trial_progress(
        self, output_path: str = TRIAL_PROGRESS_OUTPUT_PATH
    ) -> None:
        try:
            with open(output_path, "w") as f:
                f.write(f"┏{'':━^7}┳{'':━^11}┳{'':━^11}┳{'':━^43}┳{'':━^7}┳{'':━^7}┓\n")
                f.write(
                    f"┃{'':^7}┃{'':^11}┃{'Worker':^11}┃{'Hyparameter':^43}┃{'':^7}┃{'':^7}┃\n"
                )
                f.write(
                    f"┃{'Trial':^7}┃{'Status':^11}┣{'':━^4}┳{'':━^6}╋{'':━^7}┳{'':━^10}┳{'':━^12}┳{'':━^11}┫{'Iter':^7}┃{'Acc':^7}┃\n"
                )
                f.write(
                    f"┃{'':^7}┃{'':^11}┃{'ID':^4}┃{'TYPE':^6}┃{'lr':^7}┃{'momentum':^10}┃{'batch size':^12}┃{'Model':^11}┃{'':^7}┃{'':^7}┃\n"
                )
                f.write(
                    f"┣{'':━^7}╋{'':━^11}╋{'':━^4}╋{'':━^6}╋{'':━^7}╋{'':━^10}╋{'':━^12}╋{'':━^11}╋{'':━^7}╋{'':━^7}┫\n"
                )

                for i in self.trial_progress.values():
                    worker_type = "None"
                    if i.worker_type == WorkerType.CPU:
                        worker_type = "CPU"
                    elif i.worker_type == WorkerType.GPU:
                        worker_type = "GPU"
                    h = i.hyperparameter
                    f.write(
                        f"┃{i.id:>7}┃{i.status:^11}┃{i.worker_id:>4}┃{worker_type:^6}┃{h.lr:>7.3f}┃{h.momentum:>10.3f}┃{h.batch_size:>12}┃{h.model_type:^11}┃{i.iteration:>7}┃{i.accuracy:>7.3f}┃\n"
                    )
                f.write(
                    f"┗{'':━^7}┻{'':━^11}┻{'':━^4}┻{'':━^6}┻{'':━^7}┻{'':━^10}┻{'':━^12}┻{'':━^11}┻{'':━^7}┻{'':━^7}┛\n"
                )

        except Exception as e:
            print(f"{e}")

    def to_json(self) -> str:
        import json

        return json.dumps(self.table)
