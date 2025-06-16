import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn, optim

from .config import (
    STOP_ACCURACY,
    STOP_ITERATION,
    TRIAL_PROGRESS_OUTPUT_PATH,
    TRIAL_RESULT_OUTPUT_PATH,
)
from .utils import (
    Checkpoint,
    Hyperparameter,
    ModelInitFunction,
    TrialStatus,
    WorkerType,
)


class TrialState:
    def __init__(
        self,
        trial_id: int,
        hyperparameter: Hyperparameter,
        stop_iteration: int = STOP_ITERATION,
        *,
        model_init_fn: Optional[ModelInitFunction] = None,
        without_checkpoint: bool = False,
    ) -> None:
        self.id = trial_id
        self.hyperparameter = hyperparameter
        self.stop_iteration = stop_iteration
        self.status = TrialStatus.PENDING
        self.worker_id = -1
        self.worker_type: Optional[WorkerType] = None
        self.run_time = 0
        self.iteration = 0
        self.phase = 0
        self.device_iteration_count = {WorkerType.CPU: 0, WorkerType.GPU: 0}
        self.checkpoint: Optional[Checkpoint] = None
        self.model_init_fn: Optional[
            Callable[[], Tuple[nn.Module, optim.Optimizer]]
        ] = None

        if not without_checkpoint:
            if model_init_fn is None:
                msg = (
                    f"TrialState(id={self.id}) requires a model_factory to create model"
                    "and optimizer unless `without_checkpoint=True`"
                )
                raise ValueError(msg)
            self.model_init_fn = lambda: model_init_fn(self.hyperparameter)
            model, optimizer = self.model_init_fn()
            self.checkpoint: Optional[Checkpoint] = Checkpoint({}, {})
            self.update_checkpoint(model, optimizer)

        self.accuracy = 0.0
        self.stop_accuracy = STOP_ACCURACY

    def update_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        if self.checkpoint is None:
            self.checkpoint = Checkpoint({}, {})

        self.checkpoint.model_state_dict = model.cpu().state_dict()
        # 取得 state_dict
        # model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()

        # # 如果是在 CUDA，轉到 CPU 儲存
        # model_state_dict = {k: v.cpu() for k, v in model_state_dict.items()}
        for state in optimizer_state_dict["state"].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        #
        # # 儲存回 checkpoint
        # self.checkpoint.model_state_dict = model_state_dict
        self.checkpoint.optimizer_state_dict = optimizer_state_dict

    def without_checkpoint(self) -> "TrialState":
        new_trial = TrialState(
            self.id,
            self.hyperparameter,
            self.stop_iteration,
            model_init_fn=None,
            without_checkpoint=True,
        )
        new_trial.accuracy = self.accuracy
        new_trial.status = self.status
        new_trial.worker_id = self.worker_id
        new_trial.worker_type = self.worker_type
        new_trial.run_time = self.run_time
        new_trial.iteration = self.iteration
        new_trial.phase = self.phase
        return new_trial


class TrialResult:
    def __init__(self, top_k: int = 3, bottom_k: int = 3) -> None:
        self.table: Dict[int, List[Tuple[float, Hyperparameter]]] = defaultdict(list)
        self.top_k: int = top_k
        self.bottom_k: int = bottom_k
        self.history_best: Tuple[
            float,
            Hyperparameter | None,
            Checkpoint | None,
        ] = (0.0, None, None)
        self.trial_progress: Dict[int, TrialState] = {}

    def get_trial_progress(self) -> List[TrialState]:
        return list(self.trial_progress.values())

    def recordtrial_progress(self, trial_state: TrialState) -> None:
        self.trial_progress[trial_state.id] = trial_state

    def update_trial_result(self, trial_state: TrialState) -> None:
        self.record_trial_progress(trial_state)
        self.table[trial_state.iteration].append(
            (trial_state.accuracy, trial_state.hyperparameter),
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
        if len(self.table[iteration]) < 5:
            return 0.0
        return sum([i[0] for i in self.table[iteration]]) / len(self.table[iteration])

    def get_top_k_result(
        self,
        iteration: int,
        k: int,
    ) -> List[Tuple[float, Hyperparameter]]:
        return self.table[iteration][:k]

    def get_history_best_result(
        self,
    ) -> Tuple[float, Optional[Hyperparameter], Optional[Checkpoint]]:
        return self.history_best

    def get_quantile(
        self,
        ratio: float = 0.25,
    ) -> Tuple[List[TrialState], List[TrialState]]:
        trials = [
            trial for trial in self.trial_progress.values() if trial.accuracy != 0
        ]
        if len(trials) < 2:
            return [], []

        trials.sort(key=lambda x: x.accuracy)
        quantile_size = math.ceil(len(trials) * ratio)

        if quantile_size > len(trials) / 2:
            quantile_size: int = len(trials) // 2

        return trials[:quantile_size], trials[-quantile_size:]

    def display_results(self, output_path: str = TRIAL_RESULT_OUTPUT_PATH) -> None:
        if not self.table:
            print("No results available")
            return
        try:
            with Path(output_path).open("w") as f:
                f.write(f"┏{'':━^13}┳{'':━^15}┳{'':━^35}┓\n")
                f.write(
                    f"┃{'Iteration':^13}┃{'Accuracy':^15}┃{'Hyperparameter':^35}┃\n",
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
                        output = (
                            f"lr:{hyper.lr:5.2f} momentum:{hyper.momentum:5.2f} "
                            f"batch:{hyper.batch_size:4}"
                        )

                        f.write(f"┃{'':^4}┃{'':^8}┃{accuracy:^15.6f}┃{output:^35}┃\n")

                    f.write(f"\033[s\033[{(len(top) + 1) // 2}A\033[8C")
                    f.write(f"{'top-' + str(self.top_k):^4}\033[u")

                    if self.bottom_k == 0 or len(bot) == 0:
                        f.write(f"\033[s\033[{(len(top) + len(bot) + 2) // 2}A\033[1C")
                        f.write(f"{iteration:^4}\033[u")
                        f.write(f"┣{'':━^4}╋{'':━^8}╋{'':━^15}╋{'':━^35}┫\n")
                        continue

                    f.write(f"┃{'':^4}┣{'':━^8}╋{'':━^15}╋{'':━^35}┫\n")
                    for data in sorted(bot, key=lambda x: x[0] * -1):
                        hyper = data[1]
                        accuracy = data[0]
                        output = (
                            f"lr:{hyper.lr:5.2f} momentum:{hyper.momentum:5.2f} "
                            f"batch:{hyper.batch_size:4}"
                        )

                        f.write(f"┃{'':^4}┃{'':^8}┃{accuracy:^15.6f}┃{output:^35}┃\n")
                    f.write(f"\033[s\033[{(len(bot) + 1) // 2}A\033[8C")
                    f.write(f"{'bot-' + str(self.bottom_k):^4}\033[u")
                    f.write(f"\033[s\033[{(len(top) + len(bot) + 2) // 2}A\033[1C")
                    f.write(f"{iteration:^4}\033[u")
                    f.write(f"┣{'':━^4}╋{'':━^8}╋{'':━^15}╋{'':━^35}┫\n")
                f.write(f"\033[A┗{'':━^4}┻{'':━^8}┻{'':━^15}┻{'':━^35}┛\n")

                f.write(
                    f"History Best: {self.history_best[0]} {self.history_best[1]}\n",
                )
        except Exception as e:
            print(f"{e}")

    def display_trial_progress(
        self,
        output_path: str = TRIAL_PROGRESS_OUTPUT_PATH,
    ) -> None:
        try:
            with Path(output_path).open("w") as f:
                f.write(
                    f"┏{'':━^4}┳{'':━^11}┳{'':━^11}┳{'':━^37}┳{'':━^3}┳{'':━^7}┳{'':━^7}┓\n",
                )
                f.write(
                    f"┃{'':^4}┃{'':^11}┃{'Worker':^11}┃{'Hyparameter':^37}┃{'':^3}┃{'':^7}┃{'':^7}┃\n",
                )
                f.write(
                    f"┃{'ID':^4}┃{'Status':^11}┣{'':━^4}┳{'':━^6}╋{'':━^7}┳{'':━^10}┳{'':━^6}┳{'':━^11}┫{'Ph':^3}┃{'Iter':^7}┃{'Acc':^7}┃\n",
                )
                f.write(
                    f"┃{'':^4}┃{'':^11}┃{'ID':^4}┃{'TYPE':^6}┃{'lr':^7}┃{'momentum':^10}┃{'bs':^6}┃{'model':^11}┃{'':^3}┃{'':^7}┃{'':^7}┃\n",
                )
                f.write(
                    f"┣{'':━^4}╋{'':━^11}╋{'':━^4}╋{'':━^6}╋{'':━^7}╋{'':━^10}╋{'':━^6}╋{'':━^11}╋{'':━^3}╋{'':━^7}╋{'':━^7}┫\n",
                )

                for i in self.trial_progress.values():
                    worker_type = "None"
                    if i.worker_type == WorkerType.CPU:
                        worker_type = "CPU"
                    elif i.worker_type == WorkerType.GPU:
                        worker_type = "GPU"
                    h = i.hyperparameter
                    worker_id = ""
                    if i.worker_id != -1:
                        worker_id = i.worker_id
                    f.write(
                        f"┃{i.id:>4}┃{i.status:^11}┃{worker_id:>4}┃{worker_type:^6}┃{h.lr:>7.3f}┃{h.momentum:>10.3f}┃{h.batch_size:>6}┃{h.model_type:^11}┃{i.phase:>3}┃{i.iteration:>7}┃{i.accuracy:>7.3f}┃\n",
                    )
                f.write(
                    f"┗{'':━^4}┻{'':━^11}┻{'':━^4}┻{'':━^6}┻{'':━^7}┻{'':━^10}┻{'':━^6}┻{'':━^11}┻{'':━^3}┻{'':━^7}┻{'':━^7}┛\n",
                )

        except Exception as e:
            print(f"{e}")
