import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn, optim

from .config import (
    MUTATION_ITERATION,
    STOP_ACCURACY,
    STOP_ITERATION,
    TRIAL_PROGRESS_OUTPUT_PATH,
)
from .utils import (
    Checkpoint,
    Hyperparameter,
    ModelInitFunction,
    TrialStatus,
    WorkerType,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class TrialState:
    def __init__(
        self,
        trial_id: int,
        hyperparameter: Hyperparameter,
        stop_iteration: int = STOP_ITERATION,
        *,
        model_init_fn: ModelInitFunction | None = None,
        without_checkpoint: bool = False,
    ) -> None:
        self.id: int = trial_id
        self.hyperparameter: Hyperparameter = hyperparameter
        self.stop_iteration: int = stop_iteration
        self.status: TrialStatus = TrialStatus.PENDING
        self.worker_id: int = -1
        self.worker_type: WorkerType | None = None
        self.run_time: float = 0
        self.iteration: int = 0
        self.phase: int = 0
        self.device_iteration_count = {WorkerType.CPU: 0, WorkerType.GPU: 0}
        self.checkpoint: Checkpoint | None = None
        self.accuracy: float = 0.0
        self.stop_accuracy: int = STOP_ACCURACY
        self.chunk_size: int = 1

        if not without_checkpoint:
            if model_init_fn is None:
                msg = (
                    f"TrialState(id={self.id}) requires a model_factory to create model"
                    "and optimizer unless `without_checkpoint=True`"
                )
                raise ValueError(msg)

            self.model_init_fn: Callable[[], tuple[nn.Module, optim.Optimizer]] = (
                lambda: model_init_fn(self.hyperparameter, self.checkpoint)  # type: ignore[return-value]
            )
            model, optimizer = self.model_init_fn()
            self.checkpoint = Checkpoint({}, {})
            self.update_checkpoint(model, optimizer)

    def update_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        if self.checkpoint is None:
            self.checkpoint = Checkpoint({}, {})

        self.checkpoint.model_state_dict = model.cpu().state_dict()
        optimizer_state_dict = optimizer.state_dict()

        for state in optimizer_state_dict["state"].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
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

    def set_chunk_size(self, chunk_size: int) -> None:
        if chunk_size < 1:
            msg = "Chunk size must be at least 1"
            raise ValueError(msg)
        self.chunk_size = chunk_size

    def set_terminated(self) -> None:
        self.status = TrialStatus.TERMINATE

    def set_pause(self) -> None:
        self.status = TrialStatus.PAUSE

    def set_interrupted(self) -> None:
        self.status = TrialStatus.INTERRUPTED

    def set_running(self) -> None:
        self.status = TrialStatus.RUNNING

    def set_pending(self) -> None:
        self.status = TrialStatus.PENDING

    def set_need_mutation(self) -> None:
        self.status = TrialStatus.NEED_MUTATION


class TrialResult:
    def __init__(self) -> None:
        self.history_best: tuple[
            float,
            Hyperparameter | None,
            Checkpoint | None,
        ] = (0.0, None, None)
        self.trial_progress: dict[int, TrialState] = {}

    def get_trial_progress(self) -> list[TrialState]:
        return list(self.trial_progress.values())

    def record_trial_progress(self, trial_state: TrialState) -> None:
        self.trial_progress[trial_state.id] = trial_state

    def update_trial_result(self, trial_state: TrialState) -> None:
        self.record_trial_progress(trial_state)
        if trial_state.accuracy > self.history_best[0]:
            self.history_best = (
                trial_state.accuracy,
                trial_state.hyperparameter,
                trial_state.checkpoint,
            )

    def get_chunk_size(self, iteration: int) -> int:
        iterations = sorted(
            [trial.iteration for trial in self.trial_progress.values()],
            reverse=True,
        )
        length = (len(iterations) // 4) + 1
        chunk_size = sum(iterations[:length]) // length - iteration
        chunk_size = (iteration + MUTATION_ITERATION) // MUTATION_ITERATION
        return max(chunk_size, 3)

    def get_history_best_result(
        self,
    ) -> tuple[float, Hyperparameter | None, Checkpoint | None]:
        return self.history_best

    def get_quantile(
        self,
        ratio: float = 0.25,
    ) -> tuple[list[TrialState], list[TrialState]]:
        trials = [
            trial for trial in self.trial_progress.values() if trial.accuracy != 0
        ]
        min_trials = 2
        if len(trials) < min_trials:
            return [], []

        trials.sort(key=lambda x: x.accuracy)
        quantile_size = math.ceil(len(trials) * ratio)

        if quantile_size > len(trials) / 2:
            quantile_size: int = len(trials) // 2

        return trials[:quantile_size], trials[-quantile_size:]

    def display_trial_result(
        self,
        output_path: Path = TRIAL_PROGRESS_OUTPUT_PATH,
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
