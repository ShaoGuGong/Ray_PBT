import heapq
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


class TrialManager:
    def __init__(self, trial_states: list[TrialState]) -> None:
        self.all_trials = {trial.id: trial for trial in trial_states}
        self.pending = {trial.id for trial in trial_states}
        self.running = set()
        self.completed = set()
        self.history_best: TrialState | None = None

    def add_trial(self, trial_state: TrialState) -> None:
        self.all_trials[trial_state.id] = trial_state

    def run_trial(self, trial_id: int) -> None:
        trial = self.all_trials.get(trial_id, None)
        if trial is None:
            msg = f"Trial id {trial_id} not found"
            raise ValueError(msg)

        self.pending.discard(trial_id)
        trial.set_running()
        self.running.add(trial_id)

    def pend_trial(self, trial_id: int) -> None:
        trial = self.all_trials.get(trial_id, None)
        if trial is None:
            msg = f"Trial id {trial_id} not found"
            raise ValueError(msg)

        self.running.discard(trial_id)
        trial.set_terminated()
        self.pending.add(trial_id)

    def complete_trial(self, trial_id: int) -> None:
        trial = self.all_trials.get(trial_id, None)
        if trial is None:
            msg = f"Trial id {trial_id} not found"
            raise ValueError(msg)

        self.running.discard(trial_id)
        trial.set_terminated()
        self.completed.add(trial_id)

    def get_least_iterated_pending_trial(self) -> TrialState | None:
        if not self.pending:
            return None

        return min(
            (self.all_trials[tid] for tid in self.pending),
            key=lambda t: t.iteration,
            default=None,
        )

    def get_most_iterated_pending_trial(self) -> TrialState | None:
        if not self.pending:
            return None

        return max(
            (self.all_trials[tid] for tid in self.pending),
            key=lambda t: t.iteration,
            default=None,
        )

    def get_chunk_size(self, iteration: int) -> int:
        iterations = sorted(
            [trial.iteration for trial in self.all_trials.values()],
            reverse=True,
        )
        length = (len(iterations) // 4) + 1
        chunk_size = sum(iterations[:length]) // length - iteration
        chunk_size = (chunk_size + MUTATION_ITERATION) // MUTATION_ITERATION
        return max(chunk_size, 1)

    def get_history_best_result(self) -> TrialState | None:
        return self.history_best

    def get_quantile(
        self,
        ratio: float = 0.25,
    ) -> tuple[list[TrialState], list[TrialState]]:
        trials = [trial for trial in self.all_trials.values() if trial.accuracy != 0]
        quantile_size = math.ceil(len(trials) * ratio)

        min_trials = 2
        if len(trials) < min_trials:
            return [], []

        quantile_size = min(math.ceil(len(trials) * ratio), len(trials) // 2)

        bottom_k = heapq.nsmallest(quantile_size, trials, key=lambda x: x.accuracy)
        top_k = heapq.nlargest(quantile_size, trials, key=lambda x: x.accuracy)

        return bottom_k, top_k

    def update_trial(self, trial_state: TrialState) -> None:
        if trial_state.id not in self.all_trials:
            msg = f"Trial id {trial_state.id} not found"
            raise ValueError(msg)

        self.all_trials[trial_state.id] = trial_state
        if trial_state.accuracy > 0 and (
            self.history_best is None
            or trial_state.accuracy > self.history_best.accuracy
        ):
            self.history_best = trial_state

        self.display_trial_result()

    def is_finish(self) -> bool:
        return len(self.completed) >= len(self.all_trials)

    def display_trial_result(
        self,
        output_path: Path = TRIAL_PROGRESS_OUTPUT_PATH,
    ) -> None:
        try:
            with Path(output_path).open("w") as f:
                f.write(
                    f"┏{'':━^4}┳{'':━^11}┳{'':━^11}┳{'':━^37}┳{'':━^3}┳{'':━^7}┳{'':━^7}┓\n"
                    f"┃{'':^4}┃{'':^11}┃{'Worker':^11}┃{'Hyparameter':^37}┃{'':^3}┃{'':^7}┃{'':^7}┃\n"
                    f"┃{'ID':^4}┃{'Status':^11}┣{'':━^4}┳{'':━^6}╋{'':━^7}┳{'':━^10}┳{'':━^6}┳{'':━^11}┫{'Ph':^3}┃{'Iter':^7}┃{'Acc':^7}┃\n"
                    f"┃{'':^4}┃{'':^11}┃{'ID':^4}┃{'TYPE':^6}┃{'lr':^7}┃{'momentum':^10}┃{'bs':^6}┃{'model':^11}┃{'':^3}┃{'':^7}┃{'':^7}┃\n"
                    f"┣{'':━^4}╋{'':━^11}╋{'':━^4}╋{'':━^6}╋{'':━^7}╋{'':━^10}╋{'':━^6}╋{'':━^11}╋{'':━^3}╋{'':━^7}╋{'':━^7}┫\n",
                )

                for i in self.all_trials.values():
                    match i.worker_type:
                        case WorkerType.CPU:
                            worker_type = "CPU"
                        case WorkerType.GPU:
                            worker_type = "GPU"
                        case _:
                            worker_type = ""

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
