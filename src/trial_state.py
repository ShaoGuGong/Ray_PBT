import copy
import heapq
import math
from pathlib import Path
from typing import TYPE_CHECKING

import ray
import torch
from torch import device, nn, optim

from .config import (
    MUTATION_ITERATION,
    STOP_ACCURACY,
    STOP_ITERATION,
    TRIAL_PROGRESS_OUTPUT_PATH,
)
from .utils import (
    Checkpoint,
    CheckpointLocation,
    Hyperparameter,
    ModelInitFunction,
    TrialStatus,
    WorkerState,
    WorkerType,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class TrialState:
    def __init__(
        self,
        trial_id: int,
        hyperparameter: Hyperparameter,
        model_init_fn: ModelInitFunction,
        stop_iteration: int = STOP_ITERATION,
    ) -> None:
        self.id: int = trial_id
        self.hyperparameter: Hyperparameter = hyperparameter
        self.stop_iteration: int = stop_iteration
        self.status: TrialStatus = TrialStatus.PENDING
        self.worker_id: int = -1
        self.worker_type: WorkerType | None = None
        self.run_time: float = 0
        self.iteration: int = 0
        self.device_iteration_count = {WorkerType.CPU: 0, WorkerType.GPU: 0}
        self.checkpoint: Checkpoint = Checkpoint.empty()
        self.accuracy: float = 0.0
        self.stop_accuracy: int = STOP_ACCURACY
        self.chunk_size: int = 1
        self.last_checkpoint_location: CheckpointLocation = CheckpointLocation.empty()
        self.model_init_fn: Callable[
            [device],
            tuple[nn.Module, optim.Optimizer],
        ] = (
            lambda device: model_init_fn(
                self.hyperparameter,
                self.checkpoint,
                device,
            )  # type: ignore[return-value]
        )

    def update_worker_state(self, worker_state: WorkerState) -> None:
        self.worker_type = worker_state.worker_type
        self.worker_id = worker_state.id
        self.last_checkpoint_location = CheckpointLocation(
            worker_state.id,
            ray.get_runtime_context().current_actor,
        )

    def update_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        self.checkpoint.model_state_dict = model.cpu().state_dict()
        optimizer_state_dict = optimizer.state_dict()

        for state in optimizer_state_dict["state"].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        self.checkpoint.optimizer_state_dict = optimizer_state_dict

    def get_remote_checkpoint(self) -> Checkpoint:
        return ray.get(
            self.last_checkpoint_location.worker_reference.get_saved_checkpoint.remote(  # type:ignore[reportGeneralTypeIssues]
                self.id,
            ),
        )

    def pop_remote_checkpoint(self) -> None:
        self.last_checkpoint_location.worker_reference.pop_saved_checkpoint.remote(  # type:ignore[reportGeneralTypeIssues]
            self.id,
        )

    @property
    def snapshot(self) -> "TrialState":
        new_trial = copy.copy(self)
        new_trial.checkpoint = Checkpoint.empty()
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

        self._mutation_baseline: float = 0.0
        self._upper_quantile_trials: list[TrialState] = []

    def add_trial(self, trial_state: TrialState) -> None:
        self.all_trials[trial_state.id] = trial_state

    def run_trial(self, trial_id: int) -> None:
        trial = self.all_trials.get(trial_id, None)
        if trial is None:
            msg = f"Trial id {trial_id} not found"
            raise ValueError(msg)

        self.pending.discard(trial_id)
        self.running.add(trial_id)

    def pend_trial(self, trial_id: int) -> None:
        trial = self.all_trials.get(trial_id, None)
        if trial is None:
            msg = f"Trial id {trial_id} not found"
            raise ValueError(msg)

        self.running.discard(trial_id)
        self.pending.add(trial_id)

    def complete_trial(self, trial_id: int) -> None:
        trial = self.all_trials.get(trial_id, None)
        if trial is None:
            msg = f"Trial id {trial_id} not found"
            raise ValueError(msg)

        self.running.discard(trial_id)
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

    def get_kth_largest_iteration_trial(self, k: int) -> TrialState | None:
        result = heapq.nlargest(
            k,
            [trial for trial in self.all_trials.values() if trial.id in self.pending],
            key=lambda t: t.iteration,
        )

        if not result:
            return None
        return result[-1]

    def get_mutation_baseline(
        self,
        ratio: float = 0.25,
    ) -> float:
        accuracy = [
            trial.accuracy for trial in self.all_trials.values() if trial.accuracy > 0
        ]
        quantile_size = math.ceil(len(self.all_trials) * ratio)

        result = heapq.nsmallest(
            quantile_size,
            accuracy,
        )

        if len(result) < quantile_size:
            return 0.0

        return result[-1]

    def get_cached_mutation_baseline(self) -> float:
        return self._mutation_baseline

    def get_cached_upper_quantile_trials(self) -> list[TrialState]:
        return self._upper_quantile_trials

    def get_uncompleted_trial_num(self) -> int:
        return len(self.all_trials) - len(self.completed)

    def get_upper_quantile_trials(self, ratio: float = 0.25) -> list[TrialState]:
        trials = [trial for trial in self.all_trials.values() if trial.accuracy > 0]
        quantile_size = math.ceil(len(self.all_trials) * ratio)
        return heapq.nlargest(
            quantile_size,
            trials,
            key=lambda t: t.accuracy,
        )

    def maybe_update_mutation_baseline(self) -> None:
        self._mutation_baseline = self.get_mutation_baseline()
        self._upper_quantile_trials = self.get_upper_quantile_trials()

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
                    f"┏{'':━^4}┳{'':━^11}┳{'':━^11}┳{'':━^37}┳{'':━^7}┳{'':━^7}┓\n"
                    f"┃{'':^4}┃{'':^11}┃{'Worker':^11}┃{'Hyparameter':^37}┃{'':^7}┃{'':^7}┃\n"
                    f"┃{'ID':^4}┃{'Status':^11}┣{'':━^4}┳{'':━^6}╋{'':━^7}┳{'':━^10}┳{'':━^6}┳{'':━^11}┫{'Iter':^7}┃{'Acc':^7}┃\n"
                    f"┃{'':^4}┃{'':^11}┃{'ID':^4}┃{'TYPE':^6}┃{'lr':^7}┃{'momentum':^10}┃{'bs':^6}┃{'model':^11}┃{'':^7}┃{'':^7}┃\n"
                    f"┣{'':━^4}╋{'':━^11}╋{'':━^4}╋{'':━^6}╋{'':━^7}╋{'':━^10}╋{'':━^6}╋{'':━^11}╋{'':━^7}╋{'':━^7}┫\n",
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
                        f"┃{i.id:>4}┃{i.status:^11}┃{worker_id:>4}┃{worker_type:^6}┃{h.lr:>7.3f}┃{h.momentum:>10.3f}┃{h.batch_size:>6}┃{h.model_type:^11}┃{i.iteration:>7}┃{i.accuracy:>7.3f}┃\n",
                    )
                f.write(
                    f"┗{'':━^4}┻{'':━^11}┻{'':━^4}┻{'':━^6}┻{'':━^7}┻{'':━^10}┻{'':━^6}┻{'':━^11}┻{'':━^7}┻{'':━^7}┛\n",
                )

        except Exception as e:  # noqa: BLE001
            print(f"{e}")  # noqa: T201
