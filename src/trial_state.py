import copy
from typing import TYPE_CHECKING

import ray
import torch
from torch import device, nn, optim

from .config import (
    STOP_ACCURACY,
    STOP_ITERATION,
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
        if self.last_checkpoint_location.is_empty():
            return Checkpoint.empty()

        return ray.get(
            self.last_checkpoint_location.worker_reference.get_checkpoint.remote(  # type:ignore[reportGeneralTypeIssues]
                self.id,
            ),
        )

    def pop_remote_checkpoint(self) -> None:
        if self.last_checkpoint_location.is_empty():
            return
        ray.get(
            self.last_checkpoint_location.worker_reference.pop_checkpoint.remote(  # type:ignore[reportGeneralTypeIssues]
                self.id,
            ),
        )

    def remove_remote_checkpoint(self) -> None:
        if self.last_checkpoint_location.is_empty():
            return
        self.last_checkpoint_location.worker_reference.pop_checkpoint.remote(  # type:ignore[reportGeneralTypeIssues]
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
