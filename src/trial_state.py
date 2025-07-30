import copy
from collections.abc import Callable
from dataclasses import dataclass, field

import ray
import torch
from torch import device, nn, optim

from .config import (
    MAX_GENERATION,
    STOP_ACCURACY,
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


@dataclass(slots=True)
class TrialState:
    id: int
    hyperparameter: Hyperparameter
    _raw_model_init_fn: ModelInitFunction
    max_generation: int = MAX_GENERATION
    status: TrialStatus = TrialStatus.PENDING
    worker_id: int = -1
    worker_type: WorkerType | None = None
    run_time: float = 0
    generation: int = 0
    device_iteration_count: dict[WorkerType, int] = field(
        default_factory=lambda: {WorkerType.CPU: 0, WorkerType.GPU: 0},
    )
    checkpoint: Checkpoint = field(default_factory=Checkpoint.empty)
    last_checkpoint_location: CheckpointLocation = field(
        default_factory=CheckpointLocation.empty,
    )
    accuracy: float = 0.0
    stop_accuracy: float = STOP_ACCURACY
    chunk_size: int = 1
    model_init_fn: Callable[
        [device],
        tuple[nn.Module, optim.Optimizer],
    ] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.model_init_fn = (
            lambda device: self._raw_model_init_fn(
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
        self.last_checkpoint_location.worker_reference.remove_checkpoint.remote(  # type:ignore[reportGeneralTypeIssues]
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
        self.status = TrialStatus.TERMINATED

    def set_pause(self) -> None:
        self.status = TrialStatus.PAUSE

    def set_running(self) -> None:
        self.status = TrialStatus.RUNNING

    def set_pending(self) -> None:
        self.status = TrialStatus.PENDING

    def set_need_mutation(self) -> None:
        self.status = TrialStatus.NEED_MUTATION

    def set_waiting(self) -> None:
        self.status = TrialStatus.WAITING
