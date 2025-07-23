import logging
from dataclasses import dataclass, field
from itertools import count

import ray
from ray.actor import ActorHandle

from .trial_state import TrialState
from .utils import (
    DataloaderFactory,
    TrainStepFunction,
    WorkerState,
    WorkerType,
    get_head_node_address,
)
from .worker import Worker


@dataclass
class WorkerEntry:
    state: WorkerState
    ref: ActorHandle
    active_trials: list[int] = field(default_factory=list)

    @property
    def available_slots(self) -> int:
        return self.state.max_trials - len(self.active_trials)

    @property
    def id(self) -> int:
        return self.state.id


class WorkerManager:
    def __init__(
        self,
        tuner: ActorHandle,
        train_step: TrainStepFunction,
        dataloader_factory: DataloaderFactory,
    ) -> None:
        self.workers: dict[int, WorkerEntry] = {}
        self.assign_count: dict[str, int] = {"assign": 0, "locality": 0}

        worker_states = generate_all_worker_states()

        for worker_state in worker_states:
            worker_ref: ActorHandle = Worker.options(  # type: ignore[reportGeneralTypeIssues]
                max_concurrency=worker_state.max_trials + 3,
                name=f"worker-{worker_state.id}",
                num_cpus=worker_state.num_cpus,
                num_gpus=worker_state.num_gpus,
                resources={worker_state.node_name: 0.01},
            ).remote(
                worker_state,
                train_step,
                tuner,
                dataloader_factory,
            )

            self.workers[worker_state.id] = WorkerEntry(worker_state, worker_ref)

        self.cpu_workers = {
            worker_id: worker_entry
            for worker_id, worker_entry in self.workers.items()
            if worker_entry.state.worker_type == WorkerType.CPU
        }

        self.gpu_workers = {
            worker_id: worker_entry
            for worker_id, worker_entry in self.workers.items()
            if worker_entry.state.worker_type == WorkerType.GPU
        }
        [self.workers[worker_id].ref.run.remote() for worker_id in self.workers]

    def get_avaiable_cpu_workers(self) -> list[WorkerEntry]:
        return [
            worker_entry
            for worker_entry in self.cpu_workers.values()
            if worker_entry.available_slots > 0
        ]

    def get_avaiable_gpu_workers(self) -> list[WorkerEntry]:
        return [
            worker_entry
            for worker_entry in self.gpu_workers.values()
            if worker_entry.available_slots > 0
        ]

    def assign_trial_to_worker(
        self,
        worker_id: int,
        trial: TrialState,
        logger: logging.Logger | None = None,
    ) -> None:
        entry = self.workers.get(worker_id, None)
        if entry is None:
            msg = f"Worker {worker_id} 不存在."
            raise ValueError(msg)

        entry.active_trials.append(trial.id)
        self.assign_count["assign"] += 1

        if trial.last_checkpoint_location.worker_id == worker_id:
            if logger:
                logger.info(
                    "分配 Trial %d snapshot 到 Worker %d",
                    trial.id,
                    worker_id,
                )
            self.assign_count["locality"] += 1
            entry.ref.assign_trial.remote(trial.snapshot)  # type: ignore[reportGeneralTypeIssues]
        else:
            if logger:
                logger.info(
                    "分配 Trial %d 到 Worker %d",
                    trial.id,
                    worker_id,
                )
            entry.ref.assign_trial.remote(trial)  # type: ignore[reportGeneralTypeIssues]

    def release_slots(self, worker_id: int, trial_id: int) -> None:
        entry = self.workers.get(worker_id, None)
        if entry is None:
            msg = f"Worker {worker_id} 不存在."
            raise ValueError(msg)

        entry.active_trials.remove(trial_id)

    def stop_all_workers(self) -> None:
        ray.get(
            [worker_entry.ref.stop.remote() for worker_entry in self.workers.values()],  # type:ignore[reportGeneralTypeIssues]
        )


def generate_all_worker_states() -> list[WorkerState]:
    """
    根據 Ray 叢集的節點資源建立所有 Worker。


    Returns:
        List[ActorHandle]: 建立的 Worker Actor 清單。
    """

    visited_address = set()
    worker_states: list[WorkerState] = []
    count_gen = count(start=0, step=1)
    head_node_address = get_head_node_address()

    for node in ray.nodes():
        node_address = node["NodeManagerAddress"]
        if node["Alive"]:
            if node_address in visited_address:
                continue

            resource = node["Resources"]

            gpus = resource.get("GPU", 0)
            cpus = resource.get("CPU", 1)

            if "GPU" in resource:
                cpus -= 1
                worker_states.append(
                    WorkerState(
                        id=next(count_gen),
                        num_cpus=1,
                        num_gpus=gpus,
                        node_name=f"node:{node_address}",
                        max_trials=3,
                        worker_type=WorkerType.GPU,
                    ),
                )

            if "CPU" in resource:
                if node_address == head_node_address:
                    cpus -= 2

                worker_states.append(
                    WorkerState(
                        id=next(count_gen),
                        num_cpus=cpus,
                        num_gpus=0,
                        node_name=f"node:{node_address}",
                        max_trials=1,
                        worker_type=WorkerType.CPU,
                    ),
                )
            visited_address.add(node_address)

    return worker_states
