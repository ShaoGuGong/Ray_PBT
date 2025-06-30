import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import ray
from ray import ObjectRef
from ray.actor import ActorHandle

from .config import PHASE_ITERATION, STOP_ITERATION
from .trial_phase import TrialPhase
from .trial_state import TrialState
from .utils import TrialStatus, WorkerType, colored_progress_bar


def cpu_scheduling(
    trial_state: TrialState,
    cpu_workers: list[ActorHandle],
) -> bool:
    available_futures: list = [
        worker.get_available_slots.remote() for worker in cpu_workers
    ]

    available_cpu_workers = [
        worker
        for worker, available_slots in zip(
            cpu_workers,
            ray.get(available_futures),
            strict=True,
        )  # type: ignore[reportGeneralTypeIssues]
        if available_slots
    ]

    if not available_cpu_workers:
        return False

    worker = next(iter(available_cpu_workers))
    ray.get(worker.assign_trial.remote(trial_state))  # type: ignore[reportGeneralTypeIssues]

    return True


def gpu_scheduling(
    trial_state: TrialState,
    gpu_workers: list[ActorHandle],
) -> bool:
    available_futures = [worker.get_available_slots.remote() for worker in gpu_workers]

    available_gpu_workers = [
        (worker, available_slots)
        for worker, available_slots in zip(
            gpu_workers,
            ray.get(available_futures),  # type: ignore[reportGeneralTypeIssues]
            strict=True,
        )
        if available_slots
    ]

    if not available_gpu_workers:
        return False

    worker = max(available_gpu_workers, key=lambda x: x[1])[0]
    ray.get(worker.assign_trial.remote(trial_state))  # type: ignore[reportGeneralTypeIssues]
    return True


def gpu_stealing_strategy(
    cpu_workers: list[ActorHandle],
    **kargs: Any,
) -> ObjectRef | None:
    logger = kargs["logger"]

    available_futures: list = [
        worker.get_active_trials.remote() for worker in cpu_workers
    ]

    running_cpu_workers = [
        (worker, min(activate_trials, key=lambda x: x.iteration))
        for worker, activate_trials in zip(
            cpu_workers,
            ray.get(available_futures),  # type:int[reportGeneralTypeIssues]
            strict=True,
        )
        if len(activate_trials) > 0
    ]

    if running_cpu_workers:
        worker, trial_state = min(running_cpu_workers, key=lambda x: x[1].iteration)
        logger.info("對 Trial %d 執行搶奪", trial_state.id)
        ray.wait([worker.send_signal.remote(trial_state.id)], timeout=0.1)  # type: ignore[reportGeneralTypeIssues]


def get_trial_scheduler_logger() -> logging.Logger:
    """
    設置並返回一個日誌記錄器, 用於跟踪訓練過程中的 TrialScheduler 記錄。

    日誌將記錄到一個帶有時間戳的目錄中, 並包括在終端顯示和日誌文件中的訊息。

    Returns:
        logging.Logger: 配置好的 TrialScheduler 記錄器。
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path.cwd() / "logs" / timestamp
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("trial_scheduler")

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # 或者選擇更合適的級別

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s TRIAL_SCHEDULER -- %(message)s",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)  # 只顯示 INFO 級別以上的訊息
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(Path(log_dir) / "trial_scheduler.log")
        file_handler.setLevel(logging.DEBUG)  # 記錄所有級別的日誌
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TrialScheduler:
    """
    試驗調度器, 負責管理和分配訓練試驗給可用的工作者。

    Attributes:
        trial_states (List[TrialState]): 當前待分配的試驗狀態列表。
        running_futures (List[ObjectRef]): 當前正在運行的訓練任務。
        completed_trial_state (List[TrialState]): 完成的試驗狀態列表。
        logger (logging.Logger): 記錄訓練過程的日誌記錄器。
        train_result (TrainResult): 用於記錄每個訓練結果的實例。
        workers (List[ActorHandle]): 可用的工作者列表。
    """

    def __init__(
        self,
        tuner: ActorHandle,
        workers: list[ActorHandle],
        trial_states: list[TrialState],
    ) -> None:
        """
        初始化 TrialScheduler, 設置試驗狀態和工作者。

        Args:
            train_step (TrainStepFunction): 訓練步驟函數。
            trial_states (List[TrialState]): 初始的試驗狀態列表。
        """
        self.tuner = tuner
        self.trial_phase = TrialPhase(STOP_ITERATION, PHASE_ITERATION)

        self.pending_trial_states: list[TrialState] = trial_states
        self.completed_trial_states: list[TrialState] = []
        self.waiting_trial_states: list[TrialState] = []
        self.trial_state_nums: int = len(self.pending_trial_states)

        self.running_futures: list[ObjectRef] = []
        self.logger: logging.Logger = get_trial_scheduler_logger()
        self.workers: list[ActorHandle] = workers
        self._previous_time: float = time.time()
        self.is_final_phase: bool = False

        [worker.run.remote() for worker in self.workers]

        self.gpu_workers = [
            worker
            for worker in self.workers
            if ray.get(worker.get_worker_type.remote()) == WorkerType.GPU  # type: ignore[reportGeneralTypeIssues]
        ]
        self.cpu_workers = [
            worker
            for worker in self.workers
            if ray.get(worker.get_worker_type.remote()) == WorkerType.CPU  # type:ignore[reportGeneralTypeIssues]
        ]

        self.logger.info("初始化完成")

    def assign_trial_to_worker(self) -> None:  # type: ignore[reportGeneralTypeIssues]
        """
        將一個試驗分配給一個可用的工作者。

        如果所有工作者都忙碌, 則返回當前正在運行的訓練任務。

        Returns:
            List[ObjectRef]: 當前正在運行的訓練任務列表。
        """

        if not self.pending_trial_states:
            return

        trial_state = max(self.pending_trial_states, key=lambda x: x.iteration)
        trial_state.set_chunk_size(3)
        if cpu_scheduling(trial_state, self.cpu_workers):
            self.pending_trial_states.remove(trial_state)

        trial_state = min(self.pending_trial_states, key=lambda x: x.iteration)
        chunk_size = ray.get(
            self.tuner.get_chunk_size.remote(trial_state.iteration),  # type: ignore[reportGeneralTypeIssues]
        )
        trial_state.set_chunk_size(chunk_size)
        if gpu_scheduling(trial_state, self.gpu_workers):
            self.pending_trial_states.remove(trial_state)

    def submit_trial(self, trial_state: TrialState) -> None:
        status = trial_state.status

        match status:
            case TrialStatus.INTERRUPTED:
                trial_state.set_pending()
                self.pending_trial_states.append(trial_state)
                self.logger.info(
                    "🔃 Worker %d 回傳已中斷 Trial %d",
                    trial_state.worker_id,
                    trial_state.id,
                )

            case TrialStatus.PAUSE:
                trial_state.set_pending()
                self.tuner.update_trial_result.remote(trial_state)  # type: ignore[reportGeneralTypeIssues]
                self.pending_trial_states.append(trial_state)
                self.logger.info(
                    "🔃 Worker %d 回傳未完成 Trial %d, Iteration: %d, Accuracy: %.2f",
                    trial_state.worker_id,
                    trial_state.id,
                    trial_state.iteration,
                    trial_state.accuracy,
                )

            case TrialStatus.NEED_MUTATION:
                trial_state = ray.get(self.tuner.mutation.remote(trial_state))  # type: ignore[reportGeneralTypeIssues]
                if trial_state.checkpoint is None:
                    self.logger.debug("Trial %d checkpoint is None", trial_state.id)
                trial_state.status = TrialStatus.PENDING
                self.pending_trial_states.append(trial_state)
                self.tuner.update_trial_result.remote(trial_state)  # type: ignore[reportGeneralTypeIssues]

            case TrialStatus.TERMINATE:
                self.completed_trial_states.append(trial_state)
                self.tuner.update_trial_result.remote(trial_state)  # type: ignore[reportGeneralTypeIssues]
                self.logger.info(
                    "✅ Worker %d Trial %d 完成, Accuracy: %.2f",
                    trial_state.worker_id,
                    trial_state.id,
                    trial_state.accuracy,
                )
                self.logger.info(
                    "✅ 已完成的訓練任務列表: %s",
                    str(sorted([i.id for i in self.completed_trial_states])),
                )

            case TrialStatus.FAILED:
                self.completed_trial_states.append(trial_state)
                self.logger.warning(
                    "Worker %d Trial %d  發生錯誤, 已中止訓練",
                    trial_state.worker_id,
                    trial_state.id,
                )

        trial_state.worker_id = -1
        trial_state.worker_type = None
        self.tuner.update_trial_result.remote(trial_state)

    def run(self) -> None:
        """
        開始訓練過程, 將試驗分配給工作者並處理完成的結果。

        該方法會持續運行直到所有的試驗都完成。
        """
        self.logger.info("訓練開始")
        update_phase_time = time.time()
        update_assign_time = time.time()
        # while self.running_futures or self.pending_trial_states:
        while len(self.completed_trial_states) < self.trial_state_nums:
            if current_time := time.time() - update_assign_time > 1.0:
                update_assign_time = current_time
                self.assign_trial_to_worker()

            seconds = 60.0

            if current_time := time.time() - update_phase_time > seconds:
                self.update_phase()
                update_phase_time = current_time

        self.print_iteration_count()
        self.logger.info("🎉 所有 Trial 訓練完成!")
        futures = [worker.stop.remote() for worker in self.workers]
        ray.get(futures)  # type:ignore[reportGeneralTypeIssues]

    def print_iteration_count(self) -> None:
        iteration_counts = [
            (i.id, i.device_iteration_count) for i in self.completed_trial_states
        ]

        iteration_counts.sort(key=lambda x: x[0])

        for index, value in iteration_counts:
            print(
                f"Trial:{index:2} CPU/GPU",
                colored_progress_bar(
                    [value[WorkerType.CPU], value[WorkerType.GPU]],
                    40,
                ),
            )

        print(
            "Total    CPU/GPU",
            colored_progress_bar(
                [
                    sum(i[1][WorkerType.CPU] for i in iteration_counts),
                    sum(i[1][WorkerType.GPU] for i in iteration_counts),
                ],
                40,
            ),
        )

    def get_workers_logs(self) -> None:
        """
        獲取所有工作者的日誌並將其保存到文件中。
        該方法會將每個工作者的日誌寫入到相應的文件中。
        """
        log_dir = None
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_dir = Path(handler.baseFilename).parent  # 取得資料夾路徑
                break

        if log_dir is None:
            self.logger.error("logs檔案資料夾不存在")
            return

        for worker in self.workers:
            future = ray.get(worker.get_log_file.remote())  # type: ignore[reportGeneralTypeIssues]
            with (Path(log_dir) / f"worker-{future['id']}.log").open("w") as f:
                f.write(future["content"])

    def update_phase(self) -> None:
        old = self.trial_phase.current_phase
        self.trial_phase.update_phase(ray.get(self.tuner.get_trial_progress.remote()))  # type: ignore[reportGeneralTypeIssues]

        if old != self.trial_phase.current_phase:
            self.logger.info("更新階段到Phase %d", self.trial_phase.current_phase)
            futures = [
                worker.update_phase.remote(self.trial_phase.current_phase)
                for worker in self.workers
            ]
            ray.wait(futures, timeout=0.1)  # type: ignore[reportGeneralTypeIssues]
