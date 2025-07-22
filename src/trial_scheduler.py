import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import ray
from ray import ObjectRef
from ray.actor import ActorHandle

from src.worker_manager import WorkerManager

from .trial_manager import TrialManager
from .utils import WorkerType, colored_progress_bar


def gpu_scheduling(
    trial_manager: TrialManager,
    worker_manager: WorkerManager,
    logger: logging.Logger,
) -> None:
    # 若沒有任何 pending 的 Trial,結束
    if not trial_manager.has_pending_trials():
        logger.info("沒有待分配的 Trial")
        return

    available_worker_entries = worker_manager.get_avaiable_gpu_workers()

    # 選擇擁有最多 slot 的 GPU worker 作為目標 worker
    selected_worker_entry = max(
        available_worker_entries,
        key=lambda e: e.available_slots,
    )

    # 取得 iteration 次數最少的一組 pending trials
    trial_states = trial_manager.get_pending_trials_with_min_iteration()

    # 優先挑選之前曾經在該 worker 上有 checkpoint 的 trial
    selected_trial = next(
        (
            t
            for t in trial_states
            if not t.last_checkpoint_location.is_empty()
            and t.last_checkpoint_location.worker_id == selected_worker_entry.state.id
        ),
        trial_states[0],  # 若無符合者, 則選第一個
    )

    # 設定 chunk_size、worker_id、worker_type 並標記為執行狀態
    selected_trial.set_chunk_size(
        trial_manager.get_chunk_size(selected_trial.iteration),
    )
    selected_trial.worker_id = selected_worker_entry.state.id
    selected_trial.worker_type = WorkerType.GPU
    selected_trial.set_running()

    trial_manager.transition_to_running(selected_trial.id)
    trial_manager.update_trial(selected_trial)

    worker_manager.assign_trial_to_worker(
        selected_worker_entry.state.id,
        selected_trial,
    )


def cpu_scheduling(
    trial_manager: TrialManager,
    worker_manager: WorkerManager,
    logger: logging.Logger,
) -> None:
    # 若沒有任何 pending 的 Trial,直接結束
    if not trial_manager.has_pending_trials():
        logger.info("沒有待分配的 Trial")
        return

    available_worker_entries = worker_manager.get_avaiable_cpu_workers()

    if not available_worker_entries:
        return

    # 選擇第一個可用的 worker
    selected_worker_entry = available_worker_entries[0]

    # 取得 iteration 較高的 Trials, 個數為 CPU 數
    trial_states = trial_manager.get_nlargest_iteration_trials(
        len(worker_manager.cpu_workers) * 2,
    )
    trial_states = trial_states[len(trial_states) // 2 :]

    # 優先選擇 checkpoint 來自該 worker 的 trial, 否則選最後一個(iteration 最小)
    selected_trial = next(
        (
            t
            for t in trial_states
            if not t.last_checkpoint_location.is_empty()
            and t.last_checkpoint_location.worker_id == selected_worker_entry.state.id
        ),
        trial_states[-1],
    )

    # 設定 chunk_size(暫定為 2), 標記執行資訊
    selected_trial.set_chunk_size(2)
    selected_trial.worker_id = selected_worker_entry.state.id
    selected_trial.worker_type = WorkerType.CPU
    selected_trial.set_running()

    # 更新 Trial 狀態至 running
    trial_manager.transition_to_running(selected_trial.id)
    trial_manager.update_trial(selected_trial)

    worker_manager.assign_trial_to_worker(
        selected_worker_entry.state.id,
        selected_trial,
    )


def gpu_stealing_strategy(
    cpu_workers: list[ActorHandle],
    logger: logging.Logger,
) -> ObjectRef | None:
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
    timestamp = (datetime.now(UTC) + timedelta(hours=8)).strftime("%Y-%m-%d_%H-%M-%S")
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
        worker_manager: WorkerManager,
        trial_manager: TrialManager,
    ) -> None:
        """
        初始化 TrialScheduler, 設置試驗狀態和工作者。

        Args:
            train_step (TrainStepFunction): 訓練步驟函數。
            trial_states (List[TrialState]): 初始的試驗狀態列表。
        """
        self.trial_manager = trial_manager
        self.worker_manager = worker_manager

        self.running_futures: list[ObjectRef] = []
        self.logger: logging.Logger = get_trial_scheduler_logger()

        self._previous_time: float = time.time()
        self.logger.info("初始化完成")

    def assign_trial_to_worker(self) -> None:  # type: ignore[reportGeneralTypeIssues]
        """
        將一個試驗分配給一個可用的工作者。

        如果所有工作者都忙碌, 則返回當前正在運行的訓練任務。

        Returns:
            List[ObjectRef]: 當前正在運行的訓練任務列表。
        """
        time.sleep(1)

    def run(self) -> None:
        """
        開始訓練過程, 將試驗分配給工作者並處理完成的結果。

        該方法會持續運行直到所有的試驗都完成。
        """
        self.logger.info("訓練開始")
        update_assign_time = time.time()
        # while self.running_futures or self.pending_trial_states:
        while not self.trial_manager.is_finish():
            if (current_time := time.time()) - update_assign_time > 1.0:
                update_assign_time = current_time
                self.assign_trial_to_worker()

        self.print_iteration_count()
        self.logger.info("🎉 所有 Trial 訓練完成!")
        self.worker_manager.stop_all_workers()

    def print_iteration_count(self) -> None:
        iteration_counts = [
            (i.id, i.device_iteration_count)
            for i in self.trial_manager.all_trials.values()
        ]

        iteration_counts.sort(key=lambda x: x[0])

        for index, value in iteration_counts:
            self.logger.info(
                "Trial:%2d CPU/GPU %s",
                index,
                colored_progress_bar(
                    [value[WorkerType.CPU], value[WorkerType.GPU]],
                    40,
                ),
            )
        self.logger.info(
            "Total    CPU/GPU %s",
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

        for worker_entry in self.worker_manager.workers.values():
            worker = worker_entry.ref

            future = ray.get(worker.get_log_file.remote())  # type: ignore[reportGeneralTypeIssues]
            with (Path(log_dir) / f"worker-{future['id']}.log").open("w") as f:
                f.write(future["content"])
