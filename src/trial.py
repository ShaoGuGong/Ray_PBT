import asyncio
import logging
import os
import time
from datetime import datetime
from typing import List

import ray
from ray import ObjectRef
from ray.actor import ActorHandle

from .trial_state import TrialState
from .utils import TrialStatus


def get_trial_scheduler_logger() -> logging.Logger:
    """
    設置並返回一個日誌記錄器，用於跟踪訓練過程中的 TrialScheduler 記錄。

    日誌將記錄到一個帶有時間戳的目錄中，並包括在終端顯示和日誌文件中的訊息。

    Returns:
        logging.Logger: 配置好的 TrialScheduler 記錄器。
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(os.getcwd(), "logs/", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f"trial_scheduler")

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # 或者選擇更合適的級別

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s TRIAL_SCHEDULER -- %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)  # 只顯示 INFO 級別以上的訊息
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"trial_scheduler.log")
        )
        file_handler.setLevel(logging.DEBUG)  # 記錄所有級別的日誌
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TrialScheduler:
    """
    試驗調度器，負責管理和分配訓練試驗給可用的工作者。

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
        workers: List[ActorHandle],
        trial_states: List[TrialState],
    ) -> None:
        """
        初始化 TrialScheduler，設置試驗狀態和工作者。

        Args:
            train_step (TrainStepFunction): 訓練步驟函數。
            trial_states (List[TrialState]): 初始的試驗狀態列表。
        """
        self.tuner = tuner
        self.pending_trial_states = trial_states
        self.running_futures: List[ObjectRef] = []
        self.completed_trial_state = []
        self.logger = get_trial_scheduler_logger()
        self.workers = workers
        self._previous_time = time.time()

    def assign_trial_to_worker(self) -> List[ObjectRef]:
        """
        將一個試驗分配給一個可用的工作者。

        如果所有工作者都忙碌，則返回當前正在運行的訓練任務。

        Returns:
            List[ObjectRef]: 當前正在運行的訓練任務列表。
        """
        self.logger.info(
            f"⏳ 等待中訓練任務列表: {sorted([i.id for i in self.pending_trial_states])}"
        )

        available_futures = [
            worker.has_available_slots.remote() for worker in self.workers
        ]

        available_workers = [
            worker
            for worker, is_available in zip(self.workers, ray.get(available_futures))
            if is_available
        ]

        if not available_workers:
            if time.time() - self._previous_time > 10:
                self.logger.warning("沒有可用Worker")
                self._previous_time = time.time()
            return self.running_futures

        worker = next(iter(available_workers))

        if self.pending_trial_states:
            trial = self.pending_trial_states.pop(0)
            future = worker.assign_trial.remote(trial)
            self.running_futures.append(future)

        return self.running_futures

    def run(self):
        """
        開始訓練過程，將試驗分配給工作者並處理完成的結果。

        該方法會持續運行直到所有的試驗都完成。
        """
        self.logger.info("訓練開始")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while self.running_futures or self.pending_trial_states:
            self.assign_trial_to_worker()

            if not self.running_futures and not self.pending_trial_states:
                break

            done_futures, self.running_futures = ray.wait(
                self.running_futures, timeout=1.0
            )

            if done_futures:
                loop.run_until_complete(self.handle_done_futures(done_futures))
                # asyncio.create_task(self.handle_done_futures(done_futures))

        self.logger.info("🎉 所有 Trial 訓練完成！")

    def get_workers_logs(self) -> None:
        """
        獲取所有工作者的日誌並將其保存到文件中。

        該方法會將每個工作者的日誌寫入到相應的文件中。
        """

        log_dir = None

        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_dir = os.path.dirname(handler.baseFilename)  # 取得資料夾路徑
                break

        if log_dir is None:
            self.logger.error("logs檔案資料夾不存在")
            return

        for worker in self.workers:
            future = ray.get(worker.get_log_file.remote())
            with open(os.path.join(log_dir, f"worker-{future['id']}.log"), "w") as f:
                f.write(future["content"])

    async def handle_done_futures(self, done_futures: List[ObjectRef]):
        """
        處理已完成的訓練任務，將結果添加到已完成試驗狀態列表中。

        Args:
            done_futures (List[ObjectRef]): 已完成的訓練任務列表。
        """

        for future in done_futures:
            try:
                trial_state: TrialState = ray.get(future)
                if trial_state.status == TrialStatus.TERMINAL:
                    self.completed_trial_state.append(trial_state)
                    self.logger.info(
                        f"✅ Worker {trial_state.worker_id} 完成 Trial {trial_state.id} ，Accuracy: {trial_state.accuracy:.2f}"
                    )
                    self.logger.info(
                        f"✅ 已完成的訓練任務列表: {sorted([i.id for i in self.completed_trial_state])}"
                    )
                if trial_state.status == TrialStatus.PAUSE:
                    trial_state.status = TrialStatus.PENDING
                    self.pending_trial_states.append(trial_state)
                    self.logger.info(
                        f"🔃 Worker {trial_state.worker_id} 回傳未完成 Trial {trial_state.id}, Iteration: {trial_state.iteration} ，Accuracy: {trial_state.accuracy:.2f}"
                    )
                    trial_state = ray.get(self.tuner.mutation.remote(trial_state))
            except Exception as e:
                self.logger.error(f"❌ Future 執行失敗: {e}")
