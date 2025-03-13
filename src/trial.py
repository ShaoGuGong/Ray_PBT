import asyncio
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import List

import ray
from ray import ObjectRef

from trial_state import TrialState
from utils import Accuracy, TrainStepFunction
from worker import generate_all_workers


def get_trial_scheduler_logger() -> logging.Logger:
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


@ray.remote
class TrainResult:
    def __init__(self) -> None:
        self.accuracy_table: defaultdict = defaultdict(float)

    def record_accuracy(self, accuracy: Accuracy, iteration: int) -> None:
        self.accuracy_table[iteration] = max(self.accuracy_table[iteration], accuracy)

    def get_accuracy(self, iteration: int) -> Accuracy:
        return self.accuracy_table[iteration]


class TrialScheduler:
    def __init__(
        self,
        train_step: TrainStepFunction,
        trial_states: List[TrialState],
    ) -> None:
        self.trial_states = trial_states
        self.running_futures: List[ObjectRef] = []
        self.completed_trial_state = []
        self.logger = get_trial_scheduler_logger()
        self.train_result = TrainResult.options(
            max_concurrency=5,
            num_cpus=1,
            resources={
                f"node:{ray.get_runtime_context().gcs_address.split(':')[0]}": 0.01
            },
        ).remote()
        self.workers = generate_all_workers(
            train_result=self.train_result, train_step=train_step
        )

    def assign_trial_to_worker(self) -> List[ObjectRef]:
        available_futures = [
            worker.has_available_slots.remote() for worker in self.workers
        ]

        available_workers = [
            worker
            for worker, is_available in zip(self.workers, ray.get(available_futures))
            if is_available
        ]

        if not available_workers:
            self.logger.warning("沒有可用Worker")
            return self.running_futures

        worker = next(iter(available_workers))

        trial = self.trial_states.pop(0)
        future = worker.assign_trial.remote(trial)
        self.running_futures.append(future)

        return self.running_futures

    def run(self):
        self.logger.info("訓練開始")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while self.running_futures or self.trial_states:
            self.assign_trial_to_worker()

            if not self.running_futures and not self.trial_states:
                break

            done_futures, self.running_futures = ray.wait(
                self.running_futures, timeout=1.0
            )

            if done_futures:
                loop.run_until_complete(self.handle_done_futures(done_futures))
                # asyncio.create_task(self.handle_done_futures(done_futures))

        self.logger.info("🎉 所有 Trial 訓練完成！")

    def get_workers_logs(self) -> None:
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
        for future in done_futures:
            try:
                trial_state = ray.get(future)
                self.completed_trial_state.append(trial_state)
                self.logger.info(
                    f"✅ Worker {trial_state.worker_id} 完成 Trial {trial_state.id} ，Accuracy: {trial_state.accuracy:.2f}"
                )
            except Exception as e:
                self.logger.error(f"❌ Future 執行失敗: {e}")
