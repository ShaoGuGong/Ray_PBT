import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List

import ray
import torch.optim as optim
from ray import ObjectRef
from ray.actor import ActorHandle

from utils import ModelType, TrialStatus, get_model


def get_trial_scheduler_logger() -> logging.Logger:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(os.getcwd(), "logs/", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f"trial_scheduler")

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # 或者選擇更合適的級別

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s TRIAL_SCHEDULER -- %(message)s"
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


@dataclass
class Hyperparameter:
    lr: float
    momentum: float
    batch_size: int
    model_type: ModelType


@dataclass
class Checkpoint:
    model_state_dict: dict
    optimzer_state_dict: dict
    checkpoint_interval: int


class TrialState:
    def __init__(
        self,
        id: int,
        hyperparameter: Hyperparameter,
        stop_iteration: int = 10,
    ) -> None:
        self.id = id
        self.hyperparameter = hyperparameter
        self.stop_iteration = stop_iteration
        self.status = TrialStatus.PENDING
        self.worker_id = -1
        self.run_time = 0
        self.iteration = 0

        model = get_model(self.hyperparameter.model_type)
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.hyperparameter.lr,
            momentum=self.hyperparameter.momentum,
        )

        self.checkpoint = Checkpoint(
            model.state_dict(),
            optimizer.state_dict(),
            0,
        )
        self.accuracy = 0.0
        self.stop_accuracy = 0.5

    def update_checkpoint(self, model, optimizer, iteration: int) -> None:
        self.checkpoint.model_state_dict = model.state_dict()
        self.checkpoint.optimzer_state_dict = optimizer.state_dict()
        self.iteration = iteration

    def __str__(self) -> str:
        result = f"Trial: {str(self.hyperparameter)}\n {self.status=}\n {self.stop_iteration=}"
        return result


class TrialScheduler:
    def __init__(
        self, trial_states: List[TrialState], workers: List[ActorHandle]
    ) -> None:
        self.workers = workers
        self.trial_states = trial_states
        self.running_futures: List[ObjectRef] = []
        self.completed_trial_state = []
        self.logger = get_trial_scheduler_logger()

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

        # for worker in available_workers:
        #     if not self.trial_states:
        #         break
        #
        #     trial = self.trial_states.pop(0)
        #     future = worker.assign_trial.remote(trial)
        #     self.running_futures.append(future)

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
