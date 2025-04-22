import logging
import math
import os
import random
from dataclasses import replace
from datetime import datetime
from typing import List, Tuple

import ray

from .trial_scheduler import TrialScheduler
from .trial_state import TrialResult, TrialState
from .utils import TrainStepFunction, WorkerType
from .worker import generate_all_workers


def get_tuner_logger() -> logging.Logger:

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(os.getcwd(), "logs/", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f"Tuner")

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # 或者選擇更合適的級別

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s TUNER -- %(message)s",
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


# TODO:
# 陳舊度最舊插隊機制
# NOTE:
# model 的建立的時間,
# batch_size 對於 throughput 計算
@ray.remote
class Tuner:
    def __init__(
        self,
        trial_states: List[TrialState],
        train_step: TrainStepFunction,
    ) -> None:
        self.trial_states = trial_states
        self.logger = get_tuner_logger()

        self.logger.info(f"總共{len(trial_states)} 個 Trial")
        self.logger.info("\n".join([str(t.hyperparameter) for t in trial_states]))

        self.workers = generate_all_workers(
            ray.get_runtime_context().current_actor, train_step=train_step
        )

        self.scheduler: TrialScheduler = TrialScheduler(
            ray.get_runtime_context().current_actor, self.workers, trial_states
        )
        self.trial_result: TrialResult = TrialResult()

        for trial in self.trial_states:
            self.trial_result.record_trial_progress(trial)

    def run(self) -> None:
        self.logger.info("開始訓練")
        self.scheduler.run()
        self.scheduler.get_workers_logs()
        self.logger.info(f"{len(self.trial_states)}")
        self.logger.info("結束訓練")

    def update_trial_result(self, trial_state: TrialState):
        self.trial_result.update_trial_result(trial_state)
        self.logger.info(
            f"History Best: {self.trial_result.history_best[0]} {self.trial_result.history_best[1]}\n"
        )

    def record_trial_progress(self, trial_state: TrialState):
        self.trial_result.record_trial_progress(trial_state)
        self.trial_result.display_trial_progress()

    def mutation(self, trial_state: TrialState) -> TrialState:
        self.logger.info(
            f"Trial {trial_state.id}: 執行mutation, 原始超參數: {trial_state.hyperparameter}"
        )

        _, hyperparameter, _ = self.trial_result.get_history_best_result()

        if hyperparameter:
            mutation_hyperparameter = (
                (
                    "lr",
                    hyperparameter.lr
                    * 0.5
                    * (0.0001 + 0.1)
                    * (
                        1
                        + math.cos(
                            math.pi * trial_state.iteration * trial_state.stop_iteration
                        )
                    ),
                ),
                ("momentum", random.uniform(0.001, 1)),
                ("batch_size", random.choice([64, 128, 256, 512, 1024])),
            )
            trial_state.hyperparameter = replace(
                hyperparameter,
                **{k: v for k, v in random.sample(mutation_hyperparameter, 1)},
            )

        self.logger.info(
            f"Trial {trial_state.id}: 結束mutation, 新超參數: {trial_state.hyperparameter}"
        )

        return trial_state

    def get_baseline(self, iteration):
        return self.trial_result.get_mean_accuray(iteration)

    def get_min_iteration_trial(self) -> Tuple[int, int]:
        cpu_trial = [
            trial
            for trial in self.trial_result.trial_progress.values()
            if trial.worker_type == WorkerType.CPU
        ]
        target = min(cpu_trial, key=lambda x: x.iteration)
        return target.worker_id, target.id
