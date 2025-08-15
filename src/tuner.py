import logging
import os
import zipfile
from pathlib import Path

import ray

from .trial_scheduler import TrialScheduler
from .trial_state import TrialResult, TrialState
from .utils import (
    DataloaderFactory,
    TrainStepFunction,
    WorkerType,
    get_tuner_logger,
)
from .worker import generate_all_workers


class Tuner:
    __slots__ = (
        "logger",
        "scheduler",
        "trial_result",
        "trial_states",
        "workers",
    )

    def __init__(
        self,
        trial_states: list[TrialState],
        train_step: TrainStepFunction,
        dataloader_factory: DataloaderFactory,
    ) -> None:
        self.trial_states = trial_states
        self.logger = get_tuner_logger()

        self.logger.info("總共 %d 個 Trial", len(self.trial_states))
        self.logger.info("\n".join([str(t.hyperparameter) for t in self.trial_states]))

        self.workers = generate_all_workers(
            ray.get_runtime_context().current_actor,
            train_step=train_step,
            dataloader_factory=dataloader_factory,
        )

        self.scheduler: TrialScheduler = TrialScheduler(
            ray.get_runtime_context().current_actor,
            self.workers,
            self.trial_states,
        )
        self.trial_result: TrialResult = TrialResult()

        for trial in self.trial_states:
            self.trial_result.record_trial_progress(trial)

    def run(self) -> None:
        self.logger.info("開始訓練")
        self.scheduler.run()
        self.logger.info("結束訓練")
        self.scheduler.get_workers_logs()

    def update_trial_result(self, trial_state: TrialState) -> None:
        self.trial_result.update_trial_result(trial_state)
        history_best = self.trial_result.get_history_best_result()
        self.logger.info(
            "History Best: %.6f %s",
            history_best[0],
            history_best[1],
        )

    def record_trial_progress(self, trial_state: TrialState) -> None:
        self.trial_result.record_trial_progress(trial_state)
        self.trial_result.display_trial_progress()

    def get_min_iteration_trial(self) -> tuple[int, int]:
        cpu_trial = [
            trial
            for trial in self.trial_result.trial_progress.values()
            if trial.worker_type == WorkerType.CPU
        ]
        target = min(cpu_trial, key=lambda x: x.iteration)
        return target.worker_id, target.id

    def get_trial_progress(self) -> list[TrialState]:
        return self.trial_result.get_trial_progress()

    def submit_trial(self, trial_state: TrialState) -> None:
        self.scheduler.submit_trial(trial_state)

    def get_zipped_log(self) -> bytes:
        log_dir = None

        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_dir = Path(handler.baseFilename).parent  # 取得資料夾路徑
                break
        if log_dir is None:
            msg = "log_dir not found."
            raise FileNotFoundError(msg)

        zip_path = "./logs.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(log_dir):
                for file in files:
                    abs_file = Path(root) / file
                    rel_path = os.path.relpath(abs_file, log_dir)
                    zf.write(abs_file, arcname=rel_path)

        with Path(zip_path).open("rb") as f:
            return f.read()

    #                  ╭───────────────────────────────────────────╮
    #                  │ Subclass Should Implement Flowing Methods │
    #                  ╰───────────────────────────────────────────╯
    def should_mutate_trial(self, trial_state: TrialState) -> bool:
        error_msg = "Not Implemented should_mutate_trial method"
        raise NotImplementedError(error_msg)

    def mutate_trial(self, trial_state: TrialState) -> TrialState:
        error_msg = "Not Implemented mutate_trial method"
        raise NotImplementedError(error_msg)
