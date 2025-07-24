import logging
import os
import time
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import ray

from .trial_manager import TrialManager
from .trial_scheduler import TrialScheduler
from .trial_state import TrialState
from .utils import (
    DataloaderFactory,
    TrainStepFunction,
    TrialStatus,
    get_head_node_address,
)
from .worker_manager import WorkerManager

if TYPE_CHECKING:
    from ray.actor import ActorHandle


def get_tuner_logger() -> logging.Logger:
    timestamp = (datetime.now(UTC) + timedelta(hours=8)).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path.cwd() / "logs" / timestamp
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("Tuner")

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # 或者選擇更合適的級別

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s TUNER -- %(message)s",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)  # 只顯示 INFO 級別以上的訊息
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(Path(log_dir) / "tuner.log")
        file_handler.setLevel(logging.DEBUG)  # 記錄所有級別的日誌
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# NOTE:
# model 的建立的時間,
# batch_size 對於 throughput 計算
@ray.remote
class Tuner:
    def __init__(
        self,
        trial_states: list[TrialState],
        train_step: TrainStepFunction,
        dataloader_factory: DataloaderFactory,
    ) -> None:
        self.logger = get_tuner_logger()
        self.logger.info("總共 %d 個 Trial", len(trial_states))

        self.trial_manager: ActorHandle = TrialManager.options(
            max_concurrency=10,
            num_cpus=1,
            resources={f"node:{get_head_node_address()}": 0.01},
        ).remote(trial_states)  # type: ignore[reportGeneralTypeIssues]

        self.worker_manager = WorkerManager(
            ray.get_runtime_context().current_actor,
            self.trial_manager,
            train_step=train_step,
            dataloader_factory=dataloader_factory,
        )

        self.scheduler: TrialScheduler = TrialScheduler(
            self.worker_manager,
            self.trial_manager,
        )

    def run(self) -> None:
        start = time.time()
        self.logger.info("開始訓練")
        self.scheduler.run()
        self.logger.info("結束訓練")
        end = time.time()
        self.logger.info("訓練總時長: %.2f 秒", end - start)
        self.scheduler.get_workers_logs()
        self.logger.info("Assign: %d", self.worker_manager.assign_count["assign"])
        self.logger.info("Locality: %d", self.worker_manager.assign_count["locality"])

    def submit_trial(self, worker_id: int, trial_state: TrialState) -> None:
        status = trial_state.status

        match status:
            case TrialStatus.PAUSE:
                trial_state.set_pending()
                self.logger.info(
                    "🔃 Worker %2d 回傳未完成 Trial %2d, Iteration: %d, Accuracy: %.2f",
                    trial_state.worker_id,
                    trial_state.id,
                    trial_state.iteration,
                    trial_state.accuracy,
                )

                trial_state.worker_id = -1
                trial_state.worker_type = None

                self.trial_manager.transition_to_pending.remote(trial_state.id)
                self.trial_manager.update_trial.remote(trial_state)

            case TrialStatus.NEED_MUTATION:
                trial_state.set_pending()
                trial_state = ray.get(self.trial_manager.mutation.remote(trial_state))  # type: ignore[reportGeneralTypeIssues]
                if trial_state.checkpoint.is_empty():
                    self.logger.warning("Trial %d checkpoint is None", trial_state.id)

                trial_state.worker_id = -1
                trial_state.worker_type = None

                self.trial_manager.transition_to_pending.remote(trial_state.id)
                self.trial_manager.update_trial.remote(trial_state)

            case TrialStatus.TERMINATED:
                trial_state.set_terminated()
                self.logger.info(
                    "✅ Worker %2d Trial %2d 完成, Accuracy: %.2f",
                    trial_state.worker_id,
                    trial_state.id,
                    trial_state.accuracy,
                )
                trial_state.worker_id = -1
                trial_state.worker_type = None

                self.trial_manager.transition_to_completed.remote(trial_state.id)
                self.trial_manager.update_trial.remote(trial_state)

            case TrialStatus.FAILED:
                self.trial_manager.transition_to_completed.remote(trial_state.id)
                trial_state.worker_id = -1
                trial_state.worker_type = None

                self.logger.warning(
                    "Worker %2d Trial %2d  發生錯誤, 已中止訓練",
                    trial_state.worker_id,
                    trial_state.id,
                )

        self.worker_manager.release_slots(worker_id, trial_state.id)

    def get_zipped_log(self) -> bytes:
        log_dir = None

        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_dir = Path(handler.baseFilename).parent  # 取得資料夾路徑
                break

        if log_dir is None:
            msg = "log_dir not found."
            raise FileNotFoundError(msg)

        trial_manager_log_content = ray.get(
            self.trial_manager.get_log_file.remote(),  # type: ignore[reportGeneralTypeIssues]
        )
        with (log_dir / "trial_manager.log").open("w") as f:
            f.write(trial_manager_log_content)

        worker_manager_log_content = self.worker_manager.get_log_file()

        with (log_dir / "worker_manager.log").open("w") as f:
            f.write(worker_manager_log_content)

        zip_path = "./logs.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(log_dir):
                for file in files:
                    abs_file = Path(root) / file
                    rel_path = os.path.relpath(abs_file, log_dir)
                    zf.write(abs_file, arcname=rel_path)

        with Path(zip_path).open("rb") as f:
            return f.read()
