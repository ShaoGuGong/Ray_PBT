import heapq
import logging
import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import ray

from .config import (
    MUTATION_ITERATION,
    TRIAL_PROGRESS_OUTPUT_PATH,
)
from .trial_state import TrialState
from .utils import TrialStatus, WorkerType, colored_progress_bar


def get_trial_manager_logger() -> logging.Logger:
    timestamp = (datetime.now(UTC) + timedelta(hours=8)).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path.cwd() / "logs" / timestamp
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("TrialManager")

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # 或者選擇更合適的級別

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s TRIAL_MANAGER -- %(message)s",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)  # 只顯示 INFO 級別以上的訊息
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(Path(log_dir) / "trial_manager.log")
        file_handler.setLevel(logging.DEBUG)  # 記錄所有級別的日誌
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


@ray.remote
class TrialManager:
    def __init__(self, trial_states: list[TrialState]) -> None:
        self.all_trials = {trial.id: trial for trial in trial_states}
        self.pending_ids = {trial.id for trial in trial_states}
        self.running_ids = set()
        self.completed_ids = set()
        self.waiting_ids = set()
        self.history_best: TrialState | None = None

        self._mutation_baseline: float = 0.0
        self._upper_quantile_trials: list[TrialState] = []
        self.logger = get_trial_manager_logger()

    def add_trial(self, trial_state: TrialState) -> None:
        self.all_trials[trial_state.id] = trial_state

    def transition_to_waiting(self, trial_id: int) -> None:
        trial = self.all_trials.get(trial_id, None)
        if trial is None:
            msg = f"Trial id {trial_id} not found"
            raise ValueError(msg)

        self.pending_ids.discard(trial_id)
        self.waiting_ids.add(trial_id)

    def transition_to_running(self, trial_id: int) -> None:
        trial = self.all_trials.get(trial_id, None)
        if trial is None:
            msg = f"Trial id {trial_id} not found"
            raise ValueError(msg)

        self.waiting_ids.discard(trial_id)
        self.running_ids.add(trial_id)

    def transition_to_pending(self, trial_id: int) -> None:
        trial = self.all_trials.get(trial_id, None)
        if trial is None:
            msg = f"Trial id {trial_id} not found"
            raise ValueError(msg)

        self.running_ids.discard(trial_id)
        self.pending_ids.add(trial_id)

    def transition_to_completed(self, trial_id: int) -> None:
        trial = self.all_trials.get(trial_id, None)
        if trial is None:
            msg = f"Trial id {trial_id} not found"
            raise ValueError(msg)

        self.running_ids.discard(trial_id)
        self.completed_ids.add(trial_id)

    def get_pending_trials(self) -> list[TrialState]:
        return [self.all_trials[tid] for tid in self.pending_ids]

    def get_pending_trials_with_min_iteration(self) -> list[TrialState]:
        if not self.pending_ids:
            return []

        pending_trials = self.get_pending_trials()
        min_iter = min(pending_trials, key=lambda t: t.iteration).iteration
        return [trial for trial in pending_trials if trial.iteration == min_iter]

    def get_least_iterated_pending_trial(self) -> TrialState | None:
        if not self.pending_ids:
            return None

        return min(
            (self.all_trials[tid] for tid in self.pending_ids),
            key=lambda t: t.iteration,
            default=None,
        )

    def get_most_iterated_pending_trial(self) -> TrialState | None:
        if not self.pending_ids:
            return None

        return max(
            (self.all_trials[tid] for tid in self.pending_ids),
            key=lambda t: t.iteration,
            default=None,
        )

    def get_chunk_size(self, iteration: int) -> int:
        iterations = sorted(
            [trial.iteration for trial in self.all_trials.values()],
            reverse=True,
        )
        length = (len(iterations) // 4) + 1
        chunk_size = sum(iterations[:length]) // length - iteration
        chunk_size = (chunk_size + MUTATION_ITERATION) // MUTATION_ITERATION
        return max(chunk_size, 1)

    def get_history_best_result(self) -> TrialState | None:
        return self.history_best

    def get_nlargest_iteration_trials(self, k: int) -> list[TrialState]:
        return heapq.nlargest(
            k,
            [
                trial
                for trial in self.all_trials.values()
                if trial.id in self.pending_ids
            ],
            key=lambda t: t.iteration,
        )

    def get_mutation_baseline(
        self,
        ratio: float = 0.25,
    ) -> float:
        accuracy = [
            trial.accuracy for trial in self.all_trials.values() if trial.accuracy > 0
        ]
        quantile_size = math.ceil(len(self.all_trials) * ratio)

        result = heapq.nsmallest(
            quantile_size,
            accuracy,
        )

        if len(result) < quantile_size:
            return 0.0

        return result[-1]

    def get_cached_mutation_baseline(self) -> float:
        return self._mutation_baseline

    def get_cached_upper_quantile_trials(self) -> list[TrialState]:
        return self._upper_quantile_trials

    def get_uncompleted_trial_num(self) -> int:
        return len(self.all_trials) - len(self.completed_ids)

    def get_upper_quantile_trials(self, ratio: float = 0.25) -> list[TrialState]:
        trials = [trial for trial in self.all_trials.values() if trial.accuracy > 0]
        quantile_size = math.ceil(len(self.all_trials) * ratio)
        return heapq.nlargest(
            quantile_size,
            trials,
            key=lambda t: t.accuracy,
        )

    def has_pending_trials(self) -> bool:
        return bool(self.pending_ids)

    def maybe_update_mutation_baseline(self) -> None:
        self._mutation_baseline = self.get_mutation_baseline()
        self._upper_quantile_trials = self.get_upper_quantile_trials()

    def update_trial(self, trial_state: TrialState) -> None:
        if trial_state.id not in self.all_trials:
            msg = f"Trial id {trial_state.id} not found"
            raise ValueError(msg)

        if (
            trial_state.status == TrialStatus.RUNNING
            and trial_state.id in self.waiting_ids
        ):
            self.transition_to_running(trial_state.id)

        self.all_trials[trial_state.id] = trial_state

        if trial_state.accuracy > 0 and (
            self.history_best is None
            or trial_state.accuracy > self.history_best.accuracy
        ):
            self.history_best = trial_state

        if self.history_best:
            self.logger.info(
                "History best accuracy: %.2f, %s, iteration: %d",
                self.history_best.accuracy,
                str(self.history_best.hyperparameter),
                self.history_best.iteration,
            )
        self.maybe_update_mutation_baseline()
        self.display_trial_result()

    def is_finish(self) -> bool:
        return len(self.completed_ids) >= len(self.all_trials)

    def get_log_file(self) -> str:
        """
        取得 worker 對應的日誌檔案內容。

        Returns:
            dict: 包含 worker ID 與對應日誌內容的字典。
        """
        log_dir = None
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_dir = handler.baseFilename
                break

        if not log_dir:
            self.logger.error("Logs direction is not exists")
            return ""

        with Path(log_dir).open("r") as f:
            return f.read()

    def mutation(self, trial_state: TrialState) -> TrialState:
        self.logger.info(
            "Trial %d: 執行mutation, 原始超參數: %s",
            trial_state.id,
            trial_state.hyperparameter,
        )

        upper_quantile = self.get_cached_upper_quantile_trials()

        chose_trial = random.choice(upper_quantile)
        hyperparameter = chose_trial.hyperparameter.explore()

        trial_state.hyperparameter = hyperparameter
        trial_state.checkpoint = chose_trial.checkpoint

        self.logger.info(
            "Trial-%d Iter-%d, 結束mutation, 新超參數: %s",
            trial_state.id,
            trial_state.iteration,
            trial_state.hyperparameter,
        )

        return trial_state

    def display_trial_result(
        self,
        output_path: Path = TRIAL_PROGRESS_OUTPUT_PATH,
    ) -> None:
        try:
            with Path(output_path).open("w") as f:
                f.write(
                    f"┏{'':━^4}┳{'':━^11}┳{'':━^6}┳{'':━^11}┳{'':━^37}┳{'':━^7}┳{'':━^7}┓\n"
                    f"┃{'':^4}┃{'':^11}┃{'':^6}┃{'Worker':^11}┃{'Hyparameter':^37}┃{'':^7}┃{'':^7}┃\n"
                    f"┃{'ID':^4}┃{'Status':^11}┃{'SaveAt':^6}┣{'':━^4}┳{'':━^6}╋{'':━^7}┳{'':━^10}┳{'':━^6}┳{'':━^11}┫{'Iter':^7}┃{'Acc':^7}┃\n"
                    f"┃{'':^4}┃{'':^11}┃{'':^6}┃{'ID':^4}┃{'TYPE':^6}┃{'lr':^7}┃{'momentum':^10}┃{'bs':^6}┃{'model':^11}┃{'':^7}┃{'':^7}┃\n"
                    f"┣{'':━^4}╋{'':━^11}╋{'':━^6}╋{'':━^4}╋{'':━^6}╋{'':━^7}╋{'':━^10}╋{'':━^6}╋{'':━^11}╋{'':━^7}╋{'':━^7}┫\n",
                )

                for i in self.all_trials.values():
                    match i.worker_type:
                        case WorkerType.CPU:
                            worker_type = "CPU"
                        case WorkerType.GPU:
                            worker_type = "GPU"
                        case _:
                            worker_type = ""

                    h = i.hyperparameter
                    worker_id = ""
                    if i.worker_id != -1:
                        worker_id = i.worker_id
                    if i.last_checkpoint_location.is_empty():
                        save_at = ""
                    else:
                        save_at = i.last_checkpoint_location.worker_id

                    f.write(
                        f"┃{i.id:>4}┃{i.status:^11}┃{save_at:>6}┃{worker_id:>4}┃{worker_type:^6}┃{h.lr:>7.3f}┃{h.momentum:>10.3f}┃{h.batch_size:>6}┃{h.model_type:^11}┃{i.iteration:>7}┃{i.accuracy:>7.3f}┃\n",
                    )
                timestamp = (datetime.now(UTC) + timedelta(hours=8)).strftime(
                    "%Y-%m-%d %H:%M:%S",
                )

                f.write(
                    f"┗{'':━^4}┻{'':━^11}┻{'':━^6}┻{'':━^4}┻{'':━^6}┻{'':━^7}┻{'':━^10}┻{'':━^6}┻{'':━^11}┻{'':━^7}┻{'':━^7}┛\n"
                    f"{timestamp}\n",
                )

        except Exception as e:  # noqa: BLE001
            print(f"{e}")  # noqa: T201

    def print_iteration_count(self) -> None:
        iteration_counts = [
            (i.id, i.device_iteration_count) for i in self.all_trials.values()
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
