import logging
import random
from itertools import count
from pathlib import Path

import ray
import torch
from ray.actor import ActorHandle
from torch import nn, optim
from torch.utils.data import DataLoader

from .config import (
    MUTATION_ITERATION,
)
from .trial_state import TrialState
from .utils import (
    Accuracy,
    DataloaderFactory,
    TrainStepFunction,
    TrialStatus,
    WorkerState,
    WorkerType,
    get_head_node_address,
)


class WorkerLoggerFormatter(logging.Formatter):
    """
    自訂的日誌格式器, 用於在日誌中加入 worker_id 和 trial_id。

    Attributes:
        None
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日誌紀錄, 加入 worker_id 和 trial_id。

        Args:
            record (logging.LogRecord): 日誌紀錄。

        Returns:
            str: 格式化後的日誌訊息。
        """

        worker_type = getattr(record, "worker_type", WorkerType.CPU)
        match worker_type:
            case WorkerType.GPU:
                record.worker_type = "GPU"
            case WorkerType.CPU:
                record.worker_type = "CPU"

        record.worker_id = getattr(record, "worker_id", "N/A")
        record.trial_id = getattr(record, "trial_id", "N/A")
        return super().format(record)


def get_worker_logger(worker_id: int) -> logging.Logger:
    """
    建立並回傳指定 worker_id 的 logger, 支援終端輸出與檔案寫入。

    Args:
        worker_id (int): Worker 的唯一識別碼。

    Returns:
        logging.Logger: 設定完成的 logger。
    """

    log_dir = Path(Path.cwd()) / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"worker-{worker_id}")

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        formatter = WorkerLoggerFormatter(
            "[%(asctime)s] %(levelname)s %(worker_type)s "
            "WORKER_ID: %(worker_id)s TRIAL_ID: %(trial_id)s -- %(message)s",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(Path(log_dir) / f"worker-{worker_id}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


@ray.remote
class Worker:
    """
    表示一個 worker 節點, 負責訓練與回報試驗結果。

    Attributes:
        worker_state (WorkerState): Worker 的狀態資訊。
        active_trials (dict): 活躍試驗的字典。
        train_step (TrainStepFunction): 執行訓練步驟的函式。
        device (torch.device): 使用的設備 (CPU 或 GPU) 。
        logger (logging.Logger): 負責日誌紀錄。
    """

    def __init__(
        self,
        worker_state: WorkerState,
        train_step: TrainStepFunction,
        tuner: ActorHandle,
        dataloader_factory: DataloaderFactory,
    ) -> None:
        """
        初始化 Worker, 設定狀態與參數。

        Args:
            worker_state (WorkerState): Worker 的狀態資訊。
            train_step (TrainStepFunction): 訓練步驟函式。
            tuner (ActorHandle): 負責接收訓練結果的 Actor。
        """
        self.worker_state: WorkerState = worker_state
        self.active_trials: dict[int, TrialState] = {}
        self.train_step: TrainStepFunction = train_step
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = get_worker_logger(worker_id=worker_state.id)
        self.tuner: ActorHandle = tuner
        self.mutation_iteration: int = MUTATION_ITERATION
        self.interrupt_set: set = set()
        self.dataloader_factory: DataloaderFactory = dataloader_factory
        self.is_stop: bool = False
        self.trial_iteration_count: dict[int, int] = {}
        self.log("info", "初始化完成")

        if self.worker_state.worker_type == WorkerType.CPU:
            torch.set_num_threads(int(self.worker_state.num_cpus))

    def get_active_trials_nums(self) -> int:
        """
        取得目前活躍試驗的數量。

        Returns:
            int: 活躍試驗數量。
        """
        return len(self.active_trials)

    def get_active_trials(self) -> list[TrialState]:
        return list(self.active_trials.values())

    def get_available_slots(self) -> int:
        """
        取得可供分配的新試驗插槽數。

        Returns:
            int: 可分配試驗的插槽數。
        """
        return self.worker_state.max_trials - len(self.active_trials)

    def send_signal(self, trial_id: int) -> None:
        if trial_id not in self.active_trials:
            self.log("info", f"TRIAL_ID: {trial_id}不存在, {self.active_trials.keys()}")
            return
        self.log("info", f"接收到訊號 trial: {trial_id}")
        self.interrupt_set.add(trial_id)

    def assign_trial(self, trial_state: TrialState) -> None:
        """
        將試驗分配給該 worker 並開始訓練。

        Args:
            trial_state (TrialState): 試驗狀態。

        Returns:
            TrialState: 更新後的試驗狀態。
        """
        if len(self.active_trials) >= self.worker_state.max_trials:
            return

        self.active_trials[trial_state.id] = trial_state
        self.trial_iteration_count[trial_state.id] = 0
        trial_state.worker_id = self.worker_state.id
        trial_state.worker_type = self.worker_state.worker_type
        self.log("info", f"Running Trials: {list(self.active_trials)}")

    def run(self) -> None:
        while not self.is_stop:
            trial_state = min(
                self.get_active_trials(),
                key=lambda x: x.iteration,
                default=None,
            )
            if trial_state is None:
                continue

            self.log(
                "info",
                f"開始訓練, chunk_size: {trial_state.chunk_size}",
                trial_id=trial_state.id,
            )
            self.log("debug", f"{torch.cuda.is_available()=}")

            hyper = trial_state.hyperparameter
            train_loader, test_loader, _ = self.dataloader_factory(
                hyper.batch_size,
                num_workers=int(self.worker_state.num_cpus),
            )
            model, optimizer = trial_state.model_init_fn(self.device)

            for _ in range(trial_state.chunk_size):
                self.train(trial_state, model, optimizer, train_loader, test_loader)

                match trial_state.status:
                    case (
                        TrialStatus.TERMINATE
                        | TrialStatus.PAUSE
                        | TrialStatus.NEED_MUTATION
                        | TrialStatus.INTERRUPTED
                    ):
                        trial_state.update_checkpoint(model, optimizer)
                        self.tuner.submit_trial.remote(trial_state)  # type: ignore[reportGeneralTypeIssues]
                        self.active_trials.pop(trial_state.id)
                        break

                    case TrialStatus.RUNNING:
                        self.tuner.update_trial.remote(trial_state)  # type: ignore[reportGeneralTypeIssues]

            if trial_state.status == TrialStatus.RUNNING:
                trial_state.set_pause()
                trial_state.update_checkpoint(model, optimizer)
                self.tuner.submit_trial.remote(trial_state)
                self.active_trials.pop(trial_state.id)

    def _check_and_handle_trial_condition(self, trial_state: TrialState) -> bool:
        if trial_state.id in self.interrupt_set:
            trial_state.set_interrupted()
            self.interrupt_set.remove(trial_state.id)
            return True

        return False

    def train(
        self,
        trial_state: TrialState,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> TrialState:
        """
        執行試驗的訓練流程。

        Args:
            trial_state (TrialState): 試驗狀態。

        Returns:
            TrialState: 訓練後的試驗狀態。
        """
        for _ in range(
            self.mutation_iteration - trial_state.iteration % self.mutation_iteration,
        ):
            if trial_state.iteration >= trial_state.stop_iteration:
                trial_state.set_terminated()
                break

            if self._check_and_handle_trial_condition(trial_state):
                break

            self.train_step(
                model,
                optimizer,
                train_loader,
                trial_state.hyperparameter.batch_size,
                self.device,
            )

            trial_state.iteration += 1
            trial_state.device_iteration_count[self.worker_state.worker_type] += 1

        if trial_state.status == TrialStatus.INTERRUPTED:
            return trial_state

        trial_state.accuracy = self.test(model, test_loader)

        self.log(
            "info",
            f"Iteration: {trial_state.iteration} Accuracy: {trial_state.accuracy}",
            trial_id=trial_state.id,
        )

        if (
            trial_state.iteration > trial_state.stop_iteration
            or trial_state.accuracy > trial_state.stop_accuracy
        ):
            trial_state.set_terminated()

        if trial_state.status != TrialStatus.RUNNING:
            return trial_state

        baseline = ray.get(self.tuner.get_mutation_baseline.remote())  # type: ignore[reportGeneralTypeIssues]
        self.log(
            "info",
            f"Baseline: {baseline}, Accuracy: {trial_state.accuracy}",
            trial_id=trial_state.id,
        )

        mutation_ratio = 0.25
        if trial_state.accuracy <= baseline and random.random() >= mutation_ratio:
            trial_state.set_need_mutation()

        return trial_state

    def test(self, model: nn.Module, test_loader: DataLoader) -> Accuracy:
        """
        使用測試資料對模型進行測試並回傳準確率。

        Args:
            model (torch.nn.Module): 已訓練的模型。
            test_loader (torch.utils.data.DataLoader): 測試資料載入器。

        Returns:
            Accuracy: 模型測試結果的準確率。
        """
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for raw_inputs, raw_targets in test_loader:
                inputs, targets = (
                    raw_inputs.to(self.device),
                    raw_targets.to(self.device),
                )
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return correct / total

    def get_log_file(self) -> dict[str, int | str]:
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
            self.log("error", "Logs direction is not exists")
            return {"id": self.worker_state.id, "content": ""}

        with Path(log_dir).open("r") as f:
            return {"id": self.worker_state.id, "content": f.read()}

    def log(self, level: str, message: str, trial_id: int | str = "N/A") -> None:
        """
        根據指定的 log 級別輸出訊息。

        Args:
            level (str): 記錄等級 (info/debug/warning/error/critical) 。
            message (str): 要記錄的訊息。
            trial_id (Union[int, str], optional): 試驗 ID。預設為 "N/A"。
        """
        extra = {
            "worker_type": self.worker_state.worker_type,
            "worker_id": self.worker_state.id,
            "trial_id": trial_id,
        }
        if level == "info":
            self.logger.info(message, extra=extra)
            return
        if level == "debug":
            self.logger.debug(message, extra=extra)
            return
        if level == "warning":
            self.logger.warning(message, extra=extra)
            return
        if level == "critical":
            self.logger.critical(message, extra=extra)
            return
        if level == "error":
            self.logger.error(message, extra=extra)
            return

    def get_id(self) -> int:
        return self.worker_state.id

    def get_worker_type(self) -> WorkerType:
        """
        回傳 worker 類型 (CPU/GPU) 。

        Returns:
            WorkerType: Worker 類型。
        """
        return self.worker_state.worker_type

    def stop(self) -> None:
        self.is_stop = True


def generate_all_workers(
    tuner: ActorHandle,
    train_step: TrainStepFunction,
    dataloader_factory: DataloaderFactory,
) -> list[ActorHandle]:
    """
    根據 Ray 叢集的節點資源建立所有 Worker。

    Args:
        tuner (ActorHandle): 接收試驗結果的 Actor。
        train_step (TrainStepFunction): 訓練步驟函式。

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
                    cpus -= 1

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

    workers: list[ActorHandle] = []
    print(*worker_states, sep="\n")

    for index, worker_state in enumerate(worker_states):
        workers.append(
            Worker.options(  # type: ignore[reportGeneralTypeIssues]
                max_concurrency=worker_state.max_trials + 3,
                name=f"worker-{index}",
                num_cpus=worker_state.num_cpus,
                num_gpus=worker_state.num_gpus,
                resources={worker_state.node_name: 0.01},
            ).remote(
                worker_state,
                train_step,
                tuner,
                dataloader_factory,
            ),
        )

    return workers
