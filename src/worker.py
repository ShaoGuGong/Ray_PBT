import logging
import os
from typing import Dict, List, Union

import ray
import torch
import torch.nn as nn
import torch.optim as optim
from ray import ObjectRef
from ray.actor import ActorHandle
from torch.utils.data import DataLoader

from .config import (
    GPU_MAX_ITERATION,
    MUTATION_ITERATION,
    PHASE_ITERATION,
    STOP_ITERATION,
)
from .trial_phase import TrialPhase
from .trial_state import TrialState
from .utils import (
    Accuracy,
    DataloaderFactory,
    TrainStepFunction,
    TrialStatus,
    WorkerState,
    WorkerType,
    get_head_node_address,
    get_model,
)


class WorkerLoggerFormatter(logging.Formatter):
    """
    自訂的日誌格式器，用於在日誌中加入 worker_id 和 trial_id。

    Attributes:
        None
    """

    def format(self, record):
        """
        格式化日誌紀錄，加入 worker_id 和 trial_id。

        Args:
            record (logging.LogRecord): 日誌紀錄。

        Returns:
            str: 格式化後的日誌訊息。
        """

        worker_type = getattr(record, "worker_type", WorkerType.CPU)
        if worker_type == WorkerType.GPU:
            record.worker_type = "GPU"
        elif worker_type == WorkerType.CPU:
            record.worker_type = "CPU"

        record.worker_id = getattr(record, "worker_id", "N/A")
        record.trial_id = getattr(record, "trial_id", "N/A")
        return super().format(record)


def get_worker_logger(worker_id: int) -> logging.Logger:
    """
    建立並回傳指定 worker_id 的 logger，支援終端輸出與檔案寫入。

    Args:
        worker_id (int): Worker 的唯一識別碼。

    Returns:
        logging.Logger: 設定完成的 logger。
    """

    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f"worker-{worker_id}")

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        formatter = WorkerLoggerFormatter(
            "[%(asctime)s] %(levelname)s %(worker_type)s WORKER_ID: %(worker_id)s TRIAL_ID: %(trial_id)s -- %(message)s",
            # datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"worker-{worker_id}.log")
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


@ray.remote
class Worker:
    """
    表示一個 worker 節點，負責訓練與回報試驗結果。

    Attributes:
        worker_state (WorkerState): Worker 的狀態資訊。
        active_trials (dict): 活躍試驗的字典。
        train_step (TrainStepFunction): 執行訓練步驟的函式。
        device (torch.device): 使用的設備（CPU 或 GPU）。
        logger (logging.Logger): 負責日誌紀錄。
    """

    def __init__(
        self,
        worker_state: WorkerState,
        train_step: TrainStepFunction,
        tuner: ActorHandle,
        trial_phase: TrialPhase,
        dataloader_factory: DataloaderFactory,
    ) -> None:
        """
        初始化 Worker，設定狀態與參數。

        Args:
            worker_state (WorkerState): Worker 的狀態資訊。
            train_step (TrainStepFunction): 訓練步驟函式。
            tuner (ActorHandle): 負責接收訓練結果的 Actor。
        """
        self.worker_state: WorkerState = worker_state
        self.active_trials: Dict[int, TrialState] = {}
        self.train_step: TrainStepFunction = train_step
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = get_worker_logger(worker_id=worker_state.id)
        self.tuner: ActorHandle = tuner
        self.mutation_iteration: int = MUTATION_ITERATION
        self.interrupt_set: set = set()
        self.trial_phase: TrialPhase = trial_phase
        self.dataloader_factory: DataloaderFactory = dataloader_factory
        self.is_stop: bool = False
        self.trial_current_iteration: Dict[int, int] = {}
        self.log("info", "初始化完成")

    def send_signal(self, trial_id: int) -> None:
        if trial_id not in self.active_trials.keys():
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
        self.trial_current_iteration[trial_state.id] = 0
        trial_state.status = TrialStatus.RUNNING
        trial_state.worker_id = self.worker_state.id
        trial_state.worker_type = self.worker_state.worker_type
        ray.wait(
            [self.tuner.record_trial_progress.remote(trial_state.without_checkpoint())],
            timeout=0.1,
        )  # type: ignore
        self.log("info", f"Running Trials: {[i for i in self.active_trials]}")

    def run(self) -> None:
        self.log("info", "開始執行 Training Loop")
        while not self.is_stop:
            update_result_futures: List[ObjectRef] = []
            active_trials = list(self.active_trials.values())
            for trial_state in active_trials:
                self.train(trial_state)

                status = trial_state.status
                if status == TrialStatus.TERMINATE:
                    update_result_futures.append(
                        self.tuner.submit_trial.remote(trial_state)
                    )  # type: ignore
                    self.active_trials.pop(trial_state.id)

                elif status == TrialStatus.PAUSE:
                    update_result_futures.append(
                        self.tuner.submit_trial.remote(trial_state)
                    )  # type: ignore
                    self.active_trials.pop(trial_state.id)

                elif status == TrialStatus.NEED_MUTATION:
                    update_result_futures.append(
                        self.tuner.submit_trial.remote(trial_state)
                    )  # type: ignore
                    self.active_trials.pop(trial_state.id)

                elif status == TrialStatus.INTERRUPTED:
                    update_result_futures.append(
                        self.tuner.submit_trial.remote(trial_state)
                    )  # type: ignore
                    self.active_trials.pop(trial_state.id)
                    continue

                update_result_futures.append(
                    self.tuner.update_trial_result.remote(
                        trial_state.without_checkpoint()
                    )
                )  # type: ignore
            ray.wait(update_result_futures, timeout=0.1)  # type: ignore

    def get_active_trials_nums(self) -> int:
        """
        取得目前活躍試驗的數量。

        Returns:
            int: 活躍試驗數量。
        """
        return len(self.active_trials)

    def get_active_trials(self) -> List[TrialState]:
        return list(self.active_trials.values())

    def get_available_slots(self) -> int:
        """
        取得可供分配的新試驗插槽數。

        Returns:
            int: 可分配試驗的插槽數。
        """
        return self.worker_state.max_trials - len(self.active_trials)

    def train(self, trial_state: TrialState) -> TrialState:
        """
        執行試驗的訓練流程。

        Args:
            trial_state (TrialState): 試驗狀態。

        Returns:
            TrialState: 訓練後的試驗狀態。
        """
        self.log("info", "開始訓練", trial_id=trial_state.id)
        self.log("debug", f"{torch.cuda.is_available()=}")

        try:
            hyper = trial_state.hyperparameter
            checkpoint = trial_state.checkpoint
            train_loader, test_loader, _ = self.dataloader_factory(hyper.batch_size)

            model = get_model(hyper.model_type)
            if checkpoint:
                model.load_state_dict(checkpoint.model_state_dict)
            model.to(self.device)

            optimizer = optim.SGD(
                model.parameters(), lr=hyper.lr, momentum=hyper.momentum
            )  # type: ignore
            if checkpoint:
                optimizer.load_state_dict(checkpoint.optimizer_state_dict)

            for param_group in optimizer.param_groups:
                param_group["lr"] = hyper.lr
                param_group["momentum"] = hyper.momentum

        except Exception as e:
            self.log("error", f"{type(e).__name__}: {e}", trial_state.id)
            trial_state.status = TrialStatus.FAILED
            return trial_state

        while True:
            # 是否達到結束條件
            if (
                trial_state.accuracy > trial_state.stop_accuracy
                or trial_state.iteration >= trial_state.stop_iteration
            ):
                self.log("info", "訓練結束", trial_id=trial_state.id)
                self.finish_trial(trial_state)
                trial_state.accuracy = self.test(model, test_loader)
                trial_state.update_checkpoint(model, optimizer)
                return trial_state

            # 當前階段是否超前
            if self.trial_phase.is_trial_exceeding(trial_state):
                self.log(
                    "info", f"已超過當前階段 Phase{self.trial_phase.current_phase}"
                )
                self.pause_trial(trial_state)
                trial_state.update_checkpoint(model, optimizer)
                return trial_state

            # 當前階段是否落後
            if (
                self.worker_state.worker_type == WorkerType.CPU
                and trial_state.phase < self.trial_phase.current_phase
            ):
                self.log("info", "已落後當前訓練階段")
                self.interrupt_trial(trial_state)
                trial_state.update_checkpoint(model, optimizer)
                return trial_state

            # GPU 是否達到次數上限
            if (
                self.worker_state.worker_type == WorkerType.GPU
                and self.trial_current_iteration[trial_state.id] >= GPU_MAX_ITERATION
            ):
                self.log("info", "已達到 GPU_MAX_ITERATION 次數")
                self.pause_trial(trial_state)
                trial_state.update_checkpoint(model, optimizer)
                return trial_state

            # 是否被中斷
            if trial_state.id in self.interrupt_set:
                self.log("info", "收到回傳訊號")
                self.interrupt_trial(trial_state)
                self.interrupt_set.remove(trial_state.id)
                trial_state.update_checkpoint(model, optimizer)
                return trial_state

            self.train_step(
                model, optimizer, train_loader, hyper.batch_size, self.device
            )

            trial_state.iteration += 1
            self.trial_current_iteration[trial_state.id] += 1
            trial_state.phase = self.trial_phase.get_trial_phase(trial_state)
            trial_state.device_iteration_count[self.worker_state.worker_type] += 1

            if trial_state.iteration % self.mutation_iteration == 0:
                break

        trial_state.accuracy = self.test(model, test_loader)

        self.log(
            "info",
            f"Phase: {trial_state.phase}, Iteration: {trial_state.iteration} Accuracy: {trial_state.accuracy}",
            trial_id=trial_state.id,
        )

        if (
            trial_state.accuracy > trial_state.stop_accuracy
            or trial_state.iteration >= trial_state.stop_iteration
        ):
            self.log("info", "訓練結束", trial_id=trial_state.id)
            self.finish_trial(trial_state)
            trial_state.update_checkpoint(model, optimizer)
            return trial_state

        if trial_state.status == TrialStatus.TERMINATE:
            return trial_state

        # base_line = ray.get(
        #     self.tuner.get_baseline.remote(iteration=trial_state.iteration)
        # )  # type: ignore
        ray.get(
            self.tuner.record_trial_progress.remote(trial_state.without_checkpoint())
        )
        lower_quantile, _ = ray.get(self.tuner.get_quantile_trial.remote())
        if trial_state in lower_quantile:
            # self.log(
            #     "info",
            #     f"Accuracy:{trial_state.accuracy:.4f} < Base Line:{base_line:.4f}",
            #     trial_id=trial_state.id,
            # )
            # if random.randint(1, 100) < 33:
            #     self.log(
            #         "info",
            #         "↩️ 需要執行突變，已回傳",
            #         trial_id=trial_state.id,
            #     )
            self.need_mutation_trial(trial_state)
        trial_state.update_checkpoint(model, optimizer)
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
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return correct / total

    def finish_trial(self, trial_state: TrialState) -> None:
        """
        將試驗標記為終止並從活躍列表中移除。

        Args:
            trial_state (TrialState): 試驗狀態。
        """
        trial_state.status = TrialStatus.TERMINATE

    def pause_trial(self, trial_state: TrialState) -> None:
        """
        將試驗標記為暫停並從活躍列表中移除。

        Args:
            trial_state (TrialState): 試驗狀態。
        """
        trial_state.status = TrialStatus.PAUSE

    def interrupt_trial(self, trial_state: TrialState) -> None:
        trial_state.status = TrialStatus.INTERRUPTED

    def need_mutation_trial(self, trial_state: TrialState):
        trial_state.status = TrialStatus.NEED_MUTATION

    def get_log_file(self) -> Dict[str, Union[int, str]]:
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

        with open(log_dir, "r") as f:
            return {"id": self.worker_state.id, "content": f.read()}

    def log(self, level: str, message: str, trial_id: Union[int, str] = "N/A") -> None:
        """
        根據指定的 log 級別輸出訊息。

        Args:
            level (str): 記錄等級（info/debug/warning/error/critical）。
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

    def get_worker_type(self) -> WorkerType:
        """
        回傳 worker 類型（CPU/GPU）。

        Returns:
            WorkerType: Worker 類型。
        """
        return self.worker_state.worker_type

    def stop(self):
        self.is_stop = True

    def update_phase(self, phase) -> None:
        self.log("info", f"更新階段到Phase {self.trial_phase.current_phase}")
        self.trial_phase.current_phase = phase


def generate_all_workers(
    tuner: ActorHandle,
    train_step: TrainStepFunction,
    dataloader_factory: DataloaderFactory,
) -> List[ActorHandle]:
    """
    根據 Ray 叢集的節點資源建立所有 Worker。

    Args:
        tuner (ActorHandle): 接收試驗結果的 Actor。
        train_step (TrainStepFunction): 訓練步驟函式。

    Returns:
        List[ActorHandle]: 建立的 Worker Actor 清單。
    """

    visited_address = set()
    worker_states = []
    index = 0
    head_node_address = get_head_node_address()
    print(head_node_address)

    for node in ray.nodes():
        node_address = node["NodeManagerAddress"]

        if node["Alive"]:
            if node_address in visited_address:
                continue

            resource = node["Resources"]
            if "CPU" in resource:
                if node_address == head_node_address:
                    cpus = min(resource.get("CPU", 1) - 1, 1)
                else:
                    cpus = resource.get("CPU", 1)

                worker_states.append(
                    WorkerState(
                        id=index,
                        num_cpus=cpus,
                        num_gpus=0,
                        node_name=f"node:{node_address}",
                        max_trials=1,
                        worker_type=WorkerType.CPU,
                    )
                )
                index += 1

            if "GPU" in resource:
                # for _ in range(int(resource["GPU"])):
                #     worker_states.append(
                #         WorkerState(
                #             id=index,
                #             num_cpus=0,
                #             # num_gpus=resource.get("GPU", 0),
                #             num_gpus=1,
                #             node_name=f"node:{node_address}",
                #             max_trials=7,
                #             worker_type=WorkerType.GPU,
                #         )
                #     )
                #     index += 1
                worker_states.append(
                    WorkerState(
                        id=index,
                        num_cpus=0,
                        num_gpus=resource["GPU"],
                        node_name=f"node:{node_address}",
                        max_trials=12,
                        worker_type=WorkerType.GPU,
                    )
                )
                index += 1
            visited_address.add(node_address)

    workers: list[ActorHandle] = []
    print(*worker_states, sep="\n")

    for index, worker_state in enumerate(worker_states):
        workers.append(
            Worker.options(  # type: ignore
                max_concurrency=worker_state.max_trials + 3,
                name=f"worker-{index}",
                num_cpus=worker_state.num_cpus,
                num_gpus=worker_state.num_gpus,
                resources={worker_state.node_name: 0.01},
            ).remote(
                worker_state,
                train_step=train_step,
                tuner=tuner,
                trial_phase=TrialPhase(STOP_ITERATION, PHASE_ITERATION),
                dataloader_factory=dataloader_factory,
            )
        )

    return workers
