import logging
import os
import random
from typing import Dict, List, Optional, Union

import ray
import torch
import torch.optim as optim
from ray.actor import ActorHandle

from trial_state import TrialState
from utils import (Accuracy, TrainStepFunction, TrialStatus, WorkerState,
                   get_data_loader, get_head_node_address, get_model)


class WorkerLoggerFormatter(logging.Formatter):
    """
    自定義的日誌格式化器，為日誌添加 worker_id 和 trial_id。

    Attributes:
        None
    """

    def format(self, record):
        """
        格式化日誌信息，並添加 worker_id 和 trial_id。

        Args:
            record (logging.LogRecord): 日誌記錄。

        Returns:
            str: 格式化的日誌消息。
        """
        record.worker_id = getattr(record, "worker_id", "N/A")
        record.trial_id = getattr(record, "trial_id", "N/A")
        return super().format(record)


def get_worker_logger(worker_id: int) -> logging.Logger:
    """
    創建並返回一個針對特定 worker_id 的 logger，支持終端輸出與文件寫入。

    Args:
        worker_id (int): 工作者的唯一 ID。

    Returns:
        logging.Logger: 配置好的 logger。
    """

    # 確保 logs 目錄存在
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 使用 worker_id 創建唯一的 logger 名稱
    logger = logging.getLogger(f"worker-{worker_id}")

    # 防止重複添加 handler
    if not logger.handlers:
        # 設定 logger 的最基本級別
        logger.setLevel(logging.DEBUG)  # 或者選擇更合適的級別

        # 統一格式設定
        formatter = WorkerLoggerFormatter(
            "[%(asctime)s] %(levelname)s WORKER_ID: %(worker_id)s TRIAL_ID: %(trial_id)s -- %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 設定 stream handler (只顯示在終端)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)  # 只顯示 INFO 級別以上的訊息
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # 設定 file handler (將日誌寫入文件)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"worker-{worker_id}.log")
        )
        file_handler.setLevel(logging.DEBUG)  # 記錄所有級別的日誌
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


@ray.remote
class Worker:
    """
    代表一個工作者節點的類，負責訓練一個試驗並更新其結果。

    Attributes:
        worker_state (WorkerState): 工作者的狀態信息。
        active_trials (dict): 活躍的試驗字典，鍵為試驗 ID，值為試驗狀態。
        train_step (TrainStepFunction): 訓練步驟函數。
        device (torch.device): 訓練所使用的設備（CUDA 或 CPU）。
        logger (logging.Logger): 用於記錄訓練過程的 logger。
    """

    def __init__(
        self,
        worker_state: WorkerState,
        train_step: TrainStepFunction,
        tuner: ActorHandle,
    ) -> None:
        """
        初始化 Worker 類，並設置相關屬性。

        Args:
            worker_state (WorkerState): 工作者的狀態信息。
            train_result (ActorHandle): 用於更新訓練結果的 Actor。
            train_step (TrainStepFunction): 訓練步驟函數。
        """

        self.worker_state = worker_state
        self.active_trials = {}
        self.train_step = train_step
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = get_worker_logger(worker_id=worker_state.id)
        self.log("info", "初始化完成")
        self.tuner = tuner
        self.mutation_iteration: int = 2

    def assign_trial(self, trial_state: TrialState) -> TrialState:
        """
        分配一個試驗給這個工作者，並開始訓練。

        Args:
            trial_state (TrialState): 試驗的狀態。

        Returns:
            TrialState: 更新後的試驗狀態。
        """

        self.active_trials[trial_state.id] = trial_state
        trial_state.worker_id = self.worker_state.id
        return self.train(trial_state)

    def get_active_trials_nums(self) -> int:
        """
        獲取目前活躍的試驗數量。

        Returns:
            int: 活躍試驗數量。
        """
        return len(self.active_trials)

    def has_available_slots(self) -> bool:
        """
        判斷這個工作者是否還有可用的插槽來處理更多的試驗。

        Returns:
            bool: 如果有可用插槽則返回 True，否則返回 False。
        """
        return len(self.active_trials) < self.worker_state.max_trials

    def train(self, trial_state: TrialState) -> TrialState:
        """
        執行一個訓練過程，並更新試驗狀態。

        Args:
            trial_state (TrialState): 試驗的狀態。

        Returns:
            TrialState: 更新後的試驗狀態。
        """
        self.log("info", "開始訓練", trial_id=trial_state.id)

        hyper = trial_state.hyperparameter
        checkpoint = trial_state.checkpoint
        train_loader, test_loader = get_data_loader(hyper.model_type, hyper.batch_size)

        model = get_model(hyper.model_type)
        model.load_state_dict(checkpoint.model_state_dict)
        model.to(self.device)

        optimizer = optim.SGD(model.parameters(), lr=hyper.lr, momentum=hyper.momentum)
        optimizer.load_state_dict(checkpoint.optimzer_state_dict)
        for param_group in optimizer.param_groups:
            param_group["lr"] = hyper.lr
            param_group["momentum"] = hyper.momentum

        while True:
            if trial_state.accuracy > trial_state.stop_accuracy:
                break

            if trial_state.iteration >= trial_state.stop_iteration:
                break

            self.train_step(
                model, optimizer, train_loader, hyper.batch_size, self.device
            )

            trial_state.accuracy = self.test(model, test_loader)

            self.log(
                "info",
                f"Iteration: {trial_state.iteration} Accuracy: {trial_state.accuracy}",
                trial_id=trial_state.id,
            )

            trial_state.iteration += 1

            if trial_state.iteration % self.mutation_iteration:
                continue

            ray.get(
                self.tuner.update_trial_result.remote(
                    iteration=trial_state.iteration,
                    accuracy=trial_state.accuracy,
                    hyperparameter=trial_state.hyperparameter,
                )
            )

            base_line = ray.get(
                self.tuner.get_mean_accuracy.remote(iteration=trial_state.iteration)
            )

            if trial_state.accuracy >= base_line:
                continue

            self.log(
                "info",
                f"Accuracy:{trial_state.accuracy:.4f} < Base Line:{base_line:.4f}",
                trial_id=trial_state.id,
            )

            if random.choice((False, False, True)):
                self.log(
                    "info",
                    f"訓練中止，Trial回傳",
                    trial_id=trial_state.id,
                )

            self.pause_trial(trial_state)
            return trial_state

        self.log("info", f"訓練結束", trial_id=trial_state.id)
        self.finish_trial(trial_state)

        return trial_state

    def test(self, model, test_loader) -> Accuracy:
        """
        測試模型並計算準確率。

        Args:
            model (torch.nn.Module): 訓練好的模型。
            test_loader (torch.utils.data.DataLoader): 測試數據加載器。

        Returns:
            Accuracy: 模型在測試集上的準確率。
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
        結束並標記一個試驗的狀態為終止。

        Args:
            trial_state (TrialState): 試驗狀態。
        """

        self.active_trials.pop(trial_state.id)
        trial_state.status = TrialStatus.TERMINAL

    def pause_trial(self, trial_state: TrialState) -> None:
        self.active_trials.pop(trial_state.id)
        trial_state.status = TrialStatus.PAUSE

    def get_log_file(self) -> Dict[str, Union[int, str]]:
        """
        獲取工作者的日誌文件內容。

        Returns:
            dict: 包含 worker_id 和日誌內容的字典。
        """

        log_dir = None
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_dir = handler.baseFilename  # 取得資料夾路徑
                break

        if not log_dir:
            self.log("error", "Logs direction is not exists")
            return {"id": self.worker_state.id, "content": ""}

        with open(log_dir, "r") as f:
            return {"id": self.worker_state.id, "content": f.read()}

    def log(self, level: str, message: str, trial_id: Optional[int] = None) -> None:
        if level == "info":
            self.logger.info(
                message, extra={"worker_id": self.worker_state.id, "trial_id": trial_id}
            )
            return
        if level == "debug":
            self.logger.info(
                message, extra={"worker_id": self.worker_state.id, "trial_id": trial_id}
            )
            return
        if level == "warning":
            self.logger.warning(
                message, extra={"worker_id": self.worker_state.id, "trial_id": trial_id}
            )
            return
        if level == "critical":
            self.logger.critical(
                message, extra={"worker_id": self.worker_state.id, "trial_id": trial_id}
            )
            return
        if level == "error":
            self.logger.error(
                message, extra={"worker_id": self.worker_state.id, "trial_id": trial_id}
            )
            return


def generate_all_workers(
    tuner: ActorHandle, train_step: TrainStepFunction
) -> List[ActorHandle]:
    """
    根據 Ray 集群中的節點資源創建工作者，並返回工作者 Actor 列表。

    Args:
        train_result (ActorHandle): 用於更新訓練結果的 Actor。
        train_step (TrainStepFunction): 訓練步驟函數。

    Returns:
        List[ActorHandle]: 創建的工作者 Actor 列表。
    """

    visited_address = set()
    worker_states = []
    index = 0
    head_node_address = get_head_node_address()
    print(head_node_address)

    # Fetch all avaiable resource from Ray cluster.
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
                # cpus = resource.get("CPU", 1)

                worker_states.append(
                    WorkerState(
                        id=index,
                        num_cpus=cpus,
                        num_gpus=0,
                        node_name=f"node:{node_address}",
                    )
                )
                index += 1

            if "GPU" in resource:
                worker_states.append(
                    WorkerState(
                        id=index,
                        num_cpus=0,
                        num_gpus=resource.get("GPU", 0),
                        node_name=f"node:{node_address}",
                    )
                )
                index += 1
            visited_address.add(node_address)

    workers: list[ActorHandle] = []
    print(*worker_states, sep="\n")

    for index, worker_state in enumerate(worker_states):
        workers.append(
            Worker.options(
                max_concurrency=worker_state.max_trials + 1,
                name=f"worker-{index}",
                num_cpus=worker_state.num_cpus,
                num_gpus=worker_state.num_gpus,
                resources={worker_state.node_name: 0.01},
            ).remote(worker_state, train_step=train_step, tuner=tuner)
        )

    return workers
