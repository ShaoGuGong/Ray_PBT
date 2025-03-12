import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

import ray
import torch
import torch.optim as optim
from ray.actor import ActorHandle

from trial import TrialState
from utils import TrialStatus, get_data_loader, get_model

# Type Define
Accuracy = float


class WorkerLoggerFormatter(logging.Formatter):
    def format(self, record):
        record.worker_id = getattr(record, "worker_id", "N/A")
        record.trial_id = getattr(record, "trial_id", "N/A")
        return super().format(record)


def get_worker_logger(worker_id: int) -> logging.Logger:
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


class TrainStepFunction(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> None: ...


@dataclass
class WorkerState:
    id: int
    num_cpus: int
    num_gpus: int
    node_name: str
    calculate_ability: float = 0.0
    max_trials: int = 1


@ray.remote
class Worker:
    def __init__(
        self, worker_state: WorkerState, train_step: TrainStepFunction
    ) -> None:
        self.worker_state = worker_state
        self.active_trials = {}
        self.train_step = train_step
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = get_worker_logger(worker_id=worker_state.id)
        self.logger.info("初始化完成", extra={"worker_id": self.worker_state.id})

    def assign_trial(self, trial_state: TrialState) -> TrialState:
        self.active_trials[trial_state.id] = trial_state
        trial_state.worker_id = self.worker_state.id
        return self.train(trial_state)

    def get_active_trials_nums(self) -> int:
        return len(self.active_trials)

    def has_available_slots(self) -> bool:
        return len(self.active_trials) < self.worker_state.max_trials

    def train(self, trial_state: TrialState) -> TrialState:
        self.logger.info(
            f"開始訓練",
            extra={"worker_id": self.worker_state.id, "trial_id": trial_state.id},
        )

        hyper = trial_state.hyperparameter
        checkpoint = trial_state.checkpoint
        train_loader, test_loader = get_data_loader(hyper.model_type, hyper.batch_size)

        model = get_model(hyper.model_type)
        model.load_state_dict(checkpoint.model_state_dict)
        model.to(self.device)

        optimizer = optim.SGD(model.parameters())
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

            self.logger.info(
                f"Iteration: {trial_state.iteration} Accuracy: {trial_state.accuracy}",
                extra={"worker_id": self.worker_state.id, "trial_id": trial_state.id},
            )

            trial_state.iteration += 1

        self.logger.info(
            f"訓練結束",
            extra={"worker_id": self.worker_state.id, "trial_id": trial_state.id},
        )
        self.finish_trial(trial_state)

        return trial_state

    def test(self, model, test_loader) -> Accuracy:
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
        # trial_state.worker_id = -1
        self.active_trials.pop(trial_state.id)
        trial_state.status = TrialStatus.TERMINAL

    def get_log_file(self) -> Dict[str, int]:
        log_dir = None

        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_dir = handler.baseFilename  # 取得資料夾路徑
                break

        if not log_dir:
            self.logger.error("Logs direction is not exists")
            return {"id": self.worker_state.id, "content": ""}

        with open(log_dir, "r") as f:
            return {"id": self.worker_state.id, "content": f.read()}


def generate_all_workers(train_step: TrainStepFunction) -> List[ActorHandle]:
    visited_address = set()
    worker_states = []
    index = 0

    # Fetch all avaiable resource from Ray cluster.
    for node in ray.nodes():
        if node["Alive"]:
            if node["NodeManagerAddress"] in visited_address:
                continue

            resource = node["Resources"]
            if "CPU" in resource:
                worker_states.append(
                    WorkerState(
                        id=index,
                        num_cpus=resource.get("CPU", 0),
                        num_gpus=0,
                        node_name=f"node:{node['NodeManagerAddress']}",
                    )
                )
                index += 1
            if "GPU" in resource:
                worker_states.append(
                    WorkerState(
                        id=index,
                        num_cpus=0,
                        num_gpus=resource.get("GPU", 0),
                        node_name=f"node:{node['NodeManagerAddress']}",
                    )
                )
                index += 1
            visited_address.add(node["NodeManagerAddress"])

    workers: list[ActorHandle] = []
    print(*worker_states)

    for index, worker_state in enumerate(worker_states):
        workers.append(
            Worker.options(
                max_concurrency=worker_state.max_trials + 1,
                name=f"worker-{index}",
                num_cpus=worker_state.num_cpus,
                num_gpus=worker_state.num_gpus,
                resources={worker_state.node_name: 0.01},
            ).remote(worker_state, train_step=train_step)
        )

    return workers
