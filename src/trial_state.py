import torch.optim as optim

from utils import Checkpoint, Hyperparameter, TrialStatus, get_model


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
