from utils import Checkpoint, Hyperparameter


class TrialState:
    def __init__(self, hyperparameter: Hyperparameter) -> None:
        self.hyperparameter = hyperparameter
        # self.checkpoint_state =
