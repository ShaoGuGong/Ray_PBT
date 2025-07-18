import unittest

from src.trial_state import Hyperparameter, TrialState
from src.utils import ModelType


class TestTrialState(unittest.TestCase):
    def test_trial_state_without_checkpoint(self) -> None:
        trial = TrialState(
            0,
            Hyperparameter(0.1, 0.2, 128, ModelType.RESNET_18),
            None,
            1000,
        )

        assert trial.checkpoint.is_empty(), "Trial checkpoint is not None"
        assert trial.snapshot.checkpoint.is_empty(), "Trial is not without checkpoint"


if __name__ == "__main__":
    unittest.main()
