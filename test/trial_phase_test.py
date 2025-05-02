import random
from typing import List

from src.config import STOP_ITERATION
from src.trial_phase import TrialPhase
from src.trial_state import TrialState
from src.utils import Hyperparameter, ModelType


def generate_trial_states(n: int = 1) -> List[TrialState]:
    return [
        TrialState(
            i,
            Hyperparameter(
                lr=random.uniform(0.001, 1),
                momentum=random.uniform(0.001, 1),
                batch_size=random.choice([64, 128, 256, 512, 1024]),
                model_type=ModelType.RESNET_18,
            ),
            stop_iteration=STOP_ITERATION,
        )
        for i in range(n)
    ]


tp = TrialPhase(generate_trial_states(10), 1300, 500)
print(tp.current_phase)
print(tp.first_phase)
print(tp.last_phase)
print(tp.trial_id_by_phase)
print(tp.thresholds)
