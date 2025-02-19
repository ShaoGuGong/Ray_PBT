import os
import sys
from pprint import pp

import ray
from numpy import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from trial import Hyperparameter, TrialRunner
from tuner import Tuner
from utils import ModelType

if __name__ == "__main__":
    ray.init()

    hypers = []
    for i in range(10):
        hypers.append(
            Hyperparameter(
                lr=random.uniform(0.001, 1),
                momentum=random.uniform(0.001, 1),
                batch_size=512,
                model_type=ModelType.RESNET_18,
            )
        )

    pp(hypers)

    tuner = Tuner(hypers)
    runner = TrialRunner(tuner._generate_trail_states(), 100)
    print(runner.get_remaining_generation())
