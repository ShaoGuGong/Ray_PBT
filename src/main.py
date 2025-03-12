import ray
from numpy import random

from trial import Hyperparameter
from tuner import Tuner
from utils import ModelType

ray.init()

if __name__ == "__main__":
    hyperparameters = [
        Hyperparameter(
            lr=random.uniform(0.001, 1),
            momentum=random.uniform(0.001, 1),
            batch_size=32,
            model_type=ModelType.RESNET_18,
        )
        for _ in range(10)
    ]

    tuner = Tuner(hyperparameters)

    tuner.run()
