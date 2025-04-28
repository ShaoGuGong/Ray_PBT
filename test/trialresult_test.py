import os
import random
import sys
from dataclasses import replace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


from trial_state import TrialResult
from utils import Hyperparameter, ModelType

tr = TrialResult(3, 0)


for iteration in range(1, 10 + 1):
    for i in range(10):
        tr.update_trial_result(
            iteration,
            random.uniform(0.0, 1.0),
            Hyperparameter(
                lr=10 ** random.uniform(-2, -4),
                momentum=random.uniform(0.001, 1),
                batch_size=random.choice([64, 128, 256, 512, 1024]),
                model_type=ModelType.RESNET_18,
            ),
        )


tr.display_results()


def mutation(hyperparameter: Hyperparameter) -> None:
    print(hyperparameter)
    mutation_options = (
        ("lr", random.uniform(0.001, 1)),
        ("momentum", random.uniform(0.001, 1)),
        ("batch_size", random.choice([64, 128, 256, 512, 1024])),
    )
    print(mutation_options)

    print(
        replace(
            hyperparameter,
            **{k: v for k, v in random.sample(mutation_options, 2)},
        )
    )


mutation(
    Hyperparameter(
        lr=10 ** random.uniform(-2, -4),
        momentum=random.uniform(0.001, 1),
        batch_size=random.choice([64, 128, 256, 512, 1024]),
        model_type=ModelType.RESNET_18,
    ),
)

l = [i[0] for i in tr.table[1]]
print(l)
print(sum(l))
print(sum(l) / len(l))

print(tr.get_mean_accuray(1))
