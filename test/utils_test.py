import os
import sys
from pprint import pp

import ray
from numpy import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from utils import compose, pipe


def add(x):
    return x + 1


def mul(x):
    return x * 7


if __name__ == "__main__":
    print(compose(add, mul)(3))
    print(pipe(add, mul)(3))
