import os
import sys
from pprint import pp

import ray
from numpy import random

import src.config
from src.utils import colored_progress_bar, compose, pipe


def add(x):
    return x + 1


def mul(x):
    return x * 7


if __name__ == "__main__":
    x = [[i, 20 - i] for i in range(20)]
    for i in x:
        print(colored_progress_bar(i, 20))
