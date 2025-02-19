import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import ray
from pprint import pp
from worker import generate_all_workers

ray.init()
if __name__ == "__main__":
    pp(generate_all_workers())
