import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from pprint import pp

import ray

from worker import generate_all_workers

ray.init()
if __name__ == "__main__":
    workers = generate_all_workers()

    print(ray.get_actor("worker-0"))
    print(workers)
