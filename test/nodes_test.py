import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from pprint import pp

import ray

ray.init()
if __name__ == "__main__":
    print(ray.get_runtime_context().gcs_address)
