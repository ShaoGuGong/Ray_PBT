import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import ray

from worker import fetch_nodes_resources

ray.init()
if __name__ == "__main__":
    fetch_nodes_resources()
