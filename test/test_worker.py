import unittest

import ray

from src.worker import generate_all_workers


class TestWorker(unittest.TestCase):
    def test_generate_all_workers(self):
        generate_all_workers(None, None)


if __name__ == "__main__":

    ray.init()
    unittest.main()
