import unittest
import warnings

import ray

from src.worker import generate_all_workers


class TestWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore", ResourceWarning)
        ray.init(ignore_reinit_error=True)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_generate_all_workers(self):
        workers = generate_all_workers(None, None, None)
        assert len(workers) == 1, "Worker just one."


if __name__ == "__main__":
    unittest.main()
