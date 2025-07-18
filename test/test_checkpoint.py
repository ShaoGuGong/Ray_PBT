import unittest

from src.utils import Checkpoint


class TestCheckpoint(unittest.TestCase):
    def test_checkpoint_empty(self) -> None:
        empty_checkpoint = Checkpoint.empty()

        assert empty_checkpoint.is_empty(), "Checkpoint should be empty."


if __name__ == "__main__":
    unittest.main()
