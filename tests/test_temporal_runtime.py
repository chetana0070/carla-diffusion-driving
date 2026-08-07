import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carla_diffusion.temporal_runtime import TemporalHistoryBuffer


class TemporalRuntimeTests(unittest.TestCase):
    def test_first_observation_initializes_complete_history(self) -> None:
        history: TemporalHistoryBuffer[str, int] = TemporalHistoryBuffer(4)
        history.append("image-1", 1)
        images, states = history.sequences()
        self.assertEqual(images, ("image-1",) * 4)
        self.assertEqual(states, (1,) * 4)

    def test_history_rolls_forward_in_lockstep(self) -> None:
        history: TemporalHistoryBuffer[str, int] = TemporalHistoryBuffer(4)
        history.append("image-1", 1)
        history.append("image-2", 2)
        history.append("image-3", 3)
        images, states = history.sequences()
        self.assertEqual(images, ("image-1", "image-1", "image-2", "image-3"))
        self.assertEqual(states, (1, 1, 2, 3))

    def test_reset_prevents_cross_episode_leakage(self) -> None:
        history: TemporalHistoryBuffer[str, int] = TemporalHistoryBuffer(4)
        history.append("old", 1)
        history.reset()
        history.append("new", 2)
        images, states = history.sequences()
        self.assertEqual(images, ("new",) * 4)
        self.assertEqual(states, (2,) * 4)

    def test_invalid_history_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TemporalHistoryBuffer[str, int](1)


if __name__ == "__main__":
    unittest.main()
