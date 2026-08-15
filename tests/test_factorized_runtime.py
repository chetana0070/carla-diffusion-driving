import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carla_diffusion.factorized_runtime import (
    ActionChunkExecutor,
    deployment_noise_seed,
)


class ActionChunkExecutorTests(unittest.TestCase):
    def test_executes_only_the_frozen_prefix(self) -> None:
        executor = ActionChunkExecutor(2)
        executor.set_plan(((0.1, 0.2), (0.3, 0.4), (0.5, 0.6)))
        self.assertEqual(executor.pop(), (0.1, 0.2))
        self.assertEqual(executor.remaining_steps, 1)
        self.assertEqual(executor.pop(), (0.3, 0.4))
        self.assertTrue(executor.needs_replan)

    def test_reset_requires_a_new_plan(self) -> None:
        executor = ActionChunkExecutor(1)
        executor.set_plan(((0.1, 0.2),))
        executor.reset()
        with self.assertRaises(RuntimeError):
            executor.pop()

    def test_nonfinite_and_out_of_range_actions_are_rejected(self) -> None:
        executor = ActionChunkExecutor(1)
        with self.assertRaises(ValueError):
            executor.set_plan(((float("nan"), 0.0),))
        with self.assertRaises(ValueError):
            executor.set_plan(((1.1, 0.0),))

    def test_released_checkpoint_uses_frozen_selection_seed(self) -> None:
        config = {"factorized_policy": {"selection_noise_seed": 17}}
        self.assertEqual(deployment_noise_seed(config), 17)

    def test_new_deployment_metadata_overrides_fallback(self) -> None:
        config = {
            "factorized_policy": {"selection_noise_seed": 17},
            "phase7_closed_loop_evaluation": {"deployment_noise_seed": 29},
        }
        self.assertEqual(deployment_noise_seed(config), 29)

    def test_noise_seed_rejects_invalid_types(self) -> None:
        with self.assertRaises(TypeError):
            deployment_noise_seed({"factorized_policy": []})
        with self.assertRaises(TypeError):
            deployment_noise_seed(
                {"factorized_policy": {"selection_noise_seed": "17"}}
            )


if __name__ == "__main__":
    unittest.main()
