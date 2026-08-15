import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carla_diffusion.vla_runtime import HierarchicalPlanExecutor


class HierarchicalPlanExecutorTests(unittest.TestCase):
    def test_executes_only_frozen_prefix(self) -> None:
        executor = HierarchicalPlanExecutor(action_horizon=16, execute_steps=5)
        executor.install_plan([(index / 10, 0.2) for index in range(16)])
        actions = [executor.next_action() for _ in range(5)]
        self.assertEqual(actions[-1], (0.4, 0.2))
        self.assertTrue(executor.requires_plan)
        with self.assertRaises(RuntimeError):
            executor.next_action()

    def test_actions_are_bounded(self) -> None:
        executor = HierarchicalPlanExecutor(action_horizon=2, execute_steps=1)
        executor.install_plan([(2.0, -3.0), (0.0, 0.0)])
        self.assertEqual(executor.next_action(), (1.0, -1.0))

    def test_invalid_plan_is_rejected(self) -> None:
        executor = HierarchicalPlanExecutor(action_horizon=2, execute_steps=1)
        with self.assertRaises(ValueError):
            executor.install_plan([(0.0, 0.0)])


if __name__ == "__main__":
    unittest.main()
