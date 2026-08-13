import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carla_diffusion.diffusion_evaluation import (
    promotion_decision,
    summarize_action_pairs,
    summarize_chunk_smoothness,
)


class DiffusionEvaluationTests(unittest.TestCase):
    def test_action_summary_separates_dimensions_and_behavior(self) -> None:
        metrics = summarize_action_pairs(
            [[0.0, 0.2], [0.2, -0.2], [-0.2, 0.0]],
            [[0.0, 0.1], [0.0, -0.1], [0.0, 0.0]],
        )
        self.assertEqual(metrics["samples"], 3)
        self.assertAlmostEqual(metrics["steering"]["mae"], 0.4 / 3)
        self.assertAlmostEqual(metrics["longitudinal"]["bias"], 0.0)
        self.assertAlmostEqual(
            metrics["predicted_longitudinal_behavior"]["neutral_fraction"],
            1 / 3,
        )

    def test_chunk_smoothness_uses_adjacent_actions(self) -> None:
        metrics = summarize_chunk_smoothness(
            [[[0.0, 0.0], [0.2, 0.1], [0.1, -0.1]]]
        )
        self.assertAlmostEqual(metrics["mean_abs_steering_step"], 0.15)
        self.assertAlmostEqual(metrics["mean_abs_longitudinal_step"], 0.15)
        self.assertAlmostEqual(metrics["maximum_abs_longitudinal_step"], 0.2)

    def test_promotion_names_longitudinal_regression(self) -> None:
        baseline = summarize_action_pairs(
            [[0.0, 0.1], [0.1, 0.2]],
            [[0.0, 0.0], [0.0, 0.0]],
        )
        diffusion = summarize_action_pairs(
            [[0.0, 0.8], [0.1, 0.8]],
            [[0.0, 0.0], [0.0, 0.0]],
        )
        decision = promotion_decision(
            diffusion,
            baseline,
            expected_validation_samples=2,
            validation_samples=2,
            expected_test_samples=2,
            maximum_relative_rmse=1.25,
            maximum_absolute_longitudinal_bias=0.1,
            maximum_longitudinal_behavior_drift=0.2,
            latency_gate_passed=True,
        )
        self.assertFalse(decision["gate_passed"])
        self.assertIn("longitudinal_rmse", decision["failures"])
        self.assertIn("longitudinal_bias", decision["failures"])
        self.assertIn("longitudinal_behavior", decision["failures"])

    def test_identical_policy_passes_promotion(self) -> None:
        metrics = summarize_action_pairs(
            [[0.1, 0.2], [-0.1, -0.2]],
            [[0.0, 0.1], [0.0, -0.1]],
        )
        decision = promotion_decision(
            metrics,
            metrics,
            expected_validation_samples=2,
            validation_samples=2,
            expected_test_samples=2,
            maximum_relative_rmse=1.25,
            maximum_absolute_longitudinal_bias=0.1,
            maximum_longitudinal_behavior_drift=0.2,
            latency_gate_passed=True,
        )
        self.assertTrue(decision["gate_passed"])
        self.assertEqual(decision["failures"], [])


if __name__ == "__main__":
    unittest.main()
