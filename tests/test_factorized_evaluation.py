import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carla_diffusion.diffusion_evaluation import summarize_action_pairs
from carla_diffusion.factorized_evaluation import factorized_promotion_decision


class FactorizedEvaluationTests(unittest.TestCase):
    def decision(self, predicted_step: float) -> dict[str, object]:
        actions = [[0.1, -0.4], [-0.1, 0.6]]
        candidate = summarize_action_pairs(actions, actions)
        baseline = summarize_action_pairs(actions, actions)
        return factorized_promotion_decision(
            candidate,
            baseline,
            predicted_smoothness={"mean_abs_longitudinal_step": predicted_step},
            target_smoothness={"mean_abs_longitudinal_step": 0.1},
            expected_validation_samples=2,
            validation_samples=2,
            expected_test_samples=2,
            maximum_relative_rmse=1.25,
            maximum_absolute_longitudinal_bias=0.1,
            maximum_longitudinal_behavior_drift=0.2,
            maximum_longitudinal_smoothness_ratio=1.5,
            latency_gate_passed=True,
        )

    def test_factorized_candidate_passes_all_gates(self) -> None:
        decision = self.decision(0.12)
        self.assertTrue(decision["gate_passed"])
        self.assertAlmostEqual(decision["longitudinal_smoothness_ratio"], 1.2)

    def test_longitudinal_chunk_oscillation_blocks_promotion(self) -> None:
        decision = self.decision(0.2)
        self.assertFalse(decision["gate_passed"])
        self.assertIn("longitudinal_chunk_smoothness", decision["failures"])


if __name__ == "__main__":
    unittest.main()
