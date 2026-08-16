import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carla_diffusion.diffusion_evaluation import summarize_action_pairs
from carla_diffusion.vla_evaluation import (
    apply_fresh_holdout_governance,
    summarize_instruction_slices,
    vla_promotion_decision,
)


class VLAEvaluationTests(unittest.TestCase):
    def decision(self, predicted_step: float = 0.1) -> dict[str, object]:
        actions = [[0.1, -0.4], [-0.1, 0.6]]
        candidate = summarize_action_pairs(actions, actions)
        baseline = summarize_action_pairs(actions, actions)
        return vla_promotion_decision(
            candidate,
            baseline,
            predicted_smoothness={
                "mean_abs_steering_step": predicted_step,
                "mean_abs_longitudinal_step": predicted_step,
            },
            target_smoothness={
                "mean_abs_steering_step": 0.1,
                "mean_abs_longitudinal_step": 0.1,
            },
            expected_validation_samples=2,
            validation_samples=2,
            expected_test_samples=2,
            maximum_relative_rmse=1.25,
            maximum_absolute_longitudinal_bias=0.1,
            maximum_longitudinal_behavior_drift=0.2,
            maximum_chunk_smoothness_ratio=1.5,
            latency_gate_passed=True,
            checkpoint_integrity_passed=True,
            baseline_provenance_passed=True,
            instruction_slice_coverage_passed=True,
        )

    def test_identical_candidate_passes(self) -> None:
        self.assertTrue(self.decision()["gate_passed"])

    def test_provenance_and_smoothness_failures_are_named(self) -> None:
        decision = self.decision(0.2)
        self.assertFalse(decision["gate_passed"])
        self.assertIn("chunk_smoothness", decision["failures"])

    def test_instruction_slices_preserve_counts(self) -> None:
        actions = [[0.1, 0.2], [-0.2, -0.3], [0.0, 0.4]]
        summary = summarize_instruction_slices(
            actions,
            actions,
            ["left", "right", "left"],
            ["green", "red", "green"],
        )
        self.assertEqual(summary["route_command"]["left"]["samples"], 2)
        self.assertEqual(summary["traffic_light_state"]["red"]["samples"], 1)

    def test_observed_test_cannot_authorize_promotion(self) -> None:
        governed = apply_fresh_holdout_governance(
            self.decision(),
            fresh_holdout_required=True,
            fresh_holdout_passed=False,
        )
        self.assertTrue(governed["technical_gate_passed"])
        self.assertFalse(governed["gate_passed"])
        self.assertEqual(governed["failures"], ["fresh_holdout"])


if __name__ == "__main__":
    unittest.main()
