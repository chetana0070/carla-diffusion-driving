import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from carla_diffusion.vla_policy import (
    corrective_action_chunk_loss,
    corrective_selection_score,
)


class VLACorrectiveTests(unittest.TestCase):
    def loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return corrective_action_chunk_loss(
            prediction,
            target,
            torch.ones(prediction.shape[0]),
            torch.ones(prediction.shape[0]),
            chunk_steering_weight=1.0,
            chunk_longitudinal_weight=2.0,
            first_steering_weight=1.0,
            first_longitudinal_weight=3.0,
            steering_derivative_weight=10.0,
            longitudinal_derivative_weight=0.5,
            longitudinal_bias_weight=2.0,
        )

    def test_identical_chunks_have_zero_corrective_loss(self) -> None:
        target = torch.tensor([[[0.1, -0.2], [0.2, -0.1], [0.3, 0.0]]])
        loss, components = self.loss(target.clone(), target)
        self.assertAlmostEqual(float(loss), 0.0)
        self.assertTrue(all(float(value) == 0.0 for value in components.values()))

    def test_oscillation_and_positive_bias_are_penalized(self) -> None:
        target = torch.zeros(2, 4, 2)
        prediction = target.clone()
        prediction[:, :, 0] = torch.tensor([0.2, -0.2, 0.2, -0.2])
        prediction[:, :, 1] = 0.25
        loss, components = self.loss(prediction, target)
        self.assertGreater(float(loss), 0.0)
        self.assertGreater(float(components["steering_derivative_mse"]), 0.0)
        self.assertAlmostEqual(float(components["longitudinal_bias_squared"]), 0.0625)

    def test_selection_score_penalizes_only_headroom_violations(self) -> None:
        passing, passing_components = corrective_selection_score(
            joint_rmse=0.15,
            longitudinal_bias=0.05,
            steering_smoothness_ratio=1.1,
            target_maximum_absolute_bias=0.075,
            target_maximum_smoothness_ratio=1.25,
            bias_violation_weight=0.1,
            smoothness_violation_weight=0.05,
        )
        failing, failing_components = corrective_selection_score(
            joint_rmse=0.15,
            longitudinal_bias=0.1,
            steering_smoothness_ratio=2.5,
            target_maximum_absolute_bias=0.075,
            target_maximum_smoothness_ratio=1.25,
            bias_violation_weight=0.1,
            smoothness_violation_weight=0.05,
        )
        self.assertAlmostEqual(passing, 0.15)
        self.assertEqual(passing_components["bias_violation"], 0.0)
        self.assertGreater(failing, passing)
        self.assertGreater(failing_components["smoothness_violation"], 0.0)


if __name__ == "__main__":
    unittest.main()
