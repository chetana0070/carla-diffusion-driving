import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carla_diffusion.vla_smoothing import (
    SMOOTHER_TYPE,
    apply_checkpoint_smoother,
    causal_steering_smoother,
    checkpoint_smoother_alpha,
    select_steering_smoothing,
)


class VLASmoothingTests(unittest.TestCase):
    def test_smoother_preserves_first_action_and_longitudinal(self) -> None:
        actions = torch.tensor(
            [[[0.2, 0.4], [0.8, -0.2], [-0.6, 0.1], [0.4, 0.3]]]
        )
        smoothed = causal_steering_smoother(actions, 0.5)
        self.assertTrue(torch.equal(smoothed[:, 0], actions[:, 0]))
        self.assertTrue(torch.equal(smoothed[:, :, 1], actions[:, :, 1]))
        raw_steps = torch.mean(torch.abs(actions[:, 1:, 0] - actions[:, :-1, 0]))
        smooth_steps = torch.mean(
            torch.abs(smoothed[:, 1:, 0] - smoothed[:, :-1, 0])
        )
        self.assertLess(float(smooth_steps), float(raw_steps))

    def test_checkpoint_transform_is_explicit(self) -> None:
        checkpoint = {
            "deployment_transform": {
                "type": SMOOTHER_TYPE,
                "alpha": 0.4,
                "preserve_first_action": True,
            }
        }
        actions = torch.tensor([[[0.0, 0.2], [1.0, 0.3]]])
        self.assertEqual(checkpoint_smoother_alpha(checkpoint), 0.4)
        transformed = apply_checkpoint_smoother(actions, checkpoint)
        self.assertAlmostEqual(float(transformed[0, 1, 0]), 0.4)

    def test_base_checkpoint_is_unchanged(self) -> None:
        actions = torch.tensor([[[0.0, 0.2], [1.0, 0.3]]])
        self.assertIs(apply_checkpoint_smoother(actions, {}), actions)

    def test_calibration_selects_largest_eligible_alpha(self) -> None:
        predicted = [
            [[0.0, 0.2], [0.8, 0.2], [-0.8, 0.2], [0.8, 0.2]],
            [[0.0, -0.2], [-0.8, -0.2], [0.8, -0.2], [-0.8, -0.2]],
        ]
        targets = [
            [[0.0, 0.2], [0.2, 0.2], [0.0, 0.2], [0.2, 0.2]],
            [[0.0, -0.2], [-0.2, -0.2], [0.0, -0.2], [-0.2, -0.2]],
        ]
        calibration = select_steering_smoothing(
            predicted,
            targets,
            [1.0, 0.5, 0.25, 0.1],
            target_maximum_smoothness_ratio=1.5,
            maximum_full_chunk_steering_rmse_degradation_fraction=1.0,
        )
        self.assertEqual(calibration["status"], "passed")
        selected = float(calibration["selected_alpha"])
        eligible = [
            float(candidate["alpha"])
            for candidate in calibration["candidates"]
            if candidate["eligible"]
        ]
        self.assertEqual(selected, max(eligible))

    def test_invalid_alpha_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            causal_steering_smoother(torch.zeros(1, 2, 2), 0.0)


if __name__ == "__main__":
    unittest.main()
