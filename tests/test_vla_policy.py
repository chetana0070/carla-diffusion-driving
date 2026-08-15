import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from carla_diffusion.vla_policy import HierarchicalVLAPlanner, weighted_action_chunk_loss


class VLAPolicyTests(unittest.TestCase):
    def test_forward_shape_bounds_and_loss(self) -> None:
        model = HierarchicalVLAPlanner(
            vocabulary_size=37,
            state_dimension=10,
            action_horizon=16,
            language_dimension=16,
            state_hidden_dimension=16,
            fusion_dimension=32,
        )
        prediction = model(
            torch.rand(2, 4, 3, 64, 64),
            torch.rand(2, 4, 10),
            torch.ones(2, 20, dtype=torch.long),
            torch.ones(2, 20),
        )
        self.assertEqual(tuple(prediction.shape), (2, 16, 2))
        self.assertTrue(bool(torch.all(torch.abs(prediction) <= 1.0)))
        loss = weighted_action_chunk_loss(
            prediction,
            torch.zeros_like(prediction),
            torch.ones(2),
        )
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_wrong_image_rank_is_rejected(self) -> None:
        model = HierarchicalVLAPlanner(
            vocabulary_size=8,
            state_dimension=10,
            action_horizon=4,
        )
        with self.assertRaises(ValueError):
            model(
                torch.rand(2, 3, 64, 64),
                torch.rand(2, 4, 10),
                torch.ones(2, 8, dtype=torch.long),
                torch.ones(2, 8),
            )


if __name__ == "__main__":
    unittest.main()
