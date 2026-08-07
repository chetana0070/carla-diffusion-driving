import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch

    from carla_diffusion.bc_model import SingleFrameBC, weighted_action_mse
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch training dependencies are not installed")
class BCModelTests(unittest.TestCase):
    def test_forward_shape_and_action_bounds(self) -> None:
        assert torch is not None
        model = SingleFrameBC(pretrained=False)
        model.eval()
        with torch.inference_mode():
            output = model(torch.zeros(2, 3, 224, 224), torch.zeros(2, 19))
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertTrue(bool(torch.all(torch.abs(output) <= 1.0)))

    def test_weighted_loss_normalizes_by_weight_sum(self) -> None:
        assert torch is not None
        prediction = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        target = torch.zeros(2, 2)
        weight = torch.tensor([1.0, 3.0])
        self.assertAlmostEqual(float(weighted_action_mse(prediction, target, weight)), 0.25)

    def test_frozen_encoder_stays_in_evaluation_mode(self) -> None:
        model = SingleFrameBC(pretrained=False)
        model.set_encoder_trainable(False)
        model.train()
        self.assertFalse(model.encoder.training)
        self.assertTrue(model.scalar_encoder.training)


if __name__ == "__main__":
    unittest.main()
