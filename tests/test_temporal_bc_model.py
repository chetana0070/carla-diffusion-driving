import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch

    from carla_diffusion.temporal_bc_model import TemporalBC
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch training dependencies are not installed")
class TemporalBCModelTests(unittest.TestCase):
    def test_forward_shape_and_bounds(self) -> None:
        assert torch is not None
        model = TemporalBC(pretrained=False)
        model.eval()
        with torch.inference_mode():
            output = model(
                torch.zeros(2, 4, 3, 224, 224),
                torch.zeros(2, 4, 10),
                torch.zeros(2, 9),
            )
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertTrue(bool(torch.all(torch.abs(output) <= 1.0)))

    def test_history_mismatch_is_rejected(self) -> None:
        assert torch is not None
        model = TemporalBC(pretrained=False)
        with self.assertRaises(ValueError):
            model(
                torch.zeros(1, 3, 3, 224, 224),
                torch.zeros(1, 3, 10),
                torch.zeros(1, 9),
            )

    def test_frozen_backbone_stays_in_evaluation_mode(self) -> None:
        model = TemporalBC(pretrained=False)
        model.set_encoder_trainable(False)
        model.train()
        self.assertFalse(model.encoder.training)
        self.assertTrue(model.temporal_encoder.training)


if __name__ == "__main__":
    unittest.main()
