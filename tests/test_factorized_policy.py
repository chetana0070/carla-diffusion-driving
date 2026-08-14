import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch

    from carla_diffusion.diffusion_policy import DiffusionSchedule
    from carla_diffusion.factorized_policy import (
        FactorizedTemporalPolicy,
        load_diffusion_warm_start,
        longitudinal_modes,
        weighted_longitudinal_derivative_loss,
        weighted_longitudinal_loss,
        weighted_mode_classification_loss,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch training dependencies are not installed")
class FactorizedPolicyTests(unittest.TestCase):
    def build_model(self) -> "FactorizedTemporalPolicy":
        return FactorizedTemporalPolicy(
            history_frames=4,
            action_horizon=4,
            action_dimension=2,
            image_projection_dimension=16,
            state_projection_dimension=8,
            temporal_hidden_dimension=16,
            condition_projection_dimension=8,
            denoiser_dimension=16,
            denoiser_layers=1,
            denoiser_heads=4,
            longitudinal_hidden_dimension=16,
            dropout=0.0,
        )

    def inputs(self) -> tuple["torch.Tensor", ...]:
        assert torch is not None
        return (
            torch.zeros(2, 4, 3, 32, 32),
            torch.zeros(2, 4, 10),
            torch.zeros(2, 9),
        )

    def test_forward_shapes_and_bounds(self) -> None:
        assert torch is not None
        model = self.build_model().eval()
        images, states, condition = self.inputs()
        with torch.inference_mode():
            noise, longitudinal, logits = model(
                torch.zeros(2, 4, 2),
                torch.tensor([0, 9]),
                images,
                states,
                condition,
            )
        self.assertEqual(tuple(noise.shape), (2, 4, 2))
        self.assertEqual(tuple(longitudinal.shape), (2, 4))
        self.assertEqual(tuple(logits.shape), (2, 4, 3))
        self.assertTrue(bool(torch.all(torch.abs(longitudinal) <= 1.0)))

    def test_sampling_is_deterministic_and_factorized(self) -> None:
        assert torch is not None
        torch.manual_seed(4)
        model = self.build_model().eval()
        schedule = DiffusionSchedule(20)
        images, states, condition = self.inputs()
        initial = torch.zeros(2, 4, 2)
        first = model.sample(
            schedule,
            images,
            states,
            condition,
            inference_steps=5,
            initial_noise=initial,
        )
        second = model.sample(
            schedule,
            images,
            states,
            condition,
            inference_steps=5,
            initial_noise=initial,
        )
        self.assertTrue(torch.allclose(first, second))
        self.assertEqual(tuple(first.shape), (2, 4, 2))
        self.assertTrue(bool(torch.all(torch.abs(first) <= 1.0)))

    def test_longitudinal_modes(self) -> None:
        assert torch is not None
        target = torch.tensor([[-0.2, -0.01, 0.2]])
        modes = longitudinal_modes(target, neutral_threshold=0.05)
        self.assertEqual(modes.tolist(), [[0, 1, 2]])

    def test_longitudinal_losses_are_finite(self) -> None:
        assert torch is not None
        target = torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.1, 0.8]])
        prediction = torch.zeros_like(target)
        sample_weights = torch.tensor([1.0, 0.5])
        class_weights = torch.tensor([1.0, 0.5, 1.5])
        action = weighted_longitudinal_loss(
            prediction,
            target,
            sample_weights,
            mode_weights=class_weights,
        )
        derivative = weighted_longitudinal_derivative_loss(
            prediction, target, sample_weights
        )
        classification = weighted_mode_classification_loss(
            torch.zeros(2, 3, 3),
            target,
            sample_weights,
            class_weights=class_weights,
        )
        self.assertGreater(float(action), 0)
        self.assertGreater(float(derivative), 0)
        self.assertGreater(float(classification), 0)

    def test_frozen_encoder_remains_in_evaluation_mode(self) -> None:
        model = self.build_model()
        model.set_encoder_trainable(False)
        model.train()
        self.assertFalse(model.diffusion.encoder.training)
        self.assertTrue(model.longitudinal_trunk.training)

    def test_warm_start_rejects_non_mapping_state(self) -> None:
        model = self.build_model()
        checkpoint = {
            "model_type": "temporal_diffusion_policy",
            "model_state_dict": None,
        }
        with self.assertRaises(TypeError):
            load_diffusion_warm_start(model, checkpoint)


if __name__ == "__main__":
    unittest.main()
