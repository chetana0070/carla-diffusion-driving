import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch

    from carla_diffusion.diffusion_policy import (
        DiffusionSchedule,
        TemporalDiffusionPolicy,
        cosine_beta_schedule,
        weighted_action_reconstruction_loss,
        weighted_noise_mse,
        weighted_temporal_derivative_loss,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch training dependencies are not installed")
class DiffusionPolicyTests(unittest.TestCase):
    def build_model(self) -> "TemporalDiffusionPolicy":
        return TemporalDiffusionPolicy(
            history_frames=4,
            action_horizon=16,
            action_dimension=2,
            image_projection_dimension=32,
            state_projection_dimension=16,
            temporal_hidden_dimension=32,
            condition_projection_dimension=16,
            denoiser_dimension=32,
            denoiser_layers=1,
            denoiser_heads=4,
            dropout=0.0,
        )

    def inputs(self) -> tuple["torch.Tensor", ...]:
        assert torch is not None
        return (
            torch.zeros(2, 4, 3, 32, 32),
            torch.zeros(2, 4, 10),
            torch.zeros(2, 9),
        )

    def test_cosine_schedule_is_bounded_and_increasing(self) -> None:
        assert torch is not None
        betas = cosine_beta_schedule(100)
        self.assertEqual(tuple(betas.shape), (100,))
        self.assertTrue(bool(torch.all((betas > 0) & (betas < 1))))
        self.assertTrue(bool(torch.all(betas[1:] >= betas[:-1])))

    def test_noise_prediction_shape(self) -> None:
        assert torch is not None
        model = self.build_model().eval()
        images, states, condition = self.inputs()
        noisy = torch.zeros(2, 16, 2)
        timesteps = torch.tensor([0, 99])
        with torch.inference_mode():
            prediction = model(noisy, timesteps, images, states, condition)
        self.assertEqual(tuple(prediction.shape), (2, 16, 2))
        self.assertTrue(bool(torch.all(torch.isfinite(prediction))))

    def test_ddim_sampling_is_deterministic_and_bounded(self) -> None:
        assert torch is not None
        torch.manual_seed(7)
        model = self.build_model().eval()
        schedule = DiffusionSchedule(20)
        images, states, condition = self.inputs()
        initial = torch.zeros(2, 16, 2)
        first = schedule.ddim_sample(
            model,
            images,
            states,
            condition,
            inference_steps=5,
            eta=0.0,
            initial_noise=initial,
        )
        second = schedule.ddim_sample(
            model,
            images,
            states,
            condition,
            inference_steps=5,
            eta=0.0,
            initial_noise=initial,
        )
        self.assertTrue(torch.allclose(first, second))
        self.assertTrue(bool(torch.all(torch.abs(first) <= 1.0)))

    def test_weighted_noise_loss_normalizes_by_weight_sum(self) -> None:
        assert torch is not None
        prediction = torch.zeros(2, 2, 2)
        target = torch.stack((torch.ones(2, 2), torch.full((2, 2), 2.0)))
        weights = torch.tensor([1.0, 3.0])
        loss = weighted_noise_mse(prediction, target, weights)
        self.assertAlmostEqual(float(loss), 3.25)

    def test_clean_action_reconstruction_is_bounded(self) -> None:
        assert torch is not None
        schedule = DiffusionSchedule(20)
        clean = torch.tensor([[[0.2, -0.4], [0.3, 0.5]]])
        noise = torch.tensor([[[0.1, -0.2], [0.4, 0.2]]])
        timesteps = torch.tensor([7])
        noisy = schedule.add_noise(clean, noise, timesteps)
        reconstructed = schedule.predict_clean_actions(noisy, noise, timesteps)
        self.assertTrue(torch.allclose(reconstructed, clean, atol=1e-5))
        self.assertTrue(bool(torch.all(torch.abs(reconstructed) <= 1.0)))

    def test_reconstruction_and_derivative_losses(self) -> None:
        assert torch is not None
        target = torch.zeros(1, 3, 2)
        prediction = torch.tensor([[[0.0, 0.0], [0.2, 0.4], [0.4, 0.8]]])
        weights = torch.ones(1)
        reconstruction = weighted_action_reconstruction_loss(
            prediction, target, weights, longitudinal_weight=2.0
        )
        derivative = weighted_temporal_derivative_loss(
            prediction, target, weights, longitudinal_weight=2.0
        )
        self.assertGreater(float(reconstruction), 0)
        self.assertGreater(float(derivative), 0)

    def test_frozen_encoder_stays_in_evaluation_mode(self) -> None:
        model = self.build_model()
        model.set_encoder_trainable(False)
        model.train()
        self.assertFalse(model.encoder.training)
        self.assertTrue(model.denoiser.training)


if __name__ == "__main__":
    unittest.main()
