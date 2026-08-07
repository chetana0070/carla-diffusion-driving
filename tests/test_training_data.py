import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carla_diffusion.schema import DrivingSample
from carla_diffusion.splits import split_pilot_routes
from carla_diffusion.training_data import (
    WeightConfig,
    build_temporal_windows,
    categorical_condition,
    transform_state,
)


def sample(index: int, *, stopped: bool = False) -> DrivingSample:
    return DrivingSample(
        schema_version="1.0.0",
        episode_id="episode-1",
        route_id="route-1",
        frame_id=index,
        timestamp_seconds=index / 10,
        rgb_front_path=f"episode-1/images/{index:08d}.jpg",
        state=(
            0.0 if stopped else 5.0,
            20.0,
            8.0,
            0.0,
            0.0,
            -1.0,
            4.0,
            -1.0,
        ),
        traffic_light_state="red" if stopped else "none",
        route_command="follow_lane",
        expert_action=(0.0, -1.0 if stopped else 0.2),
        collision=False,
        lane_invasion=False,
        red_light_violation=False,
        intervention=False,
    )


class TrainingDataTests(unittest.TestCase):
    def test_state_transform_clips_and_adds_masks(self) -> None:
        transformed = transform_state(sample(0).state, acceleration_clip=12.0)
        self.assertEqual(len(transformed), 10)
        self.assertEqual(transformed[1], 12.0)
        self.assertEqual(transformed[5:8], (0.0, 0.0, 0.0))
        self.assertEqual(transformed[8:], (0.0, 0.0))

    def test_categorical_condition_is_one_hot(self) -> None:
        condition = categorical_condition("left", "green")
        self.assertEqual(len(condition), 9)
        self.assertEqual(sum(condition[:4]), 1.0)
        self.assertEqual(sum(condition[4:]), 1.0)

    def test_temporal_window_count_and_hold_weight(self) -> None:
        samples = [sample(index, stopped=index == 3) for index in range(20)]
        windows = build_temporal_windows(
            samples,
            split="train",
            history=4,
            horizon=16,
            acceleration_clip=12.0,
            weight_config=WeightConfig(),
        )
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].sample_weight, 0.25)
        self.assertEqual(windows[0].categories, ("stationary_brake_hold",))
        self.assertEqual(len(windows[0].state_history), 4)
        self.assertEqual(len(windows[0].action_target), 16)

    def test_pilot_split_is_deterministic_and_leakage_free(self) -> None:
        route_ids = [f"route-{index}" for index in range(10)]
        first = split_pilot_routes(route_ids, seed=7)
        second = split_pilot_routes(route_ids, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first["train"]), 8)
        self.assertEqual(len(first["validation"]), 1)
        self.assertEqual(len(first["test"]), 1)
        self.assertEqual(set(first["train"]) & set(first["validation"]), set())
        self.assertEqual(set().union(*map(set, first.values())), set(route_ids))


if __name__ == "__main__":
    unittest.main()
