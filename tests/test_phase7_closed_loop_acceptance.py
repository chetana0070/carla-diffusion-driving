import sys
import unittest
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_phase7_factorized_closed_loop import evaluate_acceptance


class Phase7ClosedLoopAcceptanceTests(unittest.TestCase):
    digest = "a" * 64
    seeds: ClassVar[list[int]] = [20260901, 20260902, 20260903]

    def passing_report(self) -> dict:
        episodes = [
            {
                "seed": seed,
                "ticks_completed": 300,
                "terminated_reason": "tick_limit",
                "collision_events": 0,
                "lane_invasion_events": 0,
                "red_light_violations": 0,
                "stationary_fraction": 0.2,
                "max_speed_mps": 4.5,
                "max_abs_lane_offset_m": 0.5,
            }
            for seed in self.seeds
        ]
        return {
            "status": "passed",
            "model_type": "factorized_temporal_policy_phase6_safety_arbitration",
            "checkpoint_sha256": self.digest,
            "episodes": episodes,
            "aggregate": {
                "mean_route_progress_fraction": 0.12,
                "mean_distance_traveled_m": 90.0,
            },
        }

    def telemetry(self) -> dict[int, list[dict]]:
        return {
            seed: [
                {
                    "seed": seed,
                    "tick": 0,
                    "policy_pipeline_latency_ms": 120.0,
                    "factorized_policy_replanned": True,
                    "safety_active": False,
                    "recovery_active": False,
                    "liveness_active": False,
                },
                {
                    "seed": seed,
                    "tick": 1,
                    "policy_pipeline_latency_ms": 20.0,
                    "factorized_policy_replanned": False,
                    "safety_active": False,
                    "recovery_active": False,
                    "liveness_active": False,
                },
            ]
            for seed in self.seeds
        }

    def evaluate(self, report: dict, telemetry: dict[int, list[dict]]) -> dict:
        return evaluate_acceptance(
            report,
            telemetry,
            expected_seeds=self.seeds,
            expected_checkpoint_sha256=self.digest,
            minimum_ticks=300,
            minimum_mean_route_progress_fraction=0.09,
            minimum_mean_distance_m=80.0,
            maximum_collisions=0,
            maximum_lane_invasions=0,
            maximum_red_light_violations=0,
            maximum_no_progress_terminations=0,
            maximum_stationary_fraction=0.6,
            maximum_speed_mps=5.0,
            maximum_lane_offset_m=1.5,
            cold_start_ticks=1,
            maximum_cold_start_latency_ms=200.0,
            maximum_steady_state_latency_ms=100.0,
        )

    def test_passing_three_seed_contract(self) -> None:
        result = self.evaluate(self.passing_report(), self.telemetry())
        self.assertTrue(result["gate_passed"])
        self.assertEqual(result["metrics"]["factorized_replan_ticks"], 3)

    def test_safety_and_liveness_failures_are_named(self) -> None:
        report = self.passing_report()
        report["episodes"][1]["collision_events"] = 1
        report["episodes"][2]["terminated_reason"] = "no_progress"
        telemetry = self.telemetry()
        telemetry[self.seeds[2]][-1]["liveness_active"] = True
        result = self.evaluate(report, telemetry)
        self.assertFalse(result["gate_passed"])
        self.assertTrue(
            {"collision_budget", "liveness_termination_budget", "liveness_disengaged"}
            <= set(result["failures"])
        )

    def test_checkpoint_mismatch_blocks_promotion(self) -> None:
        report = self.passing_report()
        report["checkpoint_sha256"] = "b" * 64
        result = self.evaluate(report, self.telemetry())
        self.assertFalse(result["checks"]["checkpoint_identity"])


if __name__ == "__main__":
    unittest.main()
