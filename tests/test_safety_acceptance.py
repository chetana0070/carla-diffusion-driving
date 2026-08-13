import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_phase6_safety_smoke import evaluate_acceptance


class SafetyAcceptanceTests(unittest.TestCase):
    def test_preemptive_only_run_is_safely_disengaged(self) -> None:
        report = {
            "status": "passed",
            "episodes": [
                {
                    "ticks_completed": 300,
                    "collision_events": 0,
                    "lane_invasion_events": 0,
                }
            ],
        }
        telemetry = [
            {
                "speed_mps": 4.7,
                "lane_offset_m": 0.25,
                "safety_active": True,
                "preemptive_safety_active": True,
                "recovery_active": False,
                "recovery_activated": False,
                "recovery_exit_reason": None,
                "tick": 0,
                "policy_pipeline_latency_ms": 40.0,
            },
            {
                "speed_mps": 3.0,
                "lane_offset_m": 0.1,
                "safety_active": False,
                "preemptive_safety_active": False,
                "recovery_active": False,
                "recovery_activated": False,
                "recovery_exit_reason": None,
                "tick": 1,
                "policy_pipeline_latency_ms": 30.0,
            },
        ]
        result = evaluate_acceptance(
            report,
            telemetry,
            minimum_ticks=300,
            maximum_recovery_speed_mps=5.0,
            maximum_lane_offset_m=1.5,
            maximum_lane_invasions=1,
            maximum_latency_ms=100.0,
        )
        self.assertTrue(result["checks"]["recovery_disengaged"])
        self.assertEqual(result["metrics"]["recovery_activation_events"], 0)

    def test_recovery_active_at_episode_end_fails_disengagement(self) -> None:
        report = {
            "status": "passed",
            "episodes": [
                {
                    "ticks_completed": 300,
                    "collision_events": 0,
                    "lane_invasion_events": 0,
                }
            ],
        }
        telemetry = [
            {
                "speed_mps": 4.0,
                "lane_offset_m": 0.5,
                "safety_active": True,
                "recovery_active": True,
                "recovery_activated": True,
                "recovery_exit_reason": None,
                "tick": 0,
                "policy_pipeline_latency_ms": 40.0,
            }
        ]
        result = evaluate_acceptance(
            report,
            telemetry,
            minimum_ticks=300,
            maximum_recovery_speed_mps=5.0,
            maximum_lane_offset_m=1.5,
            maximum_lane_invasions=1,
            maximum_latency_ms=100.0,
        )
        self.assertFalse(result["checks"]["recovery_disengaged"])

    def test_liveness_guard_active_at_episode_end_fails_disengagement(self) -> None:
        report = {
            "status": "passed",
            "episodes": [
                {
                    "ticks_completed": 300,
                    "collision_events": 0,
                    "lane_invasion_events": 0,
                }
            ],
        }
        telemetry = [
            {
                "speed_mps": 0.0,
                "lane_offset_m": 0.1,
                "safety_active": False,
                "recovery_active": False,
                "recovery_activated": False,
                "recovery_exit_reason": None,
                "liveness_active": True,
                "liveness_activated": True,
                "liveness_exit_reason": None,
                "tick": 0,
                "policy_pipeline_latency_ms": 40.0,
            }
        ]
        result = evaluate_acceptance(
            report,
            telemetry,
            minimum_ticks=300,
            maximum_recovery_speed_mps=5.0,
            maximum_lane_offset_m=1.5,
            maximum_lane_invasions=1,
            maximum_latency_ms=100.0,
        )
        self.assertFalse(result["checks"]["liveness_disengaged"])

    def test_passing_controlled_smoke(self) -> None:
        report = {
            "status": "passed",
            "episodes": [
                {
                    "ticks_completed": 300,
                    "collision_events": 0,
                    "lane_invasion_events": 1,
                    "policy_pipeline_latency": {"max_ms": 40.0},
                }
            ],
        }
        telemetry = [
            {
                "speed_mps": 4.8,
                "lane_offset_m": 0.8,
                "safety_active": True,
                "recovery_exit_reason": "center_crossed",
                "tick": 0,
                "policy_pipeline_latency_ms": 150.0,
            },
            {
                "speed_mps": 4.0,
                "lane_offset_m": 0.2,
                "safety_active": False,
                "recovery_exit_reason": None,
                "tick": 1,
                "policy_pipeline_latency_ms": 40.0,
            }
        ]
        result = evaluate_acceptance(
            report,
            telemetry,
            minimum_ticks=300,
            maximum_recovery_speed_mps=5.0,
            maximum_lane_offset_m=1.5,
            maximum_lane_invasions=1,
            maximum_latency_ms=100.0,
        )
        self.assertTrue(result["gate_passed"])

    def test_unsafe_smoke_fails_named_gates(self) -> None:
        report = {
            "status": "passed",
            "episodes": [
                {
                    "ticks_completed": 57,
                    "collision_events": 1,
                    "lane_invasion_events": 7,
                    "policy_pipeline_latency": {"max_ms": 39.0},
                }
            ],
        }
        telemetry = [
            {
                "speed_mps": 12.2,
                "lane_offset_m": 10.5,
                "safety_active": True,
                "recovery_exit_reason": None,
                "tick": 0,
                "policy_pipeline_latency_ms": 150.0,
            },
            {
                "speed_mps": 4.0,
                "lane_offset_m": 1.6,
                "safety_active": False,
                "recovery_exit_reason": None,
                "tick": 1,
                "policy_pipeline_latency_ms": 120.0,
            }
        ]
        result = evaluate_acceptance(
            report,
            telemetry,
            minimum_ticks=300,
            maximum_recovery_speed_mps=5.0,
            maximum_lane_offset_m=1.5,
            maximum_lane_invasions=1,
            maximum_latency_ms=100.0,
        )
        self.assertFalse(result["gate_passed"])
        self.assertEqual(
            set(result["failures"]),
            {
                "tick_completion",
                "collision_free",
                "recovery_speed",
                "lane_offset",
                "lane_invasions",
                "recovery_disengaged",
                "steady_state_latency",
            },
        )

    def test_cold_start_latency_has_separate_disclosed_gate(self) -> None:
        report = {
            "status": "passed",
            "episodes": [
                {
                    "ticks_completed": 300,
                    "collision_events": 0,
                    "lane_invasion_events": 0,
                    "policy_pipeline_latency": {"max_ms": 250.0},
                }
            ],
        }
        telemetry = [
            {
                "speed_mps": 1.0,
                "lane_offset_m": 0.5,
                "safety_active": True,
                "recovery_exit_reason": "center_crossed",
                "tick": 0,
                "policy_pipeline_latency_ms": 250.0,
            },
            {
                "speed_mps": 1.0,
                "lane_offset_m": 0.1,
                "safety_active": False,
                "recovery_exit_reason": None,
                "tick": 1,
                "policy_pipeline_latency_ms": 40.0,
            },
        ]
        result = evaluate_acceptance(
            report,
            telemetry,
            minimum_ticks=300,
            maximum_recovery_speed_mps=5.0,
            maximum_lane_offset_m=1.5,
            maximum_lane_invasions=1,
            maximum_latency_ms=100.0,
        )
        self.assertFalse(result["gate_passed"])
        self.assertEqual(result["failures"], ["cold_start_latency"])

    def test_speed_gate_includes_ticks_without_lane_safety(self) -> None:
        report = {
            "status": "passed",
            "episodes": [
                {
                    "ticks_completed": 300,
                    "collision_events": 0,
                    "lane_invasion_events": 0,
                }
            ],
        }
        telemetry = [
            {
                "speed_mps": 5.2,
                "lane_offset_m": 0.1,
                "safety_active": False,
                "recovery_exit_reason": "center_crossed",
                "tick": 0,
                "policy_pipeline_latency_ms": 40.0,
            },
            {
                "speed_mps": 4.0,
                "lane_offset_m": 0.1,
                "safety_active": False,
                "recovery_exit_reason": None,
                "tick": 1,
                "policy_pipeline_latency_ms": 40.0,
            },
        ]
        result = evaluate_acceptance(
            report,
            telemetry,
            minimum_ticks=300,
            maximum_recovery_speed_mps=5.0,
            maximum_lane_offset_m=1.5,
            maximum_lane_invasions=1,
            maximum_latency_ms=100.0,
        )
        self.assertFalse(result["checks"]["recovery_speed"])
        self.assertEqual(result["metrics"]["maximum_rollout_speed_mps"], 5.2)


if __name__ == "__main__":
    unittest.main()
