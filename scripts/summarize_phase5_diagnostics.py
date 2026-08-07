#!/usr/bin/env python3
"""Compare learned-policy and expert-oracle closed-loop reports."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from carla_diffusion.closed_loop import evaluate_expert_oracle_gate
from carla_diffusion.config import load_and_validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-report",
        type=Path,
        default=PROJECT_ROOT / "artifacts/evaluations/phase5_policy_diagnostic_v090.json",
    )
    parser.add_argument(
        "--expert-report",
        type=Path,
        default=PROJECT_ROOT / "artifacts/evaluations/phase5_expert_oracle_v090.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/evaluations/phase5_diagnostic_summary_v092.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    config = load_and_validate_config(PROJECT_ROOT / "configs/project.json")
    policy = load_json(args.policy_report.resolve())
    expert = load_json(args.expert_report.resolve())
    policy_seeds = [int(episode["seed"]) for episode in policy["episodes"]]
    expert_seeds = [int(episode["seed"]) for episode in expert["episodes"]]
    if policy_seeds != expert_seeds:
        raise ValueError("policy and expert reports do not use identical seeds")
    if policy["map"] != expert["map"]:
        raise ValueError("policy and expert reports do not use the same map")
    closed_loop = config["closed_loop_evaluation"]
    policy_terminations = Counter(
        str(episode["terminated_reason"]) for episode in policy["episodes"]
    )
    expert_terminations = Counter(
        str(episode["terminated_reason"]) for episode in expert["episodes"]
    )
    expert_gate = evaluate_expert_oracle_gate(
        expert["aggregate"], expert_terminations, closed_loop
    )
    oracle_passed = bool(expert_gate["passed"])
    decision = (
        "policy_failure_confirmed_collect_corrections"
        if oracle_passed
        else "route_or_oracle_setup_requires_correction"
    )
    summary = {
        "status": "passed",
        "decision": decision,
        "comparison_contract": {
            "map": policy["map"],
            "seeds": policy_seeds,
            "episodes": len(policy_seeds),
        },
        "protocol_amendment": {
            "disclosed": True,
            "reason": (
                "Protocol v1 used a single 25% mean route-progress threshold. "
                "The safe expert run exposed sensitivity to traffic-light dwell time, "
                "so v2 uses progress, physical distance, safety, and liveness gates."
            ),
        },
        "expert_oracle_gate": expert_gate,
        "policy": {
            "model_type": policy["model_type"],
            "aggregate": policy["aggregate"],
            "termination_counts": dict(sorted(policy_terminations.items())),
        },
        "expert": {
            "model_type": expert["model_type"],
            "aggregate": expert["aggregate"],
            "termination_counts": dict(sorted(expert_terminations.items())),
        },
    }
    atomic_json(args.output.resolve(), summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
