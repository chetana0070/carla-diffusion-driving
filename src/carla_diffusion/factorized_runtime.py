"""Dependency-light action-chunk execution for factorized deployment."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def deployment_noise_seed(checkpoint_config: dict[str, Any]) -> int:
    """Resolve new deployment metadata without invalidating released checkpoints."""
    factorized = checkpoint_config.get("factorized_policy")
    if not isinstance(factorized, dict):
        raise TypeError("factorized policy configuration must be a mapping")
    if "selection_noise_seed" not in factorized:
        raise ValueError("checkpoint lacks the frozen factorized noise seed")
    deployment = checkpoint_config.get("phase7_closed_loop_evaluation", {})
    if not isinstance(deployment, dict):
        raise TypeError("Phase 7 deployment configuration must be a mapping")
    seed = deployment.get(
        "deployment_noise_seed", factorized["selection_noise_seed"]
    )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("deployment noise seed must be an integer")
    if seed < 0:
        raise ValueError("deployment noise seed cannot be negative")
    return seed


class ActionChunkExecutor:
    """Execute a bounded prefix of each predicted action chunk."""

    def __init__(self, execute_steps: int) -> None:
        if execute_steps < 1:
            raise ValueError("execute_steps must be positive")
        self.execute_steps = execute_steps
        self._actions: list[tuple[float, float]] = []
        self._index = 0

    @property
    def needs_replan(self) -> bool:
        return self._index >= len(self._actions)

    @property
    def step_index(self) -> int:
        return self._index

    @property
    def remaining_steps(self) -> int:
        return max(0, len(self._actions) - self._index)

    def reset(self) -> None:
        self._actions = []
        self._index = 0

    def set_plan(self, actions: Sequence[Sequence[float]]) -> None:
        if len(actions) < self.execute_steps:
            raise ValueError("predicted chunk is shorter than execute_steps")
        validated: list[tuple[float, float]] = []
        for action in actions[: self.execute_steps]:
            if len(action) != 2:
                raise ValueError("each action must contain steering and longitudinal values")
            steering, longitudinal = (float(action[0]), float(action[1]))
            if not all(math.isfinite(value) for value in (steering, longitudinal)):
                raise ValueError("action chunk contains a non-finite value")
            if not all(-1.0 <= value <= 1.0 for value in (steering, longitudinal)):
                raise ValueError("action chunk values must be in [-1, 1]")
            validated.append((steering, longitudinal))
        self._actions = validated
        self._index = 0

    def pop(self) -> tuple[float, float]:
        if self.needs_replan:
            raise RuntimeError("action chunk is exhausted")
        action = self._actions[self._index]
        self._index += 1
        return action
