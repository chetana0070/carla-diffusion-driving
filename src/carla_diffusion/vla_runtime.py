"""Runtime contract for slow VLA planning and fast deterministic execution."""

from __future__ import annotations

import math
from collections.abc import Sequence


class HierarchicalPlanExecutor:
    """Execute a bounded prefix of each VLA action chunk at the control rate."""

    def __init__(self, *, action_horizon: int, execute_steps: int) -> None:
        if action_horizon < 1 or not 1 <= execute_steps <= action_horizon:
            raise ValueError("invalid hierarchical execution dimensions")
        self.action_horizon = action_horizon
        self.execute_steps = execute_steps
        self._plan: tuple[tuple[float, float], ...] = ()
        self._index = 0

    @property
    def requires_plan(self) -> bool:
        return not self._plan or self._index >= self.execute_steps

    def install_plan(self, actions: Sequence[Sequence[float]]) -> None:
        if len(actions) != self.action_horizon:
            raise ValueError("VLA plan does not match the frozen horizon")
        bounded = []
        for action in actions:
            if len(action) != 2:
                raise ValueError("each VLA action must contain steering and longitudinal")
            steering, longitudinal = (float(value) for value in action)
            if not math.isfinite(steering) or not math.isfinite(longitudinal):
                raise ValueError("VLA actions must be finite")
            bounded.append(
                (
                    max(-1.0, min(1.0, steering)),
                    max(-1.0, min(1.0, longitudinal)),
                )
            )
        self._plan = tuple(bounded)
        self._index = 0

    def next_action(self) -> tuple[float, float]:
        if self.requires_plan:
            raise RuntimeError("a new VLA plan is required")
        action = self._plan[self._index]
        self._index += 1
        return action

    def reset(self) -> None:
        self._plan = ()
        self._index = 0
