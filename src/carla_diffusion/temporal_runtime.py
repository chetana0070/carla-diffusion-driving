"""Small stateful contracts used by temporal closed-loop inference."""

from __future__ import annotations

from collections import deque
from typing import Generic, TypeVar

ImageT = TypeVar("ImageT")
StateT = TypeVar("StateT")


class TemporalHistoryBuffer(Generic[ImageT, StateT]):
    """Keep synchronized image/state history with deterministic cold start."""

    def __init__(self, history_frames: int) -> None:
        if history_frames <= 1:
            raise ValueError("temporal history must contain at least two frames")
        self.history_frames = history_frames
        self._images: deque[ImageT] = deque(maxlen=history_frames)
        self._states: deque[StateT] = deque(maxlen=history_frames)

    def reset(self) -> None:
        self._images.clear()
        self._states.clear()

    def append(self, image: ImageT, state: StateT) -> None:
        if not self._images:
            self._images.extend([image] * self.history_frames)
            self._states.extend([state] * self.history_frames)
            return
        self._images.append(image)
        self._states.append(state)

    def sequences(self) -> tuple[tuple[ImageT, ...], tuple[StateT, ...]]:
        if len(self._images) != self.history_frames or len(self._states) != self.history_frames:
            raise RuntimeError("temporal history is not initialized")
        return tuple(self._images), tuple(self._states)

    def __len__(self) -> int:
        return len(self._images)
