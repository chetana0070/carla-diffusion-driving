"""Deterministic language supervision for the hierarchical Phase 8 planner."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .training_data import ROUTE_COMMAND_ORDER, TRAFFIC_LIGHT_ORDER

SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
INSTRUCTION_TEMPLATES = {
    "follow_lane": "follow the current lane",
    "left": "turn left at the next junction",
    "right": "turn right at the next junction",
    "straight": "continue straight through the next junction",
}
LIGHT_SUFFIXES = {
    "none": "when the road is clear",
    "red": "and stop for the red light",
    "yellow": "and prepare to stop for the yellow light",
    "green": "while respecting the green light",
    "unknown": "while checking the traffic signal",
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+|<[^>]+>")


@dataclass(frozen=True)
class VLARecord:
    """One leakage-preserving language-conditioned planning example."""

    split: str
    episode_id: str
    route_id: str
    anchor_frame_id: int
    image_paths: tuple[str, ...]
    state_history: tuple[tuple[float, ...], ...]
    instruction: str
    instruction_token_ids: tuple[int, ...]
    instruction_attention_mask: tuple[int, ...]
    route_command: str
    traffic_light_state: str
    action_chunk_target: tuple[tuple[float, float], ...]
    sample_weight: float
    categories: tuple[str, ...]
    source_dataset: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.source_dataset is None:
            payload.pop("source_dataset")
        return payload


def decode_condition(condition: Sequence[float]) -> tuple[str, str]:
    expected = len(ROUTE_COMMAND_ORDER) + len(TRAFFIC_LIGHT_ORDER)
    if len(condition) != expected:
        raise ValueError(f"condition must contain {expected} values")
    route_values = [float(value) for value in condition[: len(ROUTE_COMMAND_ORDER)]]
    light_values = [float(value) for value in condition[len(ROUTE_COMMAND_ORDER) :]]
    if sum(value > 0.5 for value in route_values) != 1:
        raise ValueError("route condition must be one-hot")
    if sum(value > 0.5 for value in light_values) != 1:
        raise ValueError("traffic-light condition must be one-hot")
    return (
        ROUTE_COMMAND_ORDER[route_values.index(max(route_values))],
        TRAFFIC_LIGHT_ORDER[light_values.index(max(light_values))],
    )


def instruction_for(route_command: str, traffic_light_state: str) -> str:
    try:
        route_text = INSTRUCTION_TEMPLATES[route_command]
        light_text = LIGHT_SUFFIXES[traffic_light_state]
    except KeyError as error:
        raise ValueError(f"unsupported language condition: {error.args[0]}") from error
    return f"Drive safely: {route_text} {light_text}."


def build_vocabulary() -> dict[str, int]:
    words = {
        token
        for route in ROUTE_COMMAND_ORDER
        for light in TRAFFIC_LIGHT_ORDER
        for token in tokenize_text(instruction_for(route, light))
    }
    ordered = (*SPECIAL_TOKENS, *sorted(words - set(SPECIAL_TOKENS)))
    return {token: index for index, token in enumerate(ordered)}


def tokenize_text(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_PATTERN.findall(text.lower()))


def encode_instruction(
    instruction: str,
    vocabulary: Mapping[str, int],
    max_tokens: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if max_tokens < 4:
        raise ValueError("max_tokens must be at least four")
    required = set(SPECIAL_TOKENS)
    if not required.issubset(vocabulary):
        raise ValueError("vocabulary is missing required special tokens")
    tokens = ("<bos>", *tokenize_text(instruction), "<eos>")
    if len(tokens) > max_tokens:
        tokens = (*tokens[: max_tokens - 1], "<eos>")
    ids = [vocabulary.get(token, vocabulary["<unk>"]) for token in tokens]
    mask = [1] * len(ids)
    padding = max_tokens - len(ids)
    ids.extend([vocabulary["<pad>"]] * padding)
    mask.extend([0] * padding)
    return tuple(ids), tuple(mask)


def build_vla_record(
    window: Mapping[str, Any],
    *,
    vocabulary: Mapping[str, int],
    planner_horizon: int,
    max_instruction_tokens: int,
) -> VLARecord:
    if planner_horizon < 1:
        raise ValueError("planner_horizon must be positive")
    route_command, traffic_light_state = decode_condition(window["condition"])
    instruction = instruction_for(route_command, traffic_light_state)
    token_ids, attention_mask = encode_instruction(
        instruction,
        vocabulary,
        max_instruction_tokens,
    )
    actions = tuple(
        (float(action[0]), float(action[1]))
        for action in window["action_target"][:planner_horizon]
    )
    if len(actions) != planner_horizon:
        raise ValueError("window does not contain the requested planner horizon")
    source_dataset = window.get("source_dataset")
    return VLARecord(
        split=str(window["split"]),
        episode_id=str(window["episode_id"]),
        route_id=str(window["route_id"]),
        anchor_frame_id=int(window["anchor_frame_id"]),
        image_paths=tuple(str(path) for path in window["image_paths"]),
        state_history=tuple(
            tuple(float(value) for value in state) for state in window["state_history"]
        ),
        instruction=instruction,
        instruction_token_ids=token_ids,
        instruction_attention_mask=attention_mask,
        route_command=route_command,
        traffic_light_state=traffic_light_state,
        action_chunk_target=actions,
        sample_weight=float(window["sample_weight"]),
        categories=tuple(str(category) for category in window.get("categories", [])),
        source_dataset=None if source_dataset is None else str(source_dataset),
    )
