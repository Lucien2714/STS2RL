"""Shared dataclasses for training and evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Transition:
    """One environment transition observed by a trainable agent."""

    state: dict[str, Any]
    action: dict[str, Any]
    reward: float
    next_state: dict[str, Any]
    done: bool
    reward_details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingInfo:
    """Structured summary returned by an agent update."""

    updated: bool
    loss: float | None = None
    reward: float = 0.0
    action_type: str | None = None
    epsilon: float | None = None
    replay_size: int | None = None
    learn_steps: int | None = None
    won_battle: bool = False
    lost_battle: bool = False
    reward_details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    """Result of one environment step."""

    observation: Any
    reward: float
    done: bool
    info: dict[str, Any]


@dataclass(frozen=True)
class EpisodeResult:
    """Summary of one completed episode."""

    reward: float
    steps: int
    final_state_type: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
