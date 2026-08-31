"""Minimal agent contract shared by learning and evaluation code."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from sts2rl.actions import GameAction
from sts2rl.env.types import RawState


@dataclass(frozen=True)
class Transition:
    """One observed environment transition used by an agent."""

    state: RawState
    action: GameAction
    reward: float
    next_state: RawState
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """Small interface required by the episode runner."""

    def reset(self, initial_state: RawState) -> None:
        """Reset episode-local agent state."""
        return None

    @abstractmethod
    def choose_action(self, state: RawState) -> GameAction:
        """Choose one legal action for the current raw state."""

    def observe(self, transition: Transition) -> None:
        """Observe a completed transition; evaluation agents may ignore it."""
        return None

    def finish_episode(self, final_state: RawState, truncated: bool) -> None:
        """Finish pending learning work at an episode boundary."""
        return None
