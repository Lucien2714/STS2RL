"""Abstract agent interfaces shared by screen-specific policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from sts2rl.metrics.types import TrainingInfo, Transition


class ScreenAgent(ABC):
    """Base interface for an agent that controls one class of game screens."""

    @abstractmethod
    def choose_action(self, state: dict) -> dict:
        """Choose a game action for the current raw screen state."""


class TrainableScreenAgent(ScreenAgent):
    """Interface for trainable screen agents (battle and non-battle screens).

    Adds a training-aware ``choose_action`` plus optional learning/persistence
    hooks on top of :class:`ScreenAgent`. Nothing here is battle-specific; the
    battle agents were simply the first implementations.
    """

    @abstractmethod
    def choose_action(self, state: dict, *, training: bool = False) -> dict:
        """Choose an action. Training mode may enable exploration."""

    def observe(self, transition: Transition) -> TrainingInfo | None:
        """Learn from a transition. Rule-based agents can ignore experience."""
        return None

    def save(self, path: Path) -> None:
        """Persist trainable state."""
        raise NotImplementedError(f"{type(self).__name__} does not support save()")

    def load(self, path: Path) -> None:
        """Load trainable state."""
        raise NotImplementedError(f"{type(self).__name__} does not support load()")


# Backward-compatible alias: this interface began life as the battle-agent base.
BattleAgent = TrainableScreenAgent


class MapAgent(ScreenAgent):
    """Interface for map-routing agents."""


class EventAgent(ScreenAgent):
    """Interface for event-choice agents."""
