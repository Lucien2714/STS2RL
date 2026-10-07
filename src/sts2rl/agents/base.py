"""Minimal agent contract shared by learning and evaluation code."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sts2rl.actions import GameAction
from sts2rl.agents.action_space import NoLegalActionsError
from sts2rl.env.types import GameObservation


def without_excluded(
    candidates: Sequence[GameAction],
    exclude: Sequence[GameAction],
) -> tuple[GameAction, ...]:
    """Return the candidates whose full parameters match no excluded action.

    The filter is applied by the agent at choice time and never by
    ``LegalActionProvider``: a candidate set must stay a pure function of the
    state, because ``BCDataset`` re-derives it to check that ``expert_index``
    still points at the same action.

    Raises ``NoLegalActionsError`` when nothing is left, so the runner can fall
    back to choosing without the exclusion.
    """
    excluded = [action.to_dict() for action in exclude]
    kept = tuple(c for c in candidates if c.to_dict() not in excluded)
    if not kept:
        raise NoLegalActionsError("every candidate on this screen is excluded")
    return kept


@dataclass(frozen=True)
class Transition:
    """One observed environment transition used by an agent."""

    state: GameObservation
    action: GameAction
    reward: float
    next_state: GameObservation
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """Small interface required by the episode runner."""

    def reset(self, initial_state: GameObservation) -> None:
        """Reset episode-local agent state."""
        return None

    @abstractmethod
    def choose_action(
        self, state: GameObservation, exclude: Sequence[GameAction] = ()
    ) -> GameAction:
        """Choose one legal action for the current full observation.

        ``exclude`` names actions, by their full parameters, that the game
        refused twice in a row on this same unchanged screen
        (``EpisodeRunner``).  They are not
        chosen again; ``without_excluded`` raises ``NoLegalActionsError`` when
        they cover every candidate.  The runner passes it only when it is not
        empty, so an agent that predates it still works without it.
        """

    def observe(self, transition: Transition) -> None:
        """Observe a completed transition; evaluation agents may ignore it."""
        return None

    def discard_decision(self) -> None:
        """Forget the last chosen action instead of observing a result for it.

        The runner calls this when the game refused an action and the screen
        did not move.  That is not a transition worth learning from -- the
        action was legal, the screen simply was not ready -- and recording it
        would teach that resting at a rest site does nothing.
        """
        return None

    def finish_episode(
        self, final_state: GameObservation, truncated: bool
    ) -> None:
        """Finish pending learning work at an episode boundary."""
        return None
