"""Public value types for the raw environment boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias


RawState: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class GameObservation:
    """Agent input for one decision point.

    ``raw_state`` is the screen the agent is acting on.  ``player_detail`` is
    the ``/player`` response, the only source of the run-level master deck; it
    is None on a terminal state, before a run starts, or on a mod build that
    does not serve the endpoint.
    """

    raw_state: RawState
    player_detail: RawState | None = None


@dataclass(frozen=True)
class EnvStep:
    """Result of executing one action against STS2MCP."""

    raw_state: RawState
    done: bool
    info: dict[str, Any] = field(default_factory=dict)
