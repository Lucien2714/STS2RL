"""Public value types for the raw environment boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias


RawState: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class GameObservation:
    """Agent input for one decision point.

    The official API exposes everything the agent sees under one state, so this
    currently wraps exactly that; it stays a type of its own so enrichment can
    be added without changing every signature between the env and the agent.
    """

    raw_state: RawState


@dataclass(frozen=True)
class EnvStep:
    """Result of executing one action against STS2MCP."""

    raw_state: RawState
    done: bool
    info: dict[str, Any] = field(default_factory=dict)
