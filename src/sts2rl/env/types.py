"""Public value types for the raw environment boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias


RawState: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class GameObservation:
    """Agent input combining one raw state with optional full player detail."""

    raw_state: RawState
    player_detail: RawState | None = None


@dataclass(frozen=True)
class EnvStep:
    """Result of executing one action against STS2MCP."""

    raw_state: RawState
    done: bool
    info: dict[str, Any] = field(default_factory=dict)
