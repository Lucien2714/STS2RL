"""Shared plumbing for screen action spaces.

An action space enumerates the currently legal candidate actions for one screen
(each a dict with ``action`` and a stable ``action_key``) and keys arbitrary game
actions. It carries the game-rule half of what used to live in the screen
encoders; representation (feature vectors) stays in ``encoders/``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


def parse_int(value: object, default: int = 0) -> int:
    """Parse integer-like MCP fields, falling back on missing/invalid values."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class ActionSpace(ABC):
    """Base for per-screen legal-action enumeration."""

    @abstractmethod
    def candidates(self, raw_state: dict) -> list[dict]:
        """Enumerate currently legal candidates (each a dict with 'action' and 'action_key')."""

    @abstractmethod
    def action_key(self, action: dict, raw_state: dict | None = None) -> str:
        """Return a stable string key identifying an action."""

    def fallback(self, raw_state: dict) -> dict:
        """Safe default when no candidates exist; the orchestrator prefers the
        screen's rule-based policy for the real fallback."""
        return {"type": "proceed"}

    def valid_action_mask(self, raw_state: dict) -> list[bool]:
        """Return an all-true mask sized to the candidate list (API parity)."""
        return [True for _ in self.candidates(raw_state)]

    def public_action(self, candidate: dict) -> dict:
        """Return the dispatchable action dict for a candidate."""
        return dict(candidate["action"])

    def _candidate(self, action: dict, key: str) -> dict:
        action = dict(action)
        action["action_key"] = key
        return {"action": action, "action_key": key}
