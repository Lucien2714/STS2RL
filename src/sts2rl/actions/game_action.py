"""Typed game actions passed from agents to the environment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class GameAction:
    """An STS2MCP action type and its request parameters."""

    def __init__(self, action_type: str, **params: Any) -> None:
        if not isinstance(action_type, str) or not action_type:
            raise ValueError("action_type must be a non-empty string")
        self.action_type = action_type
        self.params = dict(params)

    @classmethod
    def from_dict(cls, action: Mapping[str, Any]) -> GameAction:
        """Build an action from the dictionary shape produced by existing agents."""
        if "type" not in action:
            raise ValueError(f"Action dictionary is missing 'type': {action}")
        action_type = action["type"]
        if not isinstance(action_type, str):
            raise TypeError("action 'type' must be a string")
        params = dict(action)
        del params["type"]
        return cls(action_type, **params)

    def get_type(self) -> str:
        """Return the action type."""
        return self.action_type

    def get_params(self) -> dict[str, Any]:
        """Return a copy of the action parameters."""
        return dict(self.params)

    def to_dict(self) -> dict[str, Any]:
        """Convert this action to the dictionary shape used in logs and telemetry."""
        return {"type": self.action_type, **self.params}

    def __repr__(self) -> str:
        return f"GameAction(action_type={self.action_type!r}, params={self.params!r})"


class SelectCardAction(GameAction):
    """Select one card from a non-combat card-selection prompt."""

    def __init__(self, card_index: int) -> None:
        super().__init__("select_card", index=card_index)


class MenuSelectAction(GameAction):
    """Select one option from an STS2MCP menu state."""

    def __init__(self, option: str, seed: str | None = None) -> None:
        params: dict[str, Any] = {"option": option}
        if seed is not None:
            params["seed"] = seed
        super().__init__("menu_select", **params)
