"""Stateless helpers for extracting normalized values from raw game state."""

from __future__ import annotations

from typing import Any

from sts2rl.env.mcp_client import STS2ClientError
from sts2rl.env.types import RawState


def extract_raw_state(response: Any) -> RawState:
    """Extract a raw state from any supported STS2MCP response shape."""
    if isinstance(response, dict):
        if isinstance(response.get("state"), dict):
            return response["state"]
        if isinstance(response.get("raw_state"), dict):
            return response["raw_state"]
        if response.get("state_type") is not None:
            return response

    raise STS2ClientError(f"Response did not include a game state: {response}")


def parse_int(value: object, default: int = 0) -> int:
    """Parse an integer-like value, returning a default on invalid input."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def player_field(state: dict, field: str, default: int = 0) -> int:
    """Return one integer player field from a raw state dictionary."""
    player = state.get("player", {})
    if isinstance(player, dict) and player.get(field) is not None:
        return parse_int(player.get(field), default)
    return default


def player_hp(state: dict, default: int = 0) -> int:
    """Return the player's current HP from a raw state dictionary."""
    return player_field(state, "hp", default)


def player_gold(state: dict, default: int = 0) -> int:
    """Return the player's gold from a raw state dictionary."""
    return player_field(state, "gold", default)


def player_max_hp(state: dict, default: int = 0) -> int:
    """Return the player's maximum HP from a raw state dictionary."""
    return player_field(state, "max_hp", default)


def enemy_key(enemy: dict, enemy_index: int) -> str:
    """Choose a stable enemy identifier, falling back to list position."""
    for key in ("entity_id", "id", "combat_id", "name"):
        value = enemy.get(key)
        if value is not None:
            return str(value)
    return str(enemy_index)


def enemy_hp_map(state: dict) -> dict[str, int]:
    """Return a mapping of enemy identifier to current HP."""
    enemies = state.get("battle", {}).get("enemies", state.get("enemies", []))
    return {
        enemy_key(enemy, enemy_index): parse_int(enemy.get("hp", 0))
        for enemy_index, enemy in enumerate(enemies)
    }


def battle_has_alive_enemy(state: dict) -> bool:
    """Return whether the state contains any enemy with positive HP."""
    return any(hp > 0 for hp in enemy_hp_map(state).values())
