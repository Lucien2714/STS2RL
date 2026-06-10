"""Stateless helpers for extracting normalized values from raw game state."""

from __future__ import annotations


def parse_int(value: object, default: int = 0) -> int:
    """Parse an integer-like value, returning a default on invalid input."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def player_hp(state: dict, default: int | None = 0) -> int:
    """Return the player's current HP from a raw state dictionary."""
    player = state.get("player", {})
    if isinstance(player, dict) and player.get("hp") is not None:
        return parse_int(player.get("hp"), default or 0)
    return default or 0


def player_gold(state: dict, default: int | None = 0) -> int:
    """Return the player's gold from a raw state dictionary."""
    player = state.get("player", {})
    if isinstance(player, dict) and player.get("gold") is not None:
        return parse_int(player.get("gold"), default or 0)
    return default or 0


def player_max_hp(state: dict, default: int | None = 0) -> int:
    """Return the player's maximum HP from a raw state dictionary."""
    player = state.get("player", {})
    if isinstance(player, dict) and player.get("max_hp") is not None:
        return parse_int(player.get("max_hp"), default or 0)
    return default or 0


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
