"""Helpers for refreshing full player details at map decision points."""

import logging
from typing import Any

from sts2rl.env.player import Player


logger = logging.getLogger(__name__)


def refresh_player_detail_for_map(
    game: Any,
    player: Player,
    raw_state: dict,
) -> dict | None:
    """Fetch player-detail before map routing and attach it to raw state."""
    if raw_state.get("state_type") != "map":
        return None

    try:
        detail = game.get_player_detail()
        player.update_from_detail(detail)
    except Exception as exc:
        raw_state["player_detail_error"] = str(exc)
        logger.warning("Could not refresh player-detail before map node: %s", exc)
        return None

    raw_state["player_detail"] = detail
    logger.debug(
        "Refreshed player-detail before map node: floor=%s deck=%s relics=%s potions=%s",
        detail.get("run", {}).get("floor"),
        detail.get("player", {}).get("deck_count"),
        len(detail.get("player", {}).get("relics", [])),
        len(detail.get("player", {}).get("potions", [])),
    )
    return detail
