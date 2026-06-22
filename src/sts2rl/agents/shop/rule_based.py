"""Rule-based shop policy."""

import logging

logger = logging.getLogger(__name__)


class ShopPolicy:
    """Leave shops without purchasing until buying heuristics are added."""

    def choose_action(self, state: dict) -> dict:
        """Return a proceed action for the current shop state."""
        cards = state.get("cards", [])
        relics = state.get("relics", [])
        potions = state.get("potions", [])
        logger.debug(
            "ShopPolicy: choosing action cards=%d relics=%d potions=%d",
            len(cards),
            len(relics),
            len(potions),
        )

        # Placeholder:
        # do not buy anything for now
        action = {"type": "proceed"}
        logger.debug("ShopPolicy: selected action=%s", action)
        return action
