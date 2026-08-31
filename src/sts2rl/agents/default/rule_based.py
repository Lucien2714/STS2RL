"""Fallback policy for screens that do not need specialized handling yet."""

import logging

from sts2rl.action_spaces.defaults import default_action

logger = logging.getLogger(__name__)


class DefaultPolicy:
    """Advance unhandled screens using whatever action they actually accept."""

    def choose_action(self, state: dict) -> dict:
        """Return the default transition action for the current raw state.

        This used to hardcode ``proceed``, which the API rejects on most screens
        (combat, map, card_reward, event, the selection overlays). It now asks
        ``action_spaces.defaults`` for a legal action instead.
        """
        action = default_action(state)
        logger.debug(
            "DefaultPolicy: state_type=%s selected action=%s",
            state.get("state_type"),
            action,
        )
        return action
