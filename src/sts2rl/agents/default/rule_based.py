"""Fallback policy for screens that do not need specialized handling yet."""

import logging


logger = logging.getLogger(__name__)


class DefaultPolicy:
    """Proceed through unhandled screens using the simplest legal action."""

    def choose_action(self, state: dict) -> dict:
        """Return the default transition action for the current raw state."""
        logger.debug(
            "DefaultPolicy: choosing action state_type=%s",
            state.get("state_type"),
        )
        action = {
            "type": "proceed"
        }
        logger.debug("DefaultPolicy: selected action=%s", action)
        return action
