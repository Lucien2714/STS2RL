"""Shared plumbing for screen action spaces.

An action space enumerates the currently legal candidate actions for one screen
(each a dict with ``action`` and a stable ``action_key``) and keys arbitrary game
actions. It carries the game-rule half of what used to live in the screen
encoders; representation (feature vectors) stays in ``encoders/``.

This module also holds the API legality table (:data:`LEGAL_ACTION_TYPES`) that
records which POST actions STS2MCP accepts per ``state_type``. It lives here
rather than in :mod:`sts2rl.action_spaces.defaults` so the concrete spaces can
consult it — ``defaults`` imports them, so they cannot import ``defaults``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

# A no-op "action" that re-reads game state. Dispatched by ActionDispatcher as a
# plain get_state, so a state with nothing legal to do costs one GET instead of a
# POST the API would reject.
REFRESH_STATE_ACTION = {"type": "refresh_state"}

# Potions are usable from any screen where they are accessible, so these are
# allowed on top of each screen's own action list.
POTION_ACTION_TYPES = frozenset({"use_potion", "discard_potion"})

# state_type -> POST actions the API accepts there (docs/STS2MCP-raw-full.md).
LEGAL_ACTION_TYPES: dict[str, frozenset[str]] = {
    "monster": frozenset({"play_card", "end_turn"}),
    "elite": frozenset({"play_card", "end_turn"}),
    "boss": frozenset({"play_card", "end_turn"}),
    "hand_select": frozenset({"combat_select_card", "combat_confirm_selection"}),
    "card_select": frozenset({"select_card", "confirm_selection", "cancel_selection"}),
    "rewards": frozenset({"claim_reward", "proceed"}),
    "card_reward": frozenset({"select_card_reward", "skip_card_reward"}),
    "treasure": frozenset({"claim_treasure_relic", "proceed"}),
    "map": frozenset({"choose_map_node"}),
    "event": frozenset({"advance_dialogue", "choose_event_option"}),
    "rest": frozenset({"choose_rest_option", "proceed"}),
    "rest_site": frozenset({"choose_rest_option", "proceed"}),
    "shop": frozenset({"shop_purchase", "proceed"}),
    "fake_merchant": frozenset({"shop_purchase", "proceed"}),
    "relic_select": frozenset({"select_relic", "skip_relic_selection"}),
    "menu": frozenset({"menu_select"}),
    "game_over": frozenset({"menu_select"}),
}

# Screens whose POST surface includes `proceed`, derived so the two stay in sync.
PROCEED_STATE_TYPES = frozenset(
    state_type
    for state_type, action_types in LEGAL_ACTION_TYPES.items()
    if "proceed" in action_types
)


def parse_int(value: object, default: int = 0) -> int:
    """Parse integer-like MCP fields, falling back on missing/invalid values."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def is_legal_action_type(state_type: str | None, action_type: str | None) -> bool:
    """Return whether the API accepts ``action_type`` on ``state_type``.

    Unknown screens return ``True`` — this is a guard against known-illegal
    actions, not an allowlist that should block unmapped states.
    """
    if action_type in POTION_ACTION_TYPES or action_type == REFRESH_STATE_ACTION["type"]:
        return True
    allowed = LEGAL_ACTION_TYPES.get(str(state_type))
    if allowed is None:
        return True
    return action_type in allowed


def proceed_or_refresh(raw_state: dict) -> dict:
    """Return ``proceed`` when the screen accepts it, else a state refresh."""
    if raw_state.get("state_type") in PROCEED_STATE_TYPES:
        return {"type": "proceed", "action_key": "proceed"}
    return dict(REFRESH_STATE_ACTION)


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
        screen's rule-based policy for the real fallback.

        Never returns ``proceed`` on a screen that does not accept it — screens
        with nothing legal to do get a state refresh instead.
        """
        return proceed_or_refresh(raw_state)

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
