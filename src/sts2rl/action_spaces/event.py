"""Event action space: event option choices and non-battle card-selection overlays."""

from __future__ import annotations

from sts2rl.action_spaces.base import ActionSpace, parse_int
from sts2rl.action_spaces.selection import (
    can_confirm_selection,
    can_select_more,
    selected_card_indices,
    selection_selected_count,
)


def event_option(raw_state: dict, index: int) -> dict | None:
    """Return the event option carrying a given index, if any."""
    for fallback_index, option in enumerate(raw_state.get("event", {}).get("options", [])):
        if parse_int(option.get("index", fallback_index), fallback_index) == index:
            return option
    return None


class EventActionSpace(ActionSpace):
    """Enumerate legal actions on event / non-battle card_select screens."""

    def candidates(self, raw_state: dict) -> list[dict]:
        if raw_state.get("state_type") == "card_select":
            return self._card_select_candidates(raw_state)
        return self._event_candidates(raw_state)

    def action_key(self, action: dict, raw_state: dict | None = None) -> str:
        if action.get("action_key"):
            return str(action["action_key"])
        action_type = action.get("type")
        if action_type == "choose_event_option":
            return f"choose_event_option:{action.get('index')}"
        if action_type == "select_card":
            return f"select_card:{action.get('index')}"
        if action_type in {"advance_dialogue", "confirm_selection", "cancel_selection"}:
            return action_type
        return "advance_dialogue"

    def _event_candidates(self, raw_state: dict) -> list[dict]:
        event = raw_state.get("event", {})
        if event.get("in_dialogue"):
            return [self._candidate({"type": "advance_dialogue"}, "advance_dialogue")]

        candidates = []
        for fallback_index, option in enumerate(event.get("options", [])):
            if option.get("is_locked", False):
                continue
            index = parse_int(option.get("index", fallback_index), fallback_index)
            candidates.append(
                self._candidate(
                    {"type": "choose_event_option", "index": index},
                    f"choose_event_option:{index}",
                )
            )
        return candidates

    def _card_select_candidates(self, raw_state: dict) -> list[dict]:
        card_select = raw_state.get("card_select", {})
        cards = card_select.get("cards", [])
        selected_indices = selected_card_indices(card_select)
        selected_count = selection_selected_count(raw_state, "card_select", len(selected_indices))

        candidates = []
        if can_select_more(raw_state, "card_select", selected_count):
            for fallback_index, card in enumerate(cards):
                index = parse_int(card.get("index", fallback_index), fallback_index)
                if index in selected_indices:
                    continue
                candidates.append(
                    self._candidate(
                        {"type": "select_card", "index": index},
                        f"select_card:{index}",
                    )
                )
        if can_confirm_selection(raw_state, "card_select", selected_count):
            candidates.append(self._candidate({"type": "confirm_selection"}, "confirm_selection"))
        if card_select.get("can_cancel", False):
            candidates.append(self._candidate({"type": "cancel_selection"}, "cancel_selection"))
        return candidates
