"""Rule-based event and card-selection policy."""

import logging
import re

from sts2rl.agents.base import EventAgent
from sts2rl.agents.selection import (
    can_confirm_selection,
    can_select_more,
    exact_required_count,
    needs_more_selections,
    remaining_to_min,
    selection_selected_count,
)


logger = logging.getLogger(__name__)


class EventPolicy(EventAgent):
    """Handle event choices and modal card-selection screens."""

    def __init__(self):
        self._card_select_key: tuple | None = None
        self._remembered_selected_indices: set[int] = set()

    def choose_action(self, state: dict) -> dict:
        """Choose an event or card-selection action from raw screen state."""
        state_type = state.get("state_type")

        if state_type == "card_select":
            return self._choose_card_select_action(state)

        return self._choose_event_action(state)

    def _choose_event_action(self, state: dict) -> dict:
        """Choose the first available event option or advance dialogue."""
        event = state.get("event", {})
        options = event.get("options", [])
        logger.debug(
            "EventPolicy: choosing event action event_id=%s in_dialogue=%s options=%d",
            event.get("event_id"),
            event.get("in_dialogue"),
            len(options),
        )

        if event.get("in_dialogue"):
            action = {"type": "advance_dialogue"}
            logger.debug("EventPolicy: selected action=%s", action)
            return action

        available_options = [
            option for option in options
            if not option.get("is_locked", False)
        ]

        if not available_options:
            action = {"type": "advance_dialogue"}
            logger.debug(
                "EventPolicy: no available options; selected fallback action=%s",
                action,
            )
            return action

        # Placeholder:
        # choose first unlocked event option
        option = available_options[0]
        action = {
            "type": "choose_event_option",
            "index": option.get("index", 0),
        }
        logger.debug(
            "EventPolicy: selected event option index=%s title=%s is_proceed=%s",
            option.get("index"),
            option.get("title"),
            option.get("is_proceed"),
        )
        logger.debug("EventPolicy: selected action=%s", action)
        return action

    def _choose_card_select_action(self, state: dict) -> dict:
        """Select required cards and confirm or cancel the card-selection overlay."""
        card_select = state.get("card_select", {})
        screen_type = card_select.get("screen_type")
        prompt = card_select.get("prompt", "")
        cards = card_select.get("cards", [])
        selected_indices = self._card_select_selected_indices(card_select, cards)
        selected_count = selection_selected_count(
            state,
            "card_select",
            len(selected_indices),
        )
        fallback_required_count = self._required_card_count_from_prompt(prompt)
        required_count = exact_required_count(state, "card_select")
        if required_count is None:
            required_count = fallback_required_count

        logger.debug(
            "EventPolicy: choosing card_select action screen_type=%s cards=%d selected=%d required=%s remaining_to_min=%s can_confirm=%s can_cancel=%s preview_showing=%s prompt=%s",
            screen_type,
            len(cards),
            selected_count,
            required_count,
            remaining_to_min(state, "card_select"),
            card_select.get("can_confirm"),
            card_select.get("can_cancel"),
            card_select.get("preview_showing"),
            prompt,
        )

        if card_select.get("preview_showing") and card_select.get("can_confirm"):
            action = {"type": "confirm_selection"}
            self._reset_card_select_memory()
            logger.debug("EventPolicy: selected action=%s", action)
            return action

        if can_confirm_selection(
            state,
            "card_select",
            selected_count,
            fallback_required_count,
        ):
            action = {"type": "confirm_selection"}
            self._reset_card_select_memory()
            logger.debug("EventPolicy: selected action=%s", action)
            return action

        should_select = needs_more_selections(
            state,
            "card_select",
            selected_count,
            fallback_required_count,
        ) or not card_select.get("can_confirm", False)
        if cards and should_select and can_select_more(
            state,
            "card_select",
            selected_count,
            fallback_required_count,
        ):
            card = self._choose_card(cards, screen_type, selected_indices)
            card_index = self._card_index(card)
            self._remembered_selected_indices.add(card_index)
            action = {
                "type": "select_card",
                "index": card_index,
            }
            logger.debug(
                "EventPolicy: selected card index=%s id=%s name=%s upgraded=%s",
                card_index,
                card.get("id"),
                card.get("name"),
                card.get("is_upgraded"),
            )
            logger.debug("EventPolicy: selected action=%s", action)
            return action

        if card_select.get("can_confirm"):
            action = {"type": "confirm_selection"}
            self._reset_card_select_memory()
            logger.debug("EventPolicy: selected action=%s", action)
            return action

        if card_select.get("can_cancel"):
            action = {"type": "cancel_selection"}
            self._reset_card_select_memory()
            logger.debug("EventPolicy: selected action=%s", action)
            return action

        action = {"type": "confirm_selection"}
        self._reset_card_select_memory()
        logger.debug("EventPolicy: selected action=%s", action)
        return action

    def _choose_card(
        self,
        cards: list[dict],
        screen_type: str | None,
        selected_indices: set[int] | None = None,
    ) -> dict:
        """Pick the next card, preferring unupgraded cards for upgrade screens."""
        selected_indices = selected_indices or set()
        available_cards = [
            card for card in cards
            if self._card_index(card) not in selected_indices
        ]

        if not available_cards:
            return cards[0]

        if screen_type == "upgrade":
            for card in available_cards:
                if not card.get("is_upgraded", False):
                    return card

        return available_cards[0]

    def _required_card_count_from_prompt(self, prompt: str) -> int | None:
        """Infer a required selection count from STS2MCP prompt text."""
        if re.search(r"\bChoose\s+up to\s+\d+\s+cards?\b", prompt, re.IGNORECASE):
            return None

        match = re.search(r"\bChoose\s+(\d+)\s+cards?\b", prompt, re.IGNORECASE)
        if match is None:
            if re.search(r"\bChoose\s+a\s+card\b", prompt, re.IGNORECASE):
                return 1
            return None

        return int(match.group(1))

    def _card_select_selected_indices(
        self,
        card_select: dict,
        cards: list[dict],
    ) -> set[int]:
        """Track selected card indices when the backend omits selection state."""
        key = self._card_select_memory_key(card_select, cards)
        if key != self._card_select_key:
            self._card_select_key = key
            self._remembered_selected_indices = set()

        if card_select.get("selected_cards"):
            self._remembered_selected_indices = self._selected_card_indices(
                card_select.get("selected_cards", [])
            )

        return set(self._remembered_selected_indices)

    def _card_select_memory_key(self, card_select: dict, cards: list[dict]) -> tuple:
        card_ids = tuple(
            (
                self._card_index(card),
                card.get("id"),
                card.get("name"),
            )
            for card in cards
        )
        return (
            card_select.get("screen_type"),
            card_select.get("prompt"),
            card_ids,
        )

    def _reset_card_select_memory(self) -> None:
        self._card_select_key = None
        self._remembered_selected_indices = set()

    def _selected_card_indices(self, selected_cards: list[dict]) -> set[int]:
        indices = set()
        for card in selected_cards:
            indices.add(self._card_index(card))
        return indices

    def _card_index(self, card: dict) -> int:
        try:
            return int(card.get("index", 0))
        except (TypeError, ValueError):
            return 0
