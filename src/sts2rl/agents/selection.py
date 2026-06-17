"""Helpers for MCP selection screens that require one or more cards."""

from __future__ import annotations


def selection_field(raw_state: dict, selection_key: str, field: str, default=None):
    """Read a selection field from the nested selection payload or top-level state."""
    selection = raw_state.get(selection_key, {})
    if isinstance(selection, dict) and field in selection:
        return selection.get(field)
    return raw_state.get(field, default)


def parse_optional_int(value: object) -> int | None:
    """Parse integer-like MCP fields while preserving missing/invalid values."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def selection_selected_count(
    raw_state: dict,
    selection_key: str,
    fallback_count: int = 0,
) -> int:
    """Return the backend-selected count, falling back to local selected indices."""
    selected_count = parse_optional_int(
        selection_field(raw_state, selection_key, "selected_count")
    )
    if selected_count is None:
        return max(0, fallback_count)
    return max(0, selected_count, fallback_count)


def selected_card_indices(selection: dict) -> set[int]:
    """Return selected card indices from MCP selected-card payloads."""
    indices = set()
    for card in selection.get("selected_cards", []):
        card_index = parse_optional_int(card.get("index"))
        if card_index is not None:
            indices.add(card_index)

    for card in selection.get("cards", []):
        if not (card.get("selected", False) or card.get("is_selected", False)):
            continue
        card_index = parse_optional_int(card.get("index"))
        if card_index is not None:
            indices.add(card_index)

    return indices


def first_unselected_card_index(selection: dict) -> int | None:
    """Return the first visible card index not already selected."""
    selected_indices = selected_card_indices(selection)
    for fallback_index, card in enumerate(selection.get("cards", [])):
        card_index = parse_optional_int(card.get("index"))
        if card_index is None:
            card_index = fallback_index
        if card_index not in selected_indices:
            return card_index
    return None


def exact_required_count(raw_state: dict, selection_key: str) -> int | None:
    """Return an exact required count when MCP reports one."""
    required_count = parse_optional_int(
        selection_field(raw_state, selection_key, "required_select_count")
    )
    if required_count is not None:
        return max(0, required_count)

    min_select = parse_optional_int(selection_field(raw_state, selection_key, "min_select"))
    max_select = parse_optional_int(selection_field(raw_state, selection_key, "max_select"))
    if min_select is not None and max_select is not None and min_select == max_select:
        return max(0, min_select)

    return None


def remaining_to_min(raw_state: dict, selection_key: str) -> int | None:
    """Return the MCP-reported number of additional selections needed."""
    remaining = parse_optional_int(
        selection_field(raw_state, selection_key, "remaining_to_min")
    )
    if remaining is not None:
        return max(0, remaining)
    return None


def max_select_count(raw_state: dict, selection_key: str) -> int | None:
    """Return the maximum selectable count when MCP reports one."""
    max_select = parse_optional_int(selection_field(raw_state, selection_key, "max_select"))
    if max_select is not None:
        return max(0, max_select)
    return None


def needs_more_selections(
    raw_state: dict,
    selection_key: str,
    selected_count: int,
    fallback_required_count: int | None = None,
) -> bool:
    """Return whether another card should be selected before confirming."""
    required_count = exact_required_count(raw_state, selection_key)
    if required_count is None:
        required_count = fallback_required_count
    if required_count is not None:
        return selected_count < required_count

    remaining = remaining_to_min(raw_state, selection_key)
    if remaining is not None:
        return remaining > 0

    return False


def can_select_more(
    raw_state: dict,
    selection_key: str,
    selected_count: int,
    fallback_required_count: int | None = None,
) -> bool:
    """Return whether selecting another card is allowed or still useful."""
    required_count = exact_required_count(raw_state, selection_key)
    if required_count is None:
        required_count = fallback_required_count
    if required_count is not None:
        return selected_count < required_count

    max_count = max_select_count(raw_state, selection_key)
    if max_count is not None:
        return selected_count < max_count

    return True


def can_confirm_selection(
    raw_state: dict,
    selection_key: str,
    selected_count: int,
    fallback_required_count: int | None = None,
) -> bool:
    """Return whether it is correct to confirm the current selection."""
    can_confirm = bool(selection_field(raw_state, selection_key, "can_confirm", False))
    if not can_confirm:
        return False

    required_count = exact_required_count(raw_state, selection_key)
    if required_count is None:
        required_count = fallback_required_count
    if required_count is not None:
        return selected_count >= required_count

    remaining = remaining_to_min(raw_state, selection_key)
    if remaining is not None:
        return remaining <= 0

    return True
