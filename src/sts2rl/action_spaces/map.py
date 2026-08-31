"""Map action space: choose the next node among the legal next options."""

from __future__ import annotations

from sts2rl.action_spaces.base import ActionSpace, parse_int


def next_options(raw_state: dict) -> list[dict]:
    """Return the map's next-node options (shared with the map featurizer)."""
    map_state = raw_state.get("map", {})
    return map_state.get("next_options", raw_state.get("next_options", []))


def option_by_index(raw_state: dict, index: int) -> dict | None:
    """Return the next-node option carrying a given index, if any."""
    for fallback_index, option in enumerate(next_options(raw_state)):
        if parse_int(option.get("index", fallback_index), fallback_index) == index:
            return option
    return None


class MapActionSpace(ActionSpace):
    """Enumerate the legal next-node choices on the map screen."""

    def candidates(self, raw_state: dict) -> list[dict]:
        options = next_options(raw_state)
        candidates = []
        for fallback_index, option in enumerate(options):
            index = parse_int(option.get("index", fallback_index), fallback_index)
            candidates.append(
                self._candidate(
                    {"type": "choose_map_node", "index": index},
                    f"choose_map_node:{index}",
                )
            )
        # No synthetic `proceed` filler: the map screen only accepts
        # choose_map_node, so an empty option list means there is no legal action
        # and the caller should fall back (see action_spaces.defaults).
        return candidates

    def action_key(self, action: dict, raw_state: dict | None = None) -> str:
        if action.get("action_key"):
            return str(action["action_key"])
        if action.get("type") == "choose_map_node":
            return f"choose_map_node:{action.get('index')}"
        return str(action.get("type", "unknown"))
