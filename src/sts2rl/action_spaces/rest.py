"""Rest-site action space: choose among enabled rest options."""

from __future__ import annotations

from sts2rl.action_spaces.base import ActionSpace, parse_int


def enabled_options(raw_state: dict) -> list[dict]:
    """Return the enabled rest-site options (shared with the rest featurizer)."""
    options = raw_state.get("rest_site", {}).get("options", [])
    return [option for option in options if option.get("is_enabled", True)]


def option_by_index(raw_state: dict, index: int) -> dict | None:
    """Return the rest-site option carrying a given index, if any."""
    for option in raw_state.get("rest_site", {}).get("options", []):
        if parse_int(option.get("index", -1)) == index:
            return option
    return None


class RestActionSpace(ActionSpace):
    """Enumerate the enabled rest options as candidates."""

    def candidates(self, raw_state: dict) -> list[dict]:
        candidates = []
        for option in enabled_options(raw_state):
            index = parse_int(option.get("index", 0))
            candidates.append(
                self._candidate(
                    {"type": "choose_rest_option", "index": index},
                    f"choose_rest_option:{index}",
                )
            )
        return candidates

    def action_key(self, action: dict, raw_state: dict | None = None) -> str:
        if action.get("action_key"):
            return str(action["action_key"])
        if action.get("type") == "choose_rest_option":
            return f"choose_rest_option:{action.get('index')}"
        return "proceed"
