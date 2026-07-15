"""Shop / fake-merchant action space: purchase affordable items or leave."""

from __future__ import annotations

from sts2rl.action_spaces.base import ActionSpace, parse_int


def container(raw_state: dict) -> dict:
    """Return the shop payload (shared with the shop featurizer)."""
    if raw_state.get("state_type") == "fake_merchant":
        return raw_state.get("fake_merchant", {})
    return raw_state.get("shop", {})


def items(raw_state: dict) -> list[dict]:
    """Return all shop items."""
    return container(raw_state).get("items", [])


def affordable_items(raw_state: dict) -> list[dict]:
    """Return stocked items the player can afford."""
    return [
        item
        for item in items(raw_state)
        if item.get("is_stocked", True) and item.get("can_afford", False)
    ]


def item_by_index(raw_state: dict, index: int) -> dict | None:
    """Return the shop item carrying a given index, if any."""
    for item in items(raw_state):
        if parse_int(item.get("index", -1)) == index:
            return item
    return None


class ShopActionSpace(ActionSpace):
    """Enumerate affordable purchases (+ leave) on shop screens."""

    def candidates(self, raw_state: dict) -> list[dict]:
        candidates = []
        for item in affordable_items(raw_state):
            index = parse_int(item.get("index", 0))
            candidates.append(
                self._candidate(
                    {"type": "shop_purchase", "index": index},
                    f"shop_purchase:{index}",
                )
            )
        if container(raw_state).get("can_proceed", True):
            candidates.append(self._candidate({"type": "proceed"}, "proceed"))
        return candidates

    def action_key(self, action: dict, raw_state: dict | None = None) -> str:
        if action.get("action_key"):
            return str(action["action_key"])
        if action.get("type") == "shop_purchase":
            return f"shop_purchase:{action.get('index')}"
        return "proceed"
