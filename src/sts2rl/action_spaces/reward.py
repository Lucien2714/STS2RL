"""Reward action space: rewards list, card rewards, and treasure relics."""

from __future__ import annotations

from sts2rl.action_spaces.base import ActionSpace, parse_int


class RewardActionSpace(ActionSpace):
    """Enumerate legal actions on rewards / card_reward / treasure screens."""

    def candidates(self, raw_state: dict) -> list[dict]:
        state_type = raw_state.get("state_type")
        if state_type == "card_reward":
            return self._card_reward_candidates(raw_state)
        if state_type == "treasure":
            return self._treasure_candidates(raw_state)
        return self._rewards_candidates(raw_state)

    def action_key(self, action: dict, raw_state: dict | None = None) -> str:
        if action.get("action_key"):
            return str(action["action_key"])
        action_type = action.get("type")
        if action_type == "claim_reward":
            return f"claim_reward:{action.get('index')}"
        if action_type == "select_card_reward":
            return f"select_card_reward:{action.get('card_index')}"
        if action_type == "claim_treasure_relic":
            return f"claim_treasure_relic:{action.get('index')}"
        if action_type == "skip_card_reward":
            return "skip_card_reward"
        return "proceed"

    def _rewards_candidates(self, raw_state: dict) -> list[dict]:
        rewards = raw_state.get("rewards", {})
        candidates = []
        for fallback_index, item in enumerate(rewards.get("items", [])):
            index = parse_int(item.get("index", fallback_index), fallback_index)
            candidates.append(
                self._candidate(
                    {"type": "claim_reward", "index": index},
                    f"claim_reward:{index}",
                )
            )
        if rewards.get("can_proceed", bool(candidates)):
            candidates.append(self._candidate({"type": "proceed"}, "proceed"))
        return candidates

    def _card_reward_candidates(self, raw_state: dict) -> list[dict]:
        card_reward = raw_state.get("card_reward", {})
        candidates = []
        for fallback_index, card in enumerate(card_reward.get("cards", [])):
            index = parse_int(card.get("index", fallback_index), fallback_index)
            candidates.append(
                self._candidate(
                    {"type": "select_card_reward", "card_index": index},
                    f"select_card_reward:{index}",
                )
            )
        if card_reward.get("can_skip", True):
            candidates.append(self._candidate({"type": "skip_card_reward"}, "skip_card_reward"))
        return candidates

    def _treasure_candidates(self, raw_state: dict) -> list[dict]:
        treasure = raw_state.get("treasure", {})
        candidates = []
        for fallback_index, relic in enumerate(treasure.get("relics", [])):
            index = parse_int(relic.get("index", fallback_index), fallback_index)
            candidates.append(
                self._candidate(
                    {"type": "claim_treasure_relic", "index": index},
                    f"claim_treasure_relic:{index}",
                )
            )
        if treasure.get("can_proceed", bool(candidates)):
            candidates.append(self._candidate({"type": "proceed"}, "proceed"))
        return candidates
