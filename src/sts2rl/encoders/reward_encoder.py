"""Reward-screen featurizer: rewards list, card rewards, and treasure relics."""

from __future__ import annotations

from sts2rl.action_spaces.reward import RewardActionSpace
from sts2rl.data.loader import get_card_index, get_card_map_size, get_relic_index
from sts2rl.encoders.base import CandidateEncoder

REWARD_ITEM_VOCAB = ("card", "relic", "potion", "gold")
# Schema-critical: the tuple order defines the encode_action one-hot layout.
REWARD_ACTION_TYPES = (
    "claim_reward",
    "select_card_reward",
    "skip_card_reward",
    "claim_treasure_relic",
    "proceed",
)


class RewardEncoder(CandidateEncoder):
    """Encode rewards / card_reward / treasure screen states and actions."""

    ACTION_TYPES = REWARD_ACTION_TYPES
    SCHEMA = "reward"
    DQN_SCHEMA = "reward_dqn_v1"
    PPO_SCHEMA = "reward_ppo_v1"

    MAX_ITEMS = 12

    def __init__(self):
        # The state vector includes the current candidate count, so this
        # featurizer composes the reward action space for that one feature.
        self._action_space = RewardActionSpace()
        self.item_vocab_size = len(REWARD_ITEM_VOCAB)
        self.action_feature_size = (
            len(self.ACTION_TYPES)  # action-type one-hot
            + 1  # item index (scaled)
            + self.item_vocab_size  # item-kind one-hot
            + 1  # card/relic id (scaled)
            + 1  # is_upgraded flag
        )
        self.state_size = (
            6  # hp_ratio, hp, max_hp, gold, floor, act
            + self.item_vocab_size  # present reward kinds
            + 1  # number of candidates (scaled)
        )
        self.model_input_size = self.state_size + self.action_feature_size

    def encode_state(self, raw_state: dict, action_mask=None) -> list[float]:
        features = self._player_run_features(raw_state)
        presence = [0.0 for _ in REWARD_ITEM_VOCAB]
        for item in self._reward_items(raw_state):
            for index, value in enumerate(self._onehot(item.get("type"), REWARD_ITEM_VOCAB)):
                if value:
                    presence[index] = 1.0
        features.extend(presence)
        features.append(
            self._scale(len(self._action_space.candidates(raw_state)), self.MAX_ITEMS)
        )
        return [float(value) for value in features]

    def encode_action(self, raw_state: dict, action: dict) -> list[float]:
        action_type = action.get("type")
        features = self._action_type_features(action_type)
        features.append(
            self._scale(action.get("index", action.get("card_index", 0)), self.MAX_ITEMS)
        )

        kind = ""
        card_or_relic_id = 0.0
        upgraded = 0.0
        if action_type == "claim_reward":
            item = self._rewards_item(raw_state, self._parse_int(action.get("index", -1)))
            kind = (item or {}).get("type", "")
        elif action_type == "select_card_reward":
            kind = "card"
            card = self._card_reward_card(raw_state, self._parse_int(action.get("card_index", -1)))
            if card is not None:
                card_or_relic_id = self._scale(
                    get_card_index(card.get("id") or card.get("name"), default=-1) + 1,
                    max(1, get_card_map_size()),
                )
                upgraded = 1.0 if card.get("is_upgraded", card.get("upgraded", False)) else 0.0
        elif action_type == "claim_treasure_relic":
            kind = "relic"
            relic = self._treasure_relic(raw_state, self._parse_int(action.get("index", -1)))
            if relic is not None:
                card_or_relic_id = self._scale(
                    get_relic_index(relic.get("id") or relic.get("name"), default=-1) + 1, 400
                )

        features.extend(self._onehot(kind, REWARD_ITEM_VOCAB))
        features.append(card_or_relic_id)
        features.append(upgraded)
        return [float(value) for value in features]

    # --- lookups -----------------------------------------------------------
    def _reward_items(self, raw_state: dict) -> list[dict]:
        state_type = raw_state.get("state_type")
        if state_type == "card_reward":
            return raw_state.get("card_reward", {}).get("cards", [])
        if state_type == "treasure":
            return raw_state.get("treasure", {}).get("relics", [])
        return raw_state.get("rewards", {}).get("items", [])

    def _rewards_item(self, raw_state: dict, index: int) -> dict | None:
        for fallback_index, item in enumerate(raw_state.get("rewards", {}).get("items", [])):
            if self._parse_int(item.get("index", fallback_index), fallback_index) == index:
                return item
        return None

    def _card_reward_card(self, raw_state: dict, index: int) -> dict | None:
        for fallback_index, card in enumerate(raw_state.get("card_reward", {}).get("cards", [])):
            if self._parse_int(card.get("index", fallback_index), fallback_index) == index:
                return card
        return None

    def _treasure_relic(self, raw_state: dict, index: int) -> dict | None:
        for fallback_index, relic in enumerate(raw_state.get("treasure", {}).get("relics", [])):
            if self._parse_int(relic.get("index", fallback_index), fallback_index) == index:
                return relic
        return None
