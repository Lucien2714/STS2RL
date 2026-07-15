"""Shop / fake-merchant featurizer: encode shop state and purchase actions."""

from __future__ import annotations

from sts2rl.action_spaces.shop import affordable_items, item_by_index, items
from sts2rl.encoders.base import CandidateEncoder

SHOP_CATEGORY_VOCAB = ("card", "relic", "potion", "card_removal")
# Schema-critical: the tuple order defines the encode_action one-hot layout.
SHOP_ACTION_TYPES = ("shop_purchase", "proceed")


class ShopEncoder(CandidateEncoder):
    """Encode shop state and purchase/leave actions."""

    ACTION_TYPES = SHOP_ACTION_TYPES
    SCHEMA = "shop"
    DQN_SCHEMA = "shop_dqn_v1"
    PPO_SCHEMA = "shop_ppo_v1"

    MAX_ITEMS = 16

    def __init__(self):
        self.category_vocab_size = len(SHOP_CATEGORY_VOCAB)
        self.action_feature_size = (
            len(self.ACTION_TYPES)  # action-type one-hot
            + self.category_vocab_size  # item category one-hot
            + 1  # price (scaled)
            + 1  # index (scaled)
        )
        self.state_size = (
            6  # hp_ratio, hp, max_hp, gold, floor, act
            + self.category_vocab_size  # affordable categories present
            + 2  # num affordable, num total (scaled)
        )
        self.model_input_size = self.state_size + self.action_feature_size

    def encode_state(self, raw_state: dict, action_mask=None) -> list[float]:
        features = self._player_run_features(raw_state)
        affordable = affordable_items(raw_state)
        presence = [0.0 for _ in SHOP_CATEGORY_VOCAB]
        for item in affordable:
            for index, value in enumerate(self._onehot(item.get("category"), SHOP_CATEGORY_VOCAB)):
                if value:
                    presence[index] = 1.0
        features.extend(presence)
        features.append(self._scale(len(affordable), self.MAX_ITEMS))
        features.append(self._scale(len(items(raw_state)), self.MAX_ITEMS))
        return [float(value) for value in features]

    def encode_action(self, raw_state: dict, action: dict) -> list[float]:
        action_type = action.get("type")
        features = self._action_type_features(action_type)
        if action_type == "shop_purchase":
            item = item_by_index(raw_state, self._parse_int(action.get("index", -1)))
            features.extend(self._onehot((item or {}).get("category"), SHOP_CATEGORY_VOCAB))
            features.append(self._scale((item or {}).get("price", 0), 999))
            features.append(self._scale(action.get("index", 0), self.MAX_ITEMS))
        else:
            features.extend([0.0] * self.category_vocab_size)
            features.extend([0.0, 0.0])
        return [float(value) for value in features]
