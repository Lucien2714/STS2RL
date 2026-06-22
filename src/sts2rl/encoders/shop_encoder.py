"""Shop / fake-merchant encoder: purchase affordable items or leave."""

from __future__ import annotations

from sts2rl.encoders.base import CandidateEncoder


SHOP_CATEGORY_VOCAB = ("card", "relic", "potion", "card_removal")
SHOP_ACTION_TYPES = ("shop_purchase", "proceed")


class ShopEncoder(CandidateEncoder):
  """Encode shop state and enumerate affordable purchases (+ leave)."""

  ACTION_TYPES = SHOP_ACTION_TYPES
  SCHEMA = "shop"
  DQN_SCHEMA = "shop_dqn_v1"
  PPO_SCHEMA = "shop_ppo_v1"

  MAX_ITEMS = 16

  def __init__(self):
    self.category_vocab_size = len(SHOP_CATEGORY_VOCAB)
    self.action_feature_size = (
      len(self.ACTION_TYPES)        # action-type one-hot
      + self.category_vocab_size    # item category one-hot
      + 1                           # price (scaled)
      + 1                           # index (scaled)
    )
    self.state_size = (
      6                             # hp_ratio, hp, max_hp, gold, floor, act
      + self.category_vocab_size    # affordable categories present
      + 2                           # num affordable, num total (scaled)
    )
    self.model_input_size = self.state_size + self.action_feature_size

  def valid_action_candidates(self, raw_state: dict) -> list[dict]:
    candidates = []
    for item in self._affordable_items(raw_state):
      index = self._parse_int(item.get("index", 0))
      candidates.append(self._candidate(
        {"type": "shop_purchase", "index": index},
        f"shop_purchase:{index}",
      ))
    container = self._container(raw_state)
    if container.get("can_proceed", True):
      candidates.append(self._candidate({"type": "proceed"}, "proceed"))
    return candidates

  def encode_state(self, raw_state: dict, action_mask=None) -> list[float]:
    features = self._player_run_features(raw_state)
    affordable = self._affordable_items(raw_state)
    presence = [0.0 for _ in SHOP_CATEGORY_VOCAB]
    for item in affordable:
      for index, value in enumerate(self._onehot(item.get("category"), SHOP_CATEGORY_VOCAB)):
        if value:
          presence[index] = 1.0
    features.extend(presence)
    features.append(self._scale(len(affordable), self.MAX_ITEMS))
    features.append(self._scale(len(self._items(raw_state)), self.MAX_ITEMS))
    return [float(value) for value in features]

  def encode_action(self, raw_state: dict, action: dict) -> list[float]:
    action_type = action.get("type")
    features = self._action_type_features(action_type)
    if action_type == "shop_purchase":
      item = self._item_by_index(raw_state, self._parse_int(action.get("index", -1)))
      features.extend(self._onehot((item or {}).get("category"), SHOP_CATEGORY_VOCAB))
      features.append(self._scale((item or {}).get("price", 0), 999))
      features.append(self._scale(action.get("index", 0), self.MAX_ITEMS))
    else:
      features.extend([0.0] * self.category_vocab_size)
      features.extend([0.0, 0.0])
    return [float(value) for value in features]

  def action_key(self, action: dict, raw_state: dict | None = None) -> str:
    if action.get("action_key"):
      return str(action["action_key"])
    if action.get("type") == "shop_purchase":
      return f"shop_purchase:{action.get('index')}"
    return "proceed"

  def _container(self, raw_state: dict) -> dict:
    if raw_state.get("state_type") == "fake_merchant":
      return raw_state.get("fake_merchant", {})
    return raw_state.get("shop", {})

  def _items(self, raw_state: dict) -> list[dict]:
    return self._container(raw_state).get("items", [])

  def _affordable_items(self, raw_state: dict) -> list[dict]:
    return [
      item for item in self._items(raw_state)
      if item.get("is_stocked", True) and item.get("can_afford", False)
    ]

  def _item_by_index(self, raw_state: dict, index: int) -> dict | None:
    for item in self._items(raw_state):
      if self._parse_int(item.get("index", -1)) == index:
        return item
    return None
