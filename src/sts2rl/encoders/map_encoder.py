"""Map encoder: choose the next node among the legal next options."""

from __future__ import annotations

from sts2rl.encoders.base import CandidateEncoder


MAP_NODE_VOCAB = ("monster", "elite", "boss", "rest", "shop", "treasure", "event", "ancient")
MAP_ACTION_TYPES = ("choose_map_node", "proceed")


class MapEncoder(CandidateEncoder):
  """Encode the map screen and enumerate the legal next-node choices."""

  ACTION_TYPES = MAP_ACTION_TYPES
  SCHEMA = "map"
  DQN_SCHEMA = "map_dqn_v1"
  PPO_SCHEMA = "map_ppo_v1"

  MAX_OPTIONS = 8

  def __init__(self):
    self.node_vocab_size = len(MAP_NODE_VOCAB)
    self.action_feature_size = (
      len(self.ACTION_TYPES)        # action-type one-hot
      + self.node_vocab_size        # node-type one-hot
      + 1                           # option index (scaled)
      + 2                           # node col/row (scaled)
    )
    self.state_size = (
      6                             # hp_ratio, hp, max_hp, gold, floor, act
      + self.node_vocab_size        # upcoming node-type presence among options
      + 1                           # number of options (scaled)
    )
    self.model_input_size = self.state_size + self.action_feature_size

  def valid_action_candidates(self, raw_state: dict) -> list[dict]:
    options = self._next_options(raw_state)
    candidates = []
    for fallback_index, option in enumerate(options):
      index = self._parse_int(option.get("index", fallback_index), fallback_index)
      candidates.append(self._candidate(
        {"type": "choose_map_node", "index": index},
        f"choose_map_node:{index}",
      ))
    if not candidates:
      candidates.append(self._candidate({"type": "proceed"}, "proceed"))
    return candidates

  def encode_state(self, raw_state: dict, action_mask=None) -> list[float]:
    features = self._player_run_features(raw_state)
    options = self._next_options(raw_state)
    presence = [0.0 for _ in MAP_NODE_VOCAB]
    for option in options:
      for index, value in enumerate(self._onehot(option.get("type"), MAP_NODE_VOCAB)):
        if value:
          presence[index] = 1.0
    features.extend(presence)
    features.append(self._scale(len(options), self.MAX_OPTIONS))
    return [float(value) for value in features]

  def encode_action(self, raw_state: dict, action: dict) -> list[float]:
    action_type = action.get("type")
    features = self._action_type_features(action_type)
    if action_type == "choose_map_node":
      option = self._option_by_index(raw_state, self._parse_int(action.get("index", -1)))
      features.extend(self._onehot((option or {}).get("type"), MAP_NODE_VOCAB))
      features.append(self._scale(action.get("index", 0), self.MAX_OPTIONS))
      features.append(self._scale((option or {}).get("col", 0), 12))
      features.append(self._scale((option or {}).get("row", 0), 20))
    else:
      features.extend([0.0] * self.node_vocab_size)
      features.extend([0.0, 0.0, 0.0])
    return [float(value) for value in features]

  def action_key(self, action: dict, raw_state: dict | None = None) -> str:
    if action.get("action_key"):
      return str(action["action_key"])
    if action.get("type") == "choose_map_node":
      return f"choose_map_node:{action.get('index')}"
    return "proceed"

  def _next_options(self, raw_state: dict) -> list[dict]:
    map_state = raw_state.get("map", {})
    return map_state.get("next_options", raw_state.get("next_options", []))

  def _option_by_index(self, raw_state: dict, index: int) -> dict | None:
    for fallback_index, option in enumerate(self._next_options(raw_state)):
      if self._parse_int(option.get("index", fallback_index), fallback_index) == index:
        return option
    return None
