"""Rest-site screen encoder: choose among enabled rest options."""

from __future__ import annotations

from sts2rl.encoders.base import CandidateEncoder


# Coarse vocabulary of rest-site option kinds (matched as substrings of id/name).
REST_OPTION_VOCAB = ("rest", "heal", "smith", "upgrade", "dig", "lift", "recall", "toke")

REST_ACTION_TYPES = ("choose_rest_option", "proceed")


class RestEncoder(CandidateEncoder):
  """Encode rest-site state and enumerate the enabled rest options as candidates."""

  ACTION_TYPES = REST_ACTION_TYPES
  SCHEMA = "rest"
  DQN_SCHEMA = "rest_dqn_v1"
  PPO_SCHEMA = "rest_ppo_v1"

  MAX_OPTIONS = 8

  def __init__(self):
    self.option_vocab_size = len(REST_OPTION_VOCAB)
    self.action_feature_size = (
      len(self.ACTION_TYPES)        # action-type one-hot
      + self.option_vocab_size      # chosen option kind one-hot
      + 1                           # option index (scaled)
    )
    self.state_size = (
      6                             # hp_ratio, hp, max_hp, gold, floor, act
      + self.option_vocab_size      # available option-kind presence
      + 1                           # number of enabled options (scaled)
    )
    self.model_input_size = self.state_size + self.action_feature_size

  def valid_action_candidates(self, raw_state: dict) -> list[dict]:
    options = self._enabled_options(raw_state)
    candidates = []
    for option in options:
      index = self._parse_int(option.get("index", 0))
      candidates.append(self._candidate(
        {"type": "choose_rest_option", "index": index},
        f"choose_rest_option:{index}",
      ))
    return candidates

  def encode_state(self, raw_state: dict, action_mask=None) -> list[float]:
    player = raw_state.get("player", {})
    run = raw_state.get("run", {})
    hp = self._parse_int(player.get("hp", player.get("current_hp", 0)))
    max_hp = max(1, self._parse_int(player.get("max_hp", 1)))

    features = [
      max(0.0, min(hp / max_hp, 1.0)),
      self._scale(hp, max_hp),
      self._scale(max_hp, 200),
      self._scale(player.get("gold", 0), 999),
      self._scale(run.get("floor", 0), 60),
      self._scale(run.get("act", 0), 4),
    ]
    presence = [0.0 for _ in REST_OPTION_VOCAB]
    options = self._enabled_options(raw_state)
    for option in options:
      for index, value in enumerate(self._onehot(self._option_kind(option), REST_OPTION_VOCAB)):
        if value:
          presence[index] = 1.0
    features.extend(presence)
    features.append(self._scale(len(options), self.MAX_OPTIONS))
    return [float(value) for value in features]

  def encode_action(self, raw_state: dict, action: dict) -> list[float]:
    action_type = action.get("type")
    features = self._action_type_features(action_type)
    if action_type == "choose_rest_option":
      option = self._option_by_index(raw_state, self._parse_int(action.get("index", -1)))
      features.extend(self._onehot(self._option_kind(option), REST_OPTION_VOCAB))
      features.append(self._scale(action.get("index", 0), self.MAX_OPTIONS))
    else:
      features.extend([0.0] * self.option_vocab_size)
      features.append(0.0)
    return [float(value) for value in features]

  def action_key(self, action: dict, raw_state: dict | None = None) -> str:
    if action.get("action_key"):
      return str(action["action_key"])
    action_type = action.get("type")
    if action_type == "choose_rest_option":
      return f"choose_rest_option:{action.get('index')}"
    return "proceed"

  def _enabled_options(self, raw_state: dict) -> list[dict]:
    options = raw_state.get("rest_site", {}).get("options", [])
    return [option for option in options if option.get("is_enabled", True)]

  def _option_by_index(self, raw_state: dict, index: int) -> dict | None:
    for option in raw_state.get("rest_site", {}).get("options", []):
      if self._parse_int(option.get("index", -1)) == index:
        return option
    return None

  def _option_kind(self, option: dict | None) -> str:
    if not option:
      return ""
    return f"{option.get('id', '')} {option.get('name', '')}"
