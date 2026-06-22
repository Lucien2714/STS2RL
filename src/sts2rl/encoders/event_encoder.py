"""Event encoder: event option choices and non-battle card-selection overlays."""

from __future__ import annotations

from sts2rl.agents.selection import (
  can_confirm_selection,
  can_select_more,
  remaining_to_min,
  selected_card_indices,
  selection_selected_count,
)
from sts2rl.encoders.base import CandidateEncoder


EVENT_ACTION_TYPES = (
  "choose_event_option",
  "advance_dialogue",
  "select_card",
  "confirm_selection",
  "cancel_selection",
)


class EventEncoder(CandidateEncoder):
  """Encode event / non-battle card_select screens and enumerate their actions."""

  ACTION_TYPES = EVENT_ACTION_TYPES
  SCHEMA = "event"
  DQN_SCHEMA = "event_dqn_v1"
  PPO_SCHEMA = "event_ppo_v1"

  MAX_OPTIONS = 12
  MAX_CARDS = 20

  def __init__(self):
    self.action_feature_size = (
      len(self.ACTION_TYPES)      # action-type one-hot
      + 1                         # option/card index (scaled)
      + 1                         # is_proceed flag (event option)
    )
    self.state_size = (
      6                           # hp_ratio, hp, max_hp, gold, floor, act
      + 6                         # in_dialogue, is_card_select, n_options, n_unlocked, n_cards, remaining_to_min
    )
    self.model_input_size = self.state_size + self.action_feature_size

  def valid_action_candidates(self, raw_state: dict) -> list[dict]:
    if raw_state.get("state_type") == "card_select":
      return self._card_select_candidates(raw_state)
    return self._event_candidates(raw_state)

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
    event = raw_state.get("event", {})
    card_select = raw_state.get("card_select", {})
    is_card_select = raw_state.get("state_type") == "card_select"
    options = event.get("options", [])
    unlocked = [option for option in options if not option.get("is_locked", False)]
    features.extend([
      1.0 if event.get("in_dialogue") else 0.0,
      1.0 if is_card_select else 0.0,
      self._scale(len(options), self.MAX_OPTIONS),
      self._scale(len(unlocked), self.MAX_OPTIONS),
      self._scale(len(card_select.get("cards", [])), self.MAX_CARDS),
      self._scale(remaining_to_min(raw_state, "card_select") or 0, self.MAX_CARDS),
    ])
    return [float(value) for value in features]

  def encode_action(self, raw_state: dict, action: dict) -> list[float]:
    action_type = action.get("type")
    features = self._action_type_features(action_type)
    if action_type == "choose_event_option":
      option = self._event_option(raw_state, self._parse_int(action.get("index", -1)))
      features.append(self._scale(action.get("index", 0), self.MAX_OPTIONS))
      features.append(1.0 if (option or {}).get("is_proceed") else 0.0)
    elif action_type == "select_card":
      features.append(self._scale(action.get("index", 0), self.MAX_CARDS))
      features.append(0.0)
    else:
      features.extend([0.0, 0.0])
    return [float(value) for value in features]

  def action_key(self, action: dict, raw_state: dict | None = None) -> str:
    if action.get("action_key"):
      return str(action["action_key"])
    action_type = action.get("type")
    if action_type == "choose_event_option":
      return f"choose_event_option:{action.get('index')}"
    if action_type == "select_card":
      return f"select_card:{action.get('index')}"
    if action_type in {"advance_dialogue", "confirm_selection", "cancel_selection"}:
      return action_type
    return "advance_dialogue"

  # --- candidate builders ------------------------------------------------
  def _event_candidates(self, raw_state: dict) -> list[dict]:
    event = raw_state.get("event", {})
    if event.get("in_dialogue"):
      return [self._candidate({"type": "advance_dialogue"}, "advance_dialogue")]

    candidates = []
    for fallback_index, option in enumerate(event.get("options", [])):
      if option.get("is_locked", False):
        continue
      index = self._parse_int(option.get("index", fallback_index), fallback_index)
      candidates.append(self._candidate(
        {"type": "choose_event_option", "index": index},
        f"choose_event_option:{index}",
      ))
    return candidates

  def _card_select_candidates(self, raw_state: dict) -> list[dict]:
    card_select = raw_state.get("card_select", {})
    cards = card_select.get("cards", [])
    selected_indices = selected_card_indices(card_select)
    selected_count = selection_selected_count(raw_state, "card_select", len(selected_indices))

    candidates = []
    if can_select_more(raw_state, "card_select", selected_count):
      for fallback_index, card in enumerate(cards):
        index = self._parse_int(card.get("index", fallback_index), fallback_index)
        if index in selected_indices:
          continue
        candidates.append(self._candidate(
          {"type": "select_card", "index": index},
          f"select_card:{index}",
        ))
    if can_confirm_selection(raw_state, "card_select", selected_count):
      candidates.append(self._candidate({"type": "confirm_selection"}, "confirm_selection"))
    if card_select.get("can_cancel", False):
      candidates.append(self._candidate({"type": "cancel_selection"}, "cancel_selection"))
    return candidates

  def _event_option(self, raw_state: dict, index: int) -> dict | None:
    for fallback_index, option in enumerate(raw_state.get("event", {}).get("options", [])):
      if self._parse_int(option.get("index", fallback_index), fallback_index) == index:
        return option
    return None
