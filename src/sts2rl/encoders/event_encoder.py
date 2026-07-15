"""Event featurizer: encode event screens and non-battle card-selection overlays."""

from __future__ import annotations

from sts2rl.action_spaces.event import event_option
from sts2rl.action_spaces.selection import remaining_to_min
from sts2rl.encoders.base import CandidateEncoder

# Schema-critical: the tuple order defines the encode_action one-hot layout.
EVENT_ACTION_TYPES = (
    "choose_event_option",
    "advance_dialogue",
    "select_card",
    "confirm_selection",
    "cancel_selection",
)


class EventEncoder(CandidateEncoder):
    """Encode event / non-battle card_select screen states and actions."""

    ACTION_TYPES = EVENT_ACTION_TYPES
    SCHEMA = "event"
    DQN_SCHEMA = "event_dqn_v1"
    PPO_SCHEMA = "event_ppo_v1"

    MAX_OPTIONS = 12
    MAX_CARDS = 20

    def __init__(self):
        self.action_feature_size = (
            len(self.ACTION_TYPES)  # action-type one-hot
            + 1  # option/card index (scaled)
            + 1  # is_proceed flag (event option)
        )
        self.state_size = (
            6  # hp_ratio, hp, max_hp, gold, floor, act
            + 6  # in_dialogue, is_card_select, n_options, n_unlocked, n_cards, remaining_to_min
        )
        self.model_input_size = self.state_size + self.action_feature_size

    def encode_state(self, raw_state: dict, action_mask=None) -> list[float]:
        features = self._player_run_features(raw_state)
        event = raw_state.get("event", {})
        card_select = raw_state.get("card_select", {})
        is_card_select = raw_state.get("state_type") == "card_select"
        options = event.get("options", [])
        unlocked = [option for option in options if not option.get("is_locked", False)]
        features.extend(
            [
                1.0 if event.get("in_dialogue") else 0.0,
                1.0 if is_card_select else 0.0,
                self._scale(len(options), self.MAX_OPTIONS),
                self._scale(len(unlocked), self.MAX_OPTIONS),
                self._scale(len(card_select.get("cards", [])), self.MAX_CARDS),
                self._scale(remaining_to_min(raw_state, "card_select") or 0, self.MAX_CARDS),
            ]
        )
        return [float(value) for value in features]

    def encode_action(self, raw_state: dict, action: dict) -> list[float]:
        action_type = action.get("type")
        features = self._action_type_features(action_type)
        if action_type == "choose_event_option":
            option = event_option(raw_state, self._parse_int(action.get("index", -1)))
            features.append(self._scale(action.get("index", 0), self.MAX_OPTIONS))
            features.append(1.0 if (option or {}).get("is_proceed") else 0.0)
        elif action_type == "select_card":
            features.append(self._scale(action.get("index", 0), self.MAX_CARDS))
            features.append(0.0)
        else:
            features.extend([0.0, 0.0])
        return [float(value) for value in features]
