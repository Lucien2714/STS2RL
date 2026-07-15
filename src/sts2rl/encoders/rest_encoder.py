"""Rest-site featurizer: encode rest-site state and option-choice actions."""

from __future__ import annotations

from sts2rl.action_spaces.rest import enabled_options, option_by_index
from sts2rl.encoders.base import CandidateEncoder

# Coarse vocabulary of rest-site option kinds (matched as substrings of id/name).
REST_OPTION_VOCAB = ("rest", "heal", "smith", "upgrade", "dig", "lift", "recall", "toke")

# Schema-critical: the tuple order defines the encode_action one-hot layout.
REST_ACTION_TYPES = ("choose_rest_option", "proceed")


class RestEncoder(CandidateEncoder):
    """Encode rest-site state and the enabled rest-option actions."""

    ACTION_TYPES = REST_ACTION_TYPES
    SCHEMA = "rest"
    DQN_SCHEMA = "rest_dqn_v1"
    PPO_SCHEMA = "rest_ppo_v1"

    MAX_OPTIONS = 8

    def __init__(self):
        self.option_vocab_size = len(REST_OPTION_VOCAB)
        self.action_feature_size = (
            len(self.ACTION_TYPES)  # action-type one-hot
            + self.option_vocab_size  # chosen option kind one-hot
            + 1  # option index (scaled)
        )
        self.state_size = (
            6  # hp_ratio, hp, max_hp, gold, floor, act
            + self.option_vocab_size  # available option-kind presence
            + 1  # number of enabled options (scaled)
        )
        self.model_input_size = self.state_size + self.action_feature_size

    def encode_state(self, raw_state: dict, action_mask=None) -> list[float]:
        features = self._player_run_features(raw_state)
        presence = [0.0 for _ in REST_OPTION_VOCAB]
        options = enabled_options(raw_state)
        for option in options:
            for index, value in enumerate(
                self._onehot(self._option_kind(option), REST_OPTION_VOCAB)
            ):
                if value:
                    presence[index] = 1.0
        features.extend(presence)
        features.append(self._scale(len(options), self.MAX_OPTIONS))
        return [float(value) for value in features]

    def encode_action(self, raw_state: dict, action: dict) -> list[float]:
        action_type = action.get("type")
        features = self._action_type_features(action_type)
        if action_type == "choose_rest_option":
            option = option_by_index(raw_state, self._parse_int(action.get("index", -1)))
            features.extend(self._onehot(self._option_kind(option), REST_OPTION_VOCAB))
            features.append(self._scale(action.get("index", 0), self.MAX_OPTIONS))
        else:
            features.extend([0.0] * self.option_vocab_size)
            features.append(0.0)
        return [float(value) for value in features]

    def _option_kind(self, option: dict | None) -> str:
        if not option:
            return ""
        return f"{option.get('id', '')} {option.get('name', '')}"
