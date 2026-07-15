"""Shared plumbing for non-battle screen featurizers.

A screen encoder turns a raw STS2MCP state or action into a fixed-width feature
vector. Concrete encoders implement ``encode_state`` and ``encode_action``; this
base supplies small numeric helpers and the shared player/run feature prefix.
Legal-action enumeration lives in the matching :mod:`sts2rl.action_spaces`
class; the candidate-action agents compose one of each.

(The battle encoder predates this base and keeps its own copies of these helpers;
it is intentionally not migrated to avoid touching the trained battle schema.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class CandidateEncoder(ABC):
    """Base for candidate-action screen featurizers."""

    ACTION_TYPES: tuple[str, ...] = ()
    SCHEMA: str = "screen"
    DQN_SCHEMA: str = "screen_dqn_v1"
    PPO_SCHEMA: str = "screen_ppo_v1"

    # --- primitives every concrete encoder must implement ------------------
    @abstractmethod
    def encode_state(self, raw_state: dict, action_mask=None) -> list[float]:
        """Encode the screen state as a fixed-width vector of length ``state_size``."""

    @abstractmethod
    def encode_action(self, raw_state: dict, action: dict) -> list[float]:
        """Encode one action as a fixed-width vector of length ``action_feature_size``."""

    # --- shared numeric helpers ---------------------------------------------
    def _action_type_features(self, action_type: str | None) -> list[float]:
        return [1.0 if action_type == name else 0.0 for name in self.ACTION_TYPES]

    def _onehot(self, value: str | None, vocab: tuple[str, ...]) -> list[float]:
        """One-hot a (substring-matched) value against a fixed vocabulary."""
        features = [0.0 for _ in vocab]
        token = str(value or "").strip().lower()
        for index, name in enumerate(vocab):
            if name in token:
                features[index] = 1.0
                return features
        return features

    @staticmethod
    def _parse_int(value: object, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _scale(self, value: object, denominator: int | float) -> float:
        denominator = max(float(denominator), 1.0)
        return max(0.0, min(float(self._parse_int(value)) / denominator, 1.0))

    def _player_run_features(self, raw_state: dict) -> list[float]:
        """Six shared scalar features common to every screen: hp_ratio, hp, max_hp,
        gold, floor, act. Every screen encoder prefixes its state vector with these."""
        player = raw_state.get("player", {})
        run = raw_state.get("run", {})
        hp = self._parse_int(player.get("hp", player.get("current_hp", 0)))
        max_hp = max(1, self._parse_int(player.get("max_hp", 1)))
        return [
            max(0.0, min(hp / max_hp, 1.0)),
            self._scale(hp, max_hp),
            self._scale(max_hp, 200),
            self._scale(player.get("gold", 0), 999),
            self._scale(run.get("floor", 0), 60),
            self._scale(run.get("act", 0), 4),
        ]
