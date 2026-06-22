"""Shared plumbing for non-battle screen encoders.

A screen encoder turns a raw STS2MCP state into a fixed-width state vector plus a
set of legal candidate actions (each with a stable ``action_key`` and an action
feature vector). Concrete encoders implement ``valid_action_candidates``,
``encode_state``, ``encode_action``, and ``action_key``; this base supplies the
candidate plumbing and small numeric helpers.

(The battle encoder predates this base and keeps its own copies of these helpers;
it is intentionally not migrated to avoid touching the trained battle schema.)
"""

from __future__ import annotations


class CandidateEncoder:
    """Base for candidate-action screen encoders."""

    ACTION_TYPES: tuple[str, ...] = ()
    SCHEMA: str = "screen"
    DQN_SCHEMA: str = "screen_dqn_v1"
    PPO_SCHEMA: str = "screen_ppo_v1"

    def candidate_action_vectors(self, raw_state: dict) -> list[list[float]]:
        """Encode every currently legal action candidate."""
        return [
            self.encode_action(raw_state, candidate["action"])
            for candidate in self.valid_action_candidates(raw_state)
        ]

    def valid_action_mask(self, raw_state: dict) -> list[bool]:
        """Return an all-true mask sized to the candidate list (API parity)."""
        return [True for _ in self.valid_action_candidates(raw_state)]

    def _fallback_action(self, raw_state: dict) -> dict:
        """Safe default when no candidates exist; the orchestrator prefers the
        screen's rule-based policy for the real fallback."""
        return {"type": "proceed"}

    def _candidate(self, action: dict, key: str) -> dict:
        action = dict(action)
        action["action_key"] = key
        return {"action": action, "action_key": key}

    def _public_action(self, candidate: dict) -> dict:
        return dict(candidate["action"])

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
