"""Shared base for trainable candidate-action battle agents."""

from __future__ import annotations

import torch

from sts2rl.agents.base import BattleAgent
from sts2rl.encoders.battle_encoder import BattleStateEncoder


class CandidateActionAgent(BattleAgent):
  """Base for battle agents that score legal action candidates.

  Owns the shared, model-free machinery: a composed :class:`BattleStateEncoder`
  (state/action encoding + legal-action enumeration), device resolution, and the
  common training counters. DQN and PPO compose this as siblings rather than one
  subclassing the other; each subclass supplies its own network, optimizer, and
  ``choose_action`` / ``remember`` / ``train_step`` / ``save`` / ``load`` /
  ``new_rollout`` / ``buffered_steps`` / ``current_q_values`` / ``bc_score``.
  """

  ACTION_SCHEMA = BattleStateEncoder.ACTION_SCHEMA
  ACTION_TYPES = BattleStateEncoder.ACTION_TYPES

  def __init__(self, device=None):
    self.encoder = BattleStateEncoder()
    self.device = self._resolve_device(device)
    self.epsilon = 0.0
    self.trained_steps = 0
    self.learn_steps = 0
    self.last_action_selection = {}

  # --- dimension delegation (kept on the agent for API compatibility) ---
  @property
  def state_size(self) -> int:
    return self.encoder.state_size

  @property
  def action_feature_size(self) -> int:
    return self.encoder.action_feature_size

  @property
  def model_input_size(self) -> int:
    return self.encoder.model_input_size

  # --- encoder delegation ------------------------------------------------
  def encode_state(self, raw_state: dict, action_mask=None) -> list[float]:
    return self.encoder.encode_state(raw_state, action_mask)

  def encode_action(self, raw_state: dict, action: dict) -> list[float]:
    return self.encoder.encode_action(raw_state, action)

  def candidate_action_vectors(self, raw_state: dict) -> list[list[float]]:
    return self.encoder.candidate_action_vectors(raw_state)

  def valid_action_candidates(self, raw_state: dict) -> list[dict]:
    return self.encoder.valid_action_candidates(raw_state)

  def valid_action_mask(self, raw_state: dict) -> list[bool]:
    return self.encoder.valid_action_mask(raw_state)

  def action_key(self, action: dict, raw_state: dict | None = None) -> str:
    return self.encoder.action_key(action, raw_state)

  def _fallback_action(self, raw_state: dict) -> dict:
    return self.encoder._fallback_action(raw_state)

  def _public_action(self, candidate: dict) -> dict:
    return self.encoder._public_action(candidate)

  # --- shared helpers ----------------------------------------------------
  def _resolve_device(self, device):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

  @staticmethod
  def _safe_float(value: float) -> float | None:
    try:
      value = float(value)
    except (TypeError, ValueError):
      return None
    return value if value == value and value not in {float("inf"), float("-inf")} else None
