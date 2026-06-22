"""Screen-agnostic base for trainable candidate-action agents.

A candidate-action agent scores a *set of legal action candidates* for one screen.
It composes a screen-specific **encoder** (state/action encoding + legal-action
enumeration; see ``encoders/``) and owns device resolution and the shared training
counters. Concrete DQN/PPO subclasses supply the network, optimizer, and the
``choose_action`` / ``remember`` / ``train_step`` / ``save`` / ``load`` /
``new_rollout`` / ``buffered_steps`` / ``current_q_values`` / ``bc_score`` methods.

Battle was the first such screen; the same machinery now backs the non-battle
screens (map, rewards, shop, rest, event), each with its own encoder.
"""

from __future__ import annotations

import torch

from sts2rl.agents.base import TrainableScreenAgent


class CandidateActionAgent(TrainableScreenAgent):
  """Base for agents that score a set of legal action candidates for one screen."""

  def __init__(self, encoder, device=None):
    self.encoder = encoder
    self.device = self._resolve_device(device)
    self.epsilon = 0.0
    self.trained_steps = 0
    self.learn_steps = 0
    self.last_action_selection = {}

  # --- delegation to the composed encoder (kept on the agent for API parity) ---
  @property
  def ACTION_TYPES(self) -> tuple:
    return self.encoder.ACTION_TYPES

  @property
  def state_size(self) -> int:
    return self.encoder.state_size

  @property
  def action_feature_size(self) -> int:
    return self.encoder.action_feature_size

  @property
  def model_input_size(self) -> int:
    return self.encoder.model_input_size

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
