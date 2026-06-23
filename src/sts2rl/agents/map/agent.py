"""Trainable candidate-action agents for the map screen.

Thin per-screen subclasses that bind :class:`~sts2rl.encoders.map_encoder.MapEncoder`
to the shared DQN/PPO candidate-action agents. The encoder owns all map-specific
encoding and candidate logic; these classes give the map agent a discoverable,
importable type (and a home for any future map-specific hyperparameters).
"""

from __future__ import annotations

from sts2rl.agents.candidate_dqn_agent import DQNCandidateAgent
from sts2rl.agents.candidate_ppo_agent import PPOCandidateAgent
from sts2rl.encoders.map_encoder import MapEncoder


class MapDQNAgent(DQNCandidateAgent):
    """DQN candidate-action agent for the map screen."""

    def __init__(self, **kwargs):
        super().__init__(encoder=MapEncoder(), **kwargs)


class MapPPOAgent(PPOCandidateAgent):
    """PPO candidate-action agent for the map screen."""

    def __init__(self, **kwargs):
        super().__init__(encoder=MapEncoder(), **kwargs)
