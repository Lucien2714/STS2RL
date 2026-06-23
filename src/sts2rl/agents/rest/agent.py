"""Trainable candidate-action agents for the rest-site screen.

Thin per-screen subclasses that bind
:class:`~sts2rl.encoders.rest_encoder.RestEncoder` to the shared DQN/PPO
candidate-action agents. The encoder owns all rest-specific encoding and
candidate logic.
"""

from __future__ import annotations

from sts2rl.agents.candidate_dqn_agent import DQNCandidateAgent
from sts2rl.agents.candidate_ppo_agent import PPOCandidateAgent
from sts2rl.encoders.rest_encoder import RestEncoder


class RestDQNAgent(DQNCandidateAgent):
    """DQN candidate-action agent for the rest-site screen."""

    def __init__(self, **kwargs):
        super().__init__(encoder=RestEncoder(), **kwargs)


class RestPPOAgent(PPOCandidateAgent):
    """PPO candidate-action agent for the rest-site screen."""

    def __init__(self, **kwargs):
        super().__init__(encoder=RestEncoder(), **kwargs)
