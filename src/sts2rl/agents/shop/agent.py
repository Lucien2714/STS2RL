"""Trainable candidate-action agents for the shop screen.

Thin per-screen subclasses that bind
:class:`~sts2rl.encoders.shop_encoder.ShopEncoder` to the shared DQN/PPO
candidate-action agents. The encoder owns all shop-specific encoding and
candidate logic.
"""

from __future__ import annotations

from sts2rl.action_spaces.shop import ShopActionSpace
from sts2rl.agents.candidate_dqn_agent import DQNCandidateAgent
from sts2rl.agents.candidate_ppo_agent import PPOCandidateAgent
from sts2rl.encoders.shop_encoder import ShopEncoder


class ShopDQNAgent(DQNCandidateAgent):
    """DQN candidate-action agent for the shop screen."""

    def __init__(self, **kwargs):
        super().__init__(encoder=ShopEncoder(), action_space=ShopActionSpace(), **kwargs)


class ShopPPOAgent(PPOCandidateAgent):
    """PPO candidate-action agent for the shop screen."""

    def __init__(self, **kwargs):
        super().__init__(encoder=ShopEncoder(), action_space=ShopActionSpace(), **kwargs)
