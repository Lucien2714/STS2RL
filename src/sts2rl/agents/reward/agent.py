"""Trainable candidate-action agents for the reward screens.

Thin per-screen subclasses that bind
:class:`~sts2rl.encoders.reward_encoder.RewardEncoder` (rewards / card_reward /
treasure) to the shared DQN/PPO candidate-action agents. The encoder owns all
reward-specific encoding and candidate logic.
"""

from __future__ import annotations

from sts2rl.action_spaces.reward import RewardActionSpace
from sts2rl.agents.candidate_dqn_agent import DQNCandidateAgent
from sts2rl.agents.candidate_ppo_agent import PPOCandidateAgent
from sts2rl.encoders.reward_encoder import RewardEncoder


class RewardDQNAgent(DQNCandidateAgent):
    """DQN candidate-action agent for the reward screens."""

    def __init__(self, **kwargs):
        super().__init__(encoder=RewardEncoder(), action_space=RewardActionSpace(), **kwargs)


class RewardPPOAgent(PPOCandidateAgent):
    """PPO candidate-action agent for the reward screens."""

    def __init__(self, **kwargs):
        super().__init__(encoder=RewardEncoder(), action_space=RewardActionSpace(), **kwargs)
