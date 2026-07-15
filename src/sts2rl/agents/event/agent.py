"""Trainable candidate-action agents for the event screen.

Thin per-screen subclasses that bind
:class:`~sts2rl.encoders.event_encoder.EventEncoder` to the shared DQN/PPO
candidate-action agents. The encoder owns all event-specific encoding and
candidate logic.
"""

from __future__ import annotations

from sts2rl.action_spaces.event import EventActionSpace
from sts2rl.agents.candidate_dqn_agent import DQNCandidateAgent
from sts2rl.agents.candidate_ppo_agent import PPOCandidateAgent
from sts2rl.encoders.event_encoder import EventEncoder


class EventDQNAgent(DQNCandidateAgent):
    """DQN candidate-action agent for the event screen."""

    def __init__(self, **kwargs):
        super().__init__(encoder=EventEncoder(), action_space=EventActionSpace(), **kwargs)


class EventPPOAgent(PPOCandidateAgent):
    """PPO candidate-action agent for the event screen."""

    def __init__(self, **kwargs):
        super().__init__(encoder=EventEncoder(), action_space=EventActionSpace(), **kwargs)
