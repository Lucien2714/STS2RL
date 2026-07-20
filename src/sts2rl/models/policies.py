"""Candidate-scoring policy modules (the swappable, trainable "brains").

A policy module maps encoded feature vectors to per-candidate scores: Q-values
for DQN, actor logits (plus a state value) for PPO. The shared contract is
``score(state_action: Tensor[K, D]) -> Tensor[K]`` — differentiable, one score
per encoded state/action row — which is also the single primitive behavioral
cloning needs. Actor-critic policies additionally expose
``value(states) -> Tensor``.

The algorithm agents (``agents/candidate_dqn_agent.py`` /
``candidate_ppo_agent.py``) construct these by default but accept any module
satisfying the contract via their ``policy`` argument, so alternative
architectures (or learned featurizers such as
:class:`~sts2rl.models.card_encoder.CardModelEncoder`-based ones) can be swapped
in without touching the update rules.

Checkpoint note: the attribute names (``net``, ``actor``, ``critic``) and layer
stacks are load-bearing — they define the ``state_dict`` keys stored in existing
checkpoints. Do not rename or renest them.
"""

from __future__ import annotations

from torch import nn


def layer_init(layer: nn.Linear, std: float = 1.0, bias_const: float = 0.0) -> nn.Linear:
    """Initialize a linear layer following the small CleanRL PPO convention."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class CandidateQNetwork(nn.Module):
    """Small fully connected Q-network for state/action candidate scoring."""

    def __init__(self, input_size: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state_action):
        """Return one Q-value for each encoded state/action pair."""
        return self.net(state_action).squeeze(-1)

    def score(self, state_action):
        """Policy-module contract: one differentiable score per row (the Q-value)."""
        return self.forward(state_action)


class CandidatePPOPolicy(nn.Module):
    """Actor-critic network for variable candidate-action sets."""

    def __init__(self, state_size: int, state_action_size: int, hidden_size: int = 256):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_action_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

    def action_logits(self, state_actions):
        """Return one logit per encoded state/action candidate."""
        return self.actor(state_actions).squeeze(-1)

    def score(self, state_actions):
        """Policy-module contract: one differentiable score per row (the logit)."""
        return self.action_logits(state_actions)

    def value(self, states):
        """Return state values."""
        return self.critic(states).squeeze(-1)
