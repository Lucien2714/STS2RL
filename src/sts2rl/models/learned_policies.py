"""Policy modules with a learned card embedding inside them (ADR-0007 Phase 3).

These satisfy the same ``score(state_action) -> Tensor[K]`` contract as the
handcrafted policies in :mod:`sts2rl.models.policies`, so DQN, PPO, and
behavioral cloning drive them unchanged. The difference is internal: the trailing
card-index columns written by
:class:`~sts2rl.encoders.learned_battle_encoder.LearnedBattleStateEncoder` are
sliced out, looked up in a :class:`~sts2rl.models.card_encoder.CardModelEncoder`,
and concatenated back with the handcrafted features before scoring.

This is the payoff ADR-0004 and ADR-0007 were pointing at: because the embedding
is a submodule of the policy, its parameters are already covered by the agent's
``Adam(policy.parameters())`` — it trains end-to-end under both BC and RL with no
special-casing in either update rule.

Checkpoint note: the attribute names (``card_encoder``, ``net``, ``actor``,
``critic``) define the ``state_dict`` keys. Pinned by ``tests/test_policies.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from sts2rl.models.card_encoder import CardModelEncoder
from sts2rl.models.policies import layer_init


@dataclass(frozen=True)
class CardIndexLayout:
    """Where the card-index channels sit inside a ``[state ⧺ action]`` row.

    Built from a :class:`~sts2rl.encoders.learned_battle_encoder.LearnedBattleStateEncoder`
    via :meth:`from_encoder`; kept as plain numbers so the policy has no import
    dependency on the encoder package.
    """

    state_feature_size: int  # handcrafted state features, before the index block
    state_size: int  # full state width, including the index block
    action_feature_offset: int  # handcrafted action features, relative to action start
    action_feature_size: int  # full action width, including its index triple
    hand_slots: int
    channels: int = 3

    @classmethod
    def from_encoder(cls, encoder) -> "CardIndexLayout":
        """Read the layout off a learned battle encoder."""
        return cls(
            state_feature_size=encoder.state_feature_size,
            state_size=encoder.state_size,
            action_feature_offset=encoder.action_feature_offset,
            action_feature_size=encoder.action_feature_size,
            hand_slots=encoder.MAX_HAND,
        )

    @property
    def model_input_size(self) -> int:
        """Return the full width of one encoded state/action row."""
        return self.state_size + self.action_feature_size


class _CardEmbeddingFeaturizer(nn.Module):
    """Split encoded rows into handcrafted features + learned card embeddings."""

    def __init__(self, layout: CardIndexLayout, card_dim: int = 16):
        super().__init__()
        self.layout = layout
        self.card_encoder = CardModelEncoder(out_dim=card_dim)
        self.card_dim = card_dim

    @property
    def state_out_size(self) -> int:
        """Return the width of an embedded state (handcrafted + pooled hand)."""
        return self.layout.state_feature_size + self.card_dim

    @property
    def out_size(self) -> int:
        """Return the width of an embedded state/action row."""
        return self.state_out_size + self.layout.action_feature_offset + self.card_dim

    def embed_state(self, states: torch.Tensor) -> torch.Tensor:
        """Embed state rows ``[N, state_size]`` into ``[N, state_out_size]``."""
        layout = self.layout
        continuous = states[:, : layout.state_feature_size]
        indices = states[:, layout.state_feature_size : layout.state_size]
        indices = indices.reshape(states.shape[0], layout.hand_slots, layout.channels)

        hand = self._embed_indices(indices)
        # Mean over occupied hand slots only: empty slots are the reserved row 0
        # and would otherwise drag every hand toward the "unknown card" vector.
        occupied = (indices[..., 0] > 0).float().unsqueeze(-1)
        pooled = (hand * occupied).sum(dim=1) / occupied.sum(dim=1).clamp(min=1.0)

        return torch.cat([continuous, pooled], dim=-1)

    def forward(self, state_action: torch.Tensor) -> torch.Tensor:
        """Embed full ``[K, model_input_size]`` rows into ``[K, out_size]``."""
        layout = self.layout
        state_part = self.embed_state(state_action[:, : layout.state_size])

        action = state_action[:, layout.state_size :]
        action_continuous = action[:, : layout.action_feature_offset]
        action_indices = action[:, layout.action_feature_offset :]
        action_card = self._embed_indices(action_indices.unsqueeze(1)).squeeze(1)
        # An action with no card carries the reserved row; zero it so "no card"
        # is distinct from "the unknown card".
        action_card = action_card * (action_indices[:, :1] > 0).float()

        return torch.cat([state_part, action_continuous, action_card], dim=-1)

    def _embed_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Embed ``[..., 3]`` float index channels via the card embedding tables."""
        card_idx = indices[..., 0].long().clamp(min=0, max=self.card_encoder.num_cards)
        ench_idx = indices[..., 1].long().clamp(min=0, max=self.card_encoder.num_enchantments)
        upgraded = indices[..., 2]
        return self.card_encoder(card_idx, ench_idx, upgraded)


class LearnedCandidateQNetwork(nn.Module):
    """Q-network scoring candidates over handcrafted + learned card features."""

    def __init__(self, layout: CardIndexLayout, hidden_size: int = 256, card_dim: int = 16):
        super().__init__()
        self.featurizer = _CardEmbeddingFeaturizer(layout, card_dim=card_dim)
        self.net = nn.Sequential(
            nn.Linear(self.featurizer.out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state_action):
        """Return one Q-value for each encoded state/action pair."""
        return self.net(self.featurizer(state_action)).squeeze(-1)

    def score(self, state_action):
        """Policy-module contract: one differentiable score per row (the Q-value)."""
        return self.forward(state_action)


class LearnedCandidatePPOPolicy(nn.Module):
    """Actor-critic over handcrafted + learned card features."""

    def __init__(self, layout: CardIndexLayout, hidden_size: int = 256, card_dim: int = 16):
        super().__init__()
        self.featurizer = _CardEmbeddingFeaturizer(layout, card_dim=card_dim)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.featurizer.out_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.featurizer.state_out_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

    def action_logits(self, state_actions):
        """Return one logit per encoded state/action candidate."""
        return self.actor(self.featurizer(state_actions)).squeeze(-1)

    def score(self, state_actions):
        """Policy-module contract: one differentiable score per row (the logit)."""
        return self.action_logits(state_actions)

    def value(self, states):
        """Return state values; the critic shares the same card embedding."""
        return self.critic(self.featurizer.embed_state(states)).squeeze(-1)
