"""Battle featurizer that also emits raw card-index channels for learned embeddings.

The handcrafted :class:`~sts2rl.encoders.battle_encoder.BattleStateEncoder` squashes
a card's identity into a single scaled float, which a network can only treat as an
ordinal. A learned per-card embedding
(:class:`~sts2rl.models.card_encoder.CardModelEncoder`) needs the *index* instead,
so this subclass appends integer factor channels to the end of the state and
action vectors:

* state: ``MAX_HAND`` slots × ``(card_index, enchantment_index, upgraded)``
* action: one such triple for the card the action refers to (zeros otherwise)

Everything else is byte-identical to the handcrafted layout, so the flat
``Tensor[K, state ⧺ action]`` contract is unchanged and replay buffers, PPO
rollouts, and behavioral cloning all work without modification. The learned policy
modules slice these trailing columns back out — see
:class:`~sts2rl.models.learned_policies.CardIndexLayout`.

This module stays torch-free like the rest of ``encoders/``: it only writes index
numbers, using the shared convention in :mod:`sts2rl.data.card`.
"""

from __future__ import annotations

from sts2rl.action_spaces.battle import action_item
from sts2rl.data.card import card_factor_indices
from sts2rl.encoders.battle_encoder import BattleStateEncoder

# Factors per card: (card_index, enchantment_index, upgraded).
CARD_FACTOR_CHANNELS = 3

# Action types whose target is a card, and therefore carry a card embedding.
CARD_BEARING_ACTION_TYPES = frozenset({"play_card", "combat_select_card", "select_card"})


class LearnedBattleStateEncoder(BattleStateEncoder):
    """Handcrafted battle features plus trailing card-index channels."""

    # A distinct schema: these vectors are wider and laid out differently, so
    # checkpoint loads correctly refuse to mix them with handcrafted ones.
    DQN_SCHEMA = "candidate_action_learned_v1"
    PPO_SCHEMA = "candidate_action_learned_ppo_v1"
    ACTION_SCHEMA = "candidate_action_learned_v1"

    def __init__(self):
        super().__init__()
        # Where the trailing index blocks begin, for the policy module to slice.
        self.state_feature_size = self.state_size
        self.state_card_index_offset = self.state_size
        self.action_feature_offset = self.action_feature_size

        self.state_size += self.MAX_HAND * CARD_FACTOR_CHANNELS
        self.action_feature_size += CARD_FACTOR_CHANNELS
        self.model_input_size = self.state_size + self.action_feature_size

    def encode_state(self, raw_state: dict, action_mask=None) -> list[float]:
        """Encode battle state, appending one index triple per hand slot."""
        features = super().encode_state(raw_state, action_mask)

        player = raw_state.get("player", {})
        hand = player.get("hand", raw_state.get("hand", []))
        for card_index in range(self.MAX_HAND):
            card = hand[card_index] if card_index < len(hand) else None
            features.extend(self._card_index_features(card))

        return features

    def encode_action(self, raw_state: dict, action: dict) -> list[float]:
        """Encode an action, appending the index triple for the card it plays."""
        features = super().encode_action(raw_state, action)

        card = None
        if action.get("type") in CARD_BEARING_ACTION_TYPES:
            card = action_item(action, raw_state)
        features.extend(self._card_index_features(card))

        return features

    def _card_index_features(self, card: dict | None) -> list[float]:
        """Return ``(card_index, enchantment_index, upgraded)`` as floats.

        Row 0 is the reserved unknown/none row, so an empty slot is all zeros.
        These are integers carried in a float tensor; the policy casts them back
        to long before the embedding lookup.
        """
        if not card:
            return [0.0] * CARD_FACTOR_CHANNELS
        return [float(value) for value in card_factor_indices(card)]
