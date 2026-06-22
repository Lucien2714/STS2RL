"""Learned per-card embedding (paper-style: one low-dimensional vector per card).

``CardModelEncoder`` maps a single card to a fixed-width learned embedding using
only the card's id, ``is_upgraded`` flag, and enchantment. Card id and
enchantment are categorical, so each is looked up in a learnable ``nn.Embedding``
table (the table rows *are* the embedding, trained by backprop); ``is_upgraded``
is binary, so it is fed straight in as a 0/1 scalar. The factor vectors are
concatenated and projected to ``out_dim``.

This is a learned ``nn.Module`` and therefore lives here (reusable model
components) rather than in the model-free ``encoders/`` package.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from sts2rl.data.card import Card, CardIdentity
from sts2rl.data.loader import (
    get_card_index,
    get_card_map_size,
    get_data_index_or_default,
    get_data_map_size,
)

# Index 0 of each embedding table is reserved for the unknown/none row; real
# DataIdMap indices are shifted up by one.
_RESERVED = 1


class CardModelEncoder(nn.Module):
    """Encode a card as a learned ``out_dim``-d vector from id/upgrade/enchantment."""

    SCHEMA = "card_model_v1"

    def __init__(
        self,
        out_dim: int = 16,
        id_dim: int = 16,
        ench_dim: int = 8,
        device=None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.out_dim = out_dim

        # Vocabulary sizes are read from the bundled static data once, so the
        # table shapes are reproducible across runs/checkpoints.
        self.num_cards = get_card_map_size()
        self.num_enchantments = get_data_map_size("enchantments")

        self.id_emb = nn.Embedding(self.num_cards + _RESERVED, id_dim)
        self.ench_emb = nn.Embedding(self.num_enchantments + _RESERVED, ench_dim)
        self.project = nn.Linear(id_dim + ench_dim + 1, out_dim)  # +1 = upgrade flag

        self.to(self.device)

    def forward(
        self,
        card_idx: torch.Tensor,
        ench_idx: torch.Tensor,
        upgraded: torch.Tensor,
    ) -> torch.Tensor:
        """Embed a batch of cards given their factor indices.

        ``card_idx`` / ``ench_idx`` are long index tensors of shape ``[...]``;
        ``upgraded`` is a 0/1 tensor of shape ``[...]``. Returns ``[..., out_dim]``.
        """
        features = torch.cat(
            [
                self.id_emb(card_idx),
                self.ench_emb(ench_idx),
                upgraded.unsqueeze(-1).float(),
            ],
            dim=-1,
        )
        return self.project(features)

    def save(self, path: str | Path) -> None:
        """Persist the embedding weights and table configuration to a checkpoint."""
        torch.save(
            {
                "schema": self.SCHEMA,
                "config": self._config(),
                "state_dict": self.state_dict(),
            },
            str(path),
        )

    def load(self, path: str | Path) -> None:
        """Load weights from a checkpoint, validating schema and table shapes.

        Raises ``ValueError`` if the checkpoint's schema or table configuration
        (embedding dims / card / enchantment vocabulary sizes) differs, since the
        embedding rows are keyed by card index and would otherwise mis-map.
        """
        checkpoint = torch.load(str(path), map_location=self.device)
        schema = checkpoint.get("schema")
        if schema != self.SCHEMA:
            raise ValueError(
                f"Incompatible card-model checkpoint schema={schema!r}; expected {self.SCHEMA!r}."
            )
        config = checkpoint.get("config", {})
        if config != self._config():
            raise ValueError(
                f"Incompatible card-model config {config!r}; expected {self._config()!r}. "
                "The embedding dims or card/enchantment vocabulary differ."
            )
        self.load_state_dict(checkpoint["state_dict"])

    def _config(self) -> dict:
        """Return the table configuration a checkpoint must match to load."""
        return {
            "out_dim": self.out_dim,
            "id_dim": self.id_emb.embedding_dim,
            "ench_dim": self.ench_emb.embedding_dim,
            "num_cards": self.num_cards,
            "num_enchantments": self.num_enchantments,
        }

    def encode_card(self, card: Card | CardIdentity | dict) -> torch.Tensor:
        """Encode one card object into its ``out_dim``-d embedding (shape ``[out_dim]``)."""
        card_idx, ench_idx, upgraded = self.card_to_indices(card)
        return self.forward(
            torch.tensor([card_idx], dtype=torch.long, device=self.device),
            torch.tensor([ench_idx], dtype=torch.long, device=self.device),
            torch.tensor([upgraded], dtype=torch.float, device=self.device),
        ).squeeze(0)

    def card_to_indices(self, card: Card | CardIdentity | dict) -> tuple[int, int, int]:
        """Resolve a card object to ``(card_idx, ench_idx, upgraded)`` factor inputs.

        ``card_idx`` / ``ench_idx`` are embedding-table rows (0 = unknown/none);
        ``upgraded`` is 0 or 1. Unknown ids fall back to the reserved row 0.
        """
        identity = self._coerce_identity(card)
        card_idx = get_card_index(identity.card_id, default=-1) + _RESERVED
        ench_idx = (
            get_data_index_or_default("enchantments", identity.enchantment_id, -1) + _RESERVED
        )
        upgraded = 1 if identity.upgrade_level > 0 else 0
        return card_idx, ench_idx, upgraded

    @staticmethod
    def _coerce_identity(card: Card | CardIdentity | dict) -> CardIdentity:
        if isinstance(card, CardIdentity):
            return card
        if isinstance(card, Card):
            return card.identity
        if isinstance(card, dict):
            return CardIdentity.from_raw(card)
        raise TypeError(f"Unsupported card type for encoding: {type(card).__name__}")
