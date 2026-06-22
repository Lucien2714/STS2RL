"""Tests for the learned per-card embedding (models.card_encoder)."""

import torch

from sts2rl.data.card import Card, CardIdentity
from sts2rl.data.loader import DataIdMap, get_card_index, get_data_index_or_default
from sts2rl.models.card_encoder import CardModelEncoder


def make_encoder(out_dim: int = 16) -> CardModelEncoder:
    torch.manual_seed(0)
    return CardModelEncoder(out_dim=out_dim, device="cpu")


def test_encode_card_returns_out_dim_vector():
    encoder = make_encoder()
    vector = encoder.encode_card({"id": "STRIKE", "is_upgraded": False})
    assert vector.shape == (16,)


def test_forward_is_batched():
    encoder = make_encoder()
    card_idx = torch.tensor([1, 2, 3], dtype=torch.long)
    ench_idx = torch.tensor([0, 0, 0], dtype=torch.long)
    upgraded = torch.tensor([0.0, 1.0, 0.0])
    out = encoder.forward(card_idx, ench_idx, upgraded)
    assert out.shape == (3, 16)


def test_input_forms_yield_same_vector():
    """A Card, a raw dict, and a CardIdentity for the same card agree."""
    encoder = make_encoder()
    raw = {"id": "STRIKE", "is_upgraded": True}
    from_dict = encoder.encode_card(raw)
    from_card = encoder.encode_card(Card.from_raw(raw))
    from_identity = encoder.encode_card(CardIdentity(card_id="STRIKE", upgrade_level=1))

    assert torch.allclose(from_dict, from_card)
    assert torch.allclose(from_dict, from_identity)


def test_same_card_deterministic_and_upgrade_changes_output():
    encoder = make_encoder()
    base = {"id": "STRIKE", "is_upgraded": False}
    upgraded = {"id": "STRIKE", "is_upgraded": True}

    assert torch.allclose(encoder.encode_card(base), encoder.encode_card(base))
    assert not torch.allclose(encoder.encode_card(base), encoder.encode_card(upgraded))


def test_card_to_indices_known_card():
    encoder = make_encoder()
    card_idx, ench_idx, upgraded = encoder.card_to_indices({"id": "STRIKE", "is_upgraded": True})

    assert card_idx == get_card_index("STRIKE", default=-1) + 1
    assert ench_idx == 0  # no enchantment -> reserved row
    assert upgraded == 1
    assert 0 <= card_idx <= encoder.num_cards
    assert 0 <= ench_idx <= encoder.num_enchantments


def test_unknown_card_maps_to_reserved_row():
    encoder = make_encoder()
    card_idx, ench_idx, upgraded = encoder.card_to_indices({"id": "NOT_A_CARD"})
    assert card_idx == 0
    # Encoding an unknown card must not raise.
    assert encoder.encode_card({"id": "NOT_A_CARD"}).shape == (16,)


def test_enchantment_index_resolves():
    encoder = make_encoder()
    enchantment_id = next(iter(DataIdMap.get_id_map("enchantments")))
    identity = CardIdentity(card_id="STRIKE", enchantment_id=enchantment_id)
    _, ench_idx, _ = encoder.card_to_indices(identity)
    assert ench_idx == get_data_index_or_default("enchantments", enchantment_id, -1) + 1
    assert ench_idx >= 1


def test_gradient_flows_to_embeddings():
    encoder = make_encoder()
    encoder.encode_card({"id": "STRIKE", "is_upgraded": True}).sum().backward()
    assert encoder.id_emb.weight.grad is not None
    assert encoder.id_emb.weight.grad.abs().sum() > 0
