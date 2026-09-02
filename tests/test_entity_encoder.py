"""Trainable entity Transformer behavior tests."""

from __future__ import annotations

import pytest
import torch

from sts2rl.encoder import (
    EncoderConfig,
    EntityTransformer,
    GameTokenizer,
    GameVocabulary,
)
from sts2rl.env import GameObservation


@pytest.fixture(scope="module")
def vocabulary() -> GameVocabulary:
    return GameVocabulary.from_bundled_data()


@pytest.fixture(scope="module")
def tokenizer(vocabulary: GameVocabulary) -> GameTokenizer:
    return GameTokenizer(vocabulary)


def _player(**changes: object) -> dict[str, object]:
    player: dict[str, object] = {
        "character": "The Ironclad",
        "hp": 70,
        "max_hp": 80,
        "block": 0,
        "gold": 50,
        "relics": [],
        "potions": [],
        "status": [],
    }
    player.update(changes)
    return player


def _tokenize(tokenizer: GameTokenizer, state: dict[str, object]):
    return tokenizer.tokenize_state(GameObservation(state))


def test_default_config_builds_two_zero_dropout_layers(
    vocabulary: GameVocabulary,
):
    config = EncoderConfig()
    model = EntityTransformer(vocabulary, config)

    assert config.hidden_dim == 128
    assert len(model.transformer.layers) == 2
    assert all(layer.dropout.p == 0 for layer in model.transformer.layers)
    assert model.transformer.layers[0].linear1.out_features == 256


def test_encoder_config_rejects_invalid_heads_and_stochastic_dropout():
    with pytest.raises(ValueError, match="divisible"):
        EncoderConfig(hidden_dim=10, entity_heads=3)
    with pytest.raises(ValueError, match="dropout"):
        EncoderConfig(dropout=0.1)


def test_empty_entity_collections_still_encode_the_state_token(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    torch.manual_seed(1)
    model = EntityTransformer(vocabulary, EncoderConfig(hidden_dim=16, entity_heads=4))
    state = _tokenize(tokenizer, {"state_type": "menu"})

    encoded = model(state)

    assert encoded.state_embedding.shape == (16,)
    assert set(encoded.entity_embeddings) == set(state.entities)
    assert all(values.shape == (0, 16) for values in encoded.entity_embeddings.values())


def test_unordered_entity_permutation_keeps_state_and_permutes_entity_outputs(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    torch.manual_seed(2)
    model = EntityTransformer(vocabulary, EncoderConfig(hidden_dim=16, entity_heads=4))
    model.eval()
    first = _tokenize(
        tokenizer,
        {
            "state_type": "map",
            "player": _player(
                relics=[{"id": "BURNING_BLOOD"}, {"id": "BLACK_STAR"}]
            ),
            "map": {},
        },
    )
    second = _tokenize(
        tokenizer,
        {
            "state_type": "map",
            "player": _player(
                relics=[{"id": "BLACK_STAR"}, {"id": "BURNING_BLOOD"}]
            ),
            "map": {},
        },
    )

    first_encoded = model(first)
    second_encoded = model(second)

    assert torch.allclose(
        first_encoded.state_embedding,
        second_encoded.state_embedding,
        atol=1e-5,
    )
    assert torch.allclose(
        first_encoded.entity_embeddings["relic"],
        second_encoded.entity_embeddings["relic"].flip(0),
        atol=1e-5,
    )


def test_status_is_aggregated_into_owner_and_receives_state_gradient(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    torch.manual_seed(3)
    model = EntityTransformer(vocabulary, EncoderConfig(hidden_dim=16, entity_heads=4))
    state = _tokenize(
        tokenizer,
        {
            "state_type": "monster",
            "player": _player(
                status=[{"id": "STRENGTH", "type": "Buff", "amount": 2}]
            ),
            "battle": {"enemies": []},
        },
    )

    encoded = model(state)
    weights = torch.arange(1, 17, dtype=torch.float32)
    (encoded.state_embedding * weights).sum().backward()

    power_index = vocabulary.lookup("powers", "STRENGTH")
    gradient = model.entity_embeddings["power__power_id"].weight.grad
    assert gradient is not None
    assert gradient[power_index].abs().sum().item() > 0


def test_bundle_children_are_aggregated_and_receive_gradient(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    torch.manual_seed(4)
    model = EntityTransformer(vocabulary, EncoderConfig(hidden_dim=16, entity_heads=4))
    state = _tokenize(
        tokenizer,
        {
            "state_type": "bundle_select",
            "player": _player(),
            "bundle_select": {
                "bundles": [
                    {
                        "index": 0,
                        "cards": [
                            {
                                "id": "UPPERCUT",
                                "type": "Attack",
                                "cost": 2,
                                "rarity": "Uncommon",
                            }
                        ],
                    }
                ]
            },
        },
    )

    encoded = model(state)
    encoded.entity_embeddings["bundle"][0, 0].backward()

    card_index = vocabulary.lookup("cards", "UPPERCUT")
    gradient = model.entity_embeddings["card__card_id"].weight.grad
    assert gradient is not None
    assert gradient[card_index].abs().sum().item() > 0


def test_transformer_and_numeric_projection_receive_gradients(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    torch.manual_seed(5)
    model = EntityTransformer(vocabulary, EncoderConfig(hidden_dim=16, entity_heads=4))
    state = _tokenize(
        tokenizer,
        {
            "state_type": "shop",
            "player": _player(),
            "shop": {
                "items": [
                    {
                        "index": 0,
                        "category": "card",
                        "card_id": "UPPERCUT",
                        "price": 70,
                    }
                ]
            },
        },
    )

    encoded = model(state)
    weights = torch.arange(1, 17, dtype=torch.float32)
    (encoded.state_embedding * weights).sum().backward()

    assert model.global_numeric_projection.weight.grad is not None
    assert model.entity_numeric_projections["shop_item"].weight.grad is not None
    assert model.transformer.layers[0].self_attn.in_proj_weight.grad is not None
    assert model.transformer.layers[1].linear1.weight.grad is not None
