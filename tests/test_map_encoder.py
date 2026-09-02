"""Conditioned reverse-DAG map encoder tests."""

from __future__ import annotations

import pytest
import torch

from sts2rl.encoder import (
    EncoderConfig,
    GameTokenizer,
    GameVocabulary,
    MapDAGEncoder,
)
from sts2rl.env import GameObservation


@pytest.fixture(scope="module")
def vocabulary() -> GameVocabulary:
    return GameVocabulary.from_bundled_data()


@pytest.fixture(scope="module")
def tokenizer(vocabulary: GameVocabulary) -> GameTokenizer:
    return GameTokenizer(vocabulary)


def _state(*, deep_type: str = "Shop", unreachable_type: str = "Event"):
    return {
        "state_type": "map",
        "player": {
            "character": "The Ironclad",
            "hp": 70,
            "max_hp": 80,
            "relics": [],
            "potions": [],
            "status": [],
        },
        "map": {
            "current_position": {"col": 0, "row": 0},
            "visited": [{"col": 0, "row": 0}],
            "next_options": [{"index": 0, "col": 0, "row": 1}],
            "nodes": [
                {"col": 0, "row": 0, "type": "Start", "children": [[0, 1]]},
                {
                    "col": 0,
                    "row": 1,
                    "type": "Monster",
                    "children": [[0, 2]],
                },
                {
                    "col": 0,
                    "row": 2,
                    "type": deep_type,
                    "children": [[0, 3]],
                },
                {
                    "col": 2,
                    "row": 1,
                    "type": unreachable_type,
                    "children": [],
                },
            ],
            "bosses": [{"col": 0, "row": 3}],
        },
    }


def _map(tokenizer: GameTokenizer, **changes: str):
    result = tokenizer.tokenize_state(GameObservation(_state(**changes))).game_map
    assert result is not None
    return result


def test_non_map_state_skips_dag_module(vocabulary: GameVocabulary):
    encoder = MapDAGEncoder(
        vocabulary,
        EncoderConfig(hidden_dim=16, entity_heads=4),
    )

    assert encoder(None, torch.randn(16)) is None


def test_deep_descendant_changes_candidate_future_embedding(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    torch.manual_seed(10)
    encoder = MapDAGEncoder(
        vocabulary,
        EncoderConfig(hidden_dim=16, entity_heads=4),
    )
    player_context = torch.randn(16)
    first_map = _map(tokenizer, deep_type="Shop")
    second_map = _map(tokenizer, deep_type="Elite")

    first = encoder(first_map, player_context)
    second = encoder(second_map, player_context)
    assert first is not None and second is not None
    candidate = first_map.candidate_indices[0].item()

    assert not torch.allclose(
        first.node_embeddings[candidate],
        second.node_embeddings[candidate],
    )


def test_candidate_gradient_reaches_deep_base_node_embedding(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    torch.manual_seed(11)
    encoder = MapDAGEncoder(
        vocabulary,
        EncoderConfig(hidden_dim=16, entity_heads=4),
    )
    game_map = _map(tokenizer)
    encoded = encoder(game_map, torch.randn(16))
    assert encoded is not None
    candidate = game_map.candidate_indices[0].item()
    deep_node = 3
    weights = torch.arange(1, 17, dtype=torch.float32)

    gradient = torch.autograd.grad(
        (encoded.node_embeddings[candidate] * weights).sum(),
        encoded.base_node_embeddings,
    )[0]

    assert gradient[deep_node].abs().sum().item() > 0


def test_unreachable_branch_does_not_change_global_map_pooling(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    torch.manual_seed(12)
    encoder = MapDAGEncoder(
        vocabulary,
        EncoderConfig(hidden_dim=16, entity_heads=4),
    )
    player_context = torch.randn(16)
    first = encoder(_map(tokenizer, unreachable_type="Event"), player_context)
    second = encoder(_map(tokenizer, unreachable_type="Elite"), player_context)
    assert first is not None and second is not None

    assert torch.allclose(first.global_embedding, second.global_embedding)


def test_global_pooling_has_no_gradient_to_disconnected_node(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    torch.manual_seed(13)
    encoder = MapDAGEncoder(
        vocabulary,
        EncoderConfig(hidden_dim=16, entity_heads=4),
    )
    game_map = _map(tokenizer)
    encoded = encoder(game_map, torch.randn(16))
    assert encoded is not None
    unreachable = 2
    assert not game_map.reachable_mask[unreachable]

    gradient = torch.autograd.grad(
        encoded.global_embedding[0],
        encoded.base_node_embeddings,
    )[0]

    assert gradient[unreachable].abs().sum().item() == 0


def test_player_context_conditions_future_and_pooling(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    torch.manual_seed(14)
    encoder = MapDAGEncoder(
        vocabulary,
        EncoderConfig(hidden_dim=16, entity_heads=4),
    )
    game_map = _map(tokenizer)
    first = encoder(game_map, torch.zeros(16))
    second = encoder(game_map, torch.ones(16))
    assert first is not None and second is not None

    candidate = game_map.candidate_indices[0].item()
    assert not torch.allclose(
        first.node_embeddings[candidate], second.node_embeddings[candidate]
    )
    assert not torch.allclose(first.global_embedding, second.global_embedding)


def test_empty_map_produces_a_valid_conditioned_global_embedding(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    encoder = MapDAGEncoder(
        vocabulary,
        EncoderConfig(hidden_dim=16, entity_heads=4),
    )
    game_map = tokenizer.tokenize_state(
        GameObservation({"state_type": "map", "map": {}})
    ).game_map
    assert game_map is not None

    encoded = encoder(game_map, torch.randn(16))

    assert encoded is not None
    assert encoded.node_embeddings.shape == (0, 16)
    assert encoded.global_embedding.shape == (16,)
