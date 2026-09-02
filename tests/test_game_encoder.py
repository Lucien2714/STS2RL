"""Composed structured state/action encoder tests."""

from __future__ import annotations

import math

import pytest
import torch

from sts2rl.actions import GameAction
from sts2rl.agents import LegalActionProvider
from sts2rl.encoder import (
    EncoderConfig,
    GameEncoder,
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
        "hand": [],
        "status": [],
    }
    player.update(changes)
    return player


def _card(index: int, card_id: str) -> dict[str, object]:
    return {
        "index": index,
        "id": card_id,
        "type": "Attack",
        "cost": 1,
        "rarity": "Common",
        "target_type": "Self",
        "can_play": True,
    }


def _decision(
    tokenizer: GameTokenizer,
    state: dict[str, object],
    actions: list[GameAction],
):
    return tokenizer.tokenize_decision(GameObservation(state), actions)


def _map_state() -> dict[str, object]:
    return {
        "state_type": "map",
        "player": _player(),
        "map": {
            "current_position": {"col": 0, "row": 0},
            "visited": [{"col": 0, "row": 0}],
            "next_options": [
                {"index": 0, "col": 0, "row": 1},
                {"index": 1, "col": 1, "row": 1},
            ],
            "nodes": [
                {
                    "col": 0,
                    "row": 0,
                    "type": "Start",
                    "children": [[0, 1], [1, 1]],
                },
                {
                    "col": 0,
                    "row": 1,
                    "type": "Monster",
                    "children": [[0, 2]],
                },
                {
                    "col": 1,
                    "row": 1,
                    "type": "Monster",
                    "children": [[1, 2]],
                },
                {"col": 0, "row": 2, "type": "Elite", "children": [[0, 3]]},
                {"col": 1, "row": 2, "type": "Event", "children": [[0, 3]]},
            ],
            "bosses": [{"col": 0, "row": 3}],
        },
    }


def test_shared_encoder_handles_battle_event_selection_and_map_decisions(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    torch.manual_seed(20)
    model = GameEncoder(vocabulary, EncoderConfig(hidden_dim=16, entity_heads=4))
    decisions = [
        _decision(
            tokenizer,
            {
                "state_type": "monster",
                "player": _player(hand=[_card(0, "UPPERCUT")]),
                "battle": {"enemies": []},
            },
            [GameAction("play_card", card_index=0), GameAction("end_turn")],
        ),
        _decision(
            tokenizer,
            {
                "state_type": "event",
                "player": _player(),
                "event": {"in_dialogue": True},
            },
            [GameAction("advance_dialogue")],
        ),
        _decision(
            tokenizer,
            {
                "state_type": "card_select",
                "player": _player(),
                "card_select": {"cards": [_card(4, "UPPERCUT")]},
            },
            [GameAction("select_card", index=4), GameAction("confirm_selection")],
        ),
        _decision(
            tokenizer,
            _map_state(),
            [GameAction("choose_map_node", index=0)],
        ),
    ]

    for decision in decisions:
        output = model.policy_value(decision)
        assert output.encoded.state_embedding.shape == (16,)
        assert output.encoded.candidate_embeddings.shape == (
            len(decision.actions),
            16,
        )
        assert output.logits.shape == (len(decision.actions),)
        assert output.value.shape == ()


def test_policy_uses_documented_dot_product_and_learned_candidate_bias(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    torch.manual_seed(21)
    model = GameEncoder(vocabulary, EncoderConfig(hidden_dim=16, entity_heads=4))
    decision = _decision(
        tokenizer,
        {"state_type": "rewards", "player": _player()},
        [GameAction("proceed"), GameAction("skip_card_reward")],
    )

    output = model.policy_value(decision)
    expected = (
        output.encoded.candidate_embeddings.matmul(output.encoded.state_embedding)
        / math.sqrt(16)
        + model.candidate_bias(output.encoded.candidate_embeddings).squeeze(-1)
    )

    assert torch.allclose(output.logits, expected)


def test_semantic_card_sources_change_candidate_embeddings(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    torch.manual_seed(22)
    model = GameEncoder(vocabulary, EncoderConfig(hidden_dim=16, entity_heads=4))
    state = {
        "state_type": "monster",
        "player": _player(
            hand=[_card(10, "UPPERCUT"), _card(20, "ABRASIVE")]
        ),
        "battle": {"enemies": []},
    }
    decision = _decision(
        tokenizer,
        state,
        [
            GameAction("play_card", card_index=10),
            GameAction("play_card", card_index=20),
        ],
    )

    encoded = model(decision)

    assert not torch.allclose(
        encoded.candidate_embeddings[0], encoded.candidate_embeddings[1]
    )


def test_map_candidates_use_their_distinct_future_node_embeddings(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    torch.manual_seed(23)
    model = GameEncoder(vocabulary, EncoderConfig(hidden_dim=16, entity_heads=4))
    state = _map_state()
    candidates = LegalActionProvider().require_candidates(state)
    decision = tokenizer.tokenize_decision(GameObservation(state), candidates)

    encoded = model(decision)

    assert encoded.candidate_embeddings.shape == (2, 16)
    assert not torch.allclose(
        encoded.candidate_embeddings[0], encoded.candidate_embeddings[1]
    )


def test_policy_and_value_gradients_reach_all_composed_modules(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    torch.manual_seed(24)
    model = GameEncoder(vocabulary, EncoderConfig(hidden_dim=16, entity_heads=4))
    state = _map_state()
    decision = tokenizer.tokenize_decision(
        GameObservation(state),
        [
            GameAction("choose_map_node", index=0),
            GameAction("choose_map_node", index=1),
        ],
    )

    output = model.policy_value(decision)
    battle = model.policy_value(
        _decision(
            tokenizer,
            {
                "state_type": "monster",
                "player": _player(hand=[_card(3, "UPPERCUT")]),
                "battle": {"enemies": []},
            },
            [GameAction("play_card", card_index=3)],
        )
    )
    loss = output.logits[0] - output.logits[1] + output.value + battle.logits[0]
    loss.backward()

    assert model.entity_encoder.state_token.grad is not None
    assert model.map_encoder.node_type_embedding.weight.grad is not None
    assert model.action_type_embedding.weight.grad is not None
    assert model.source_projection.weight.grad is not None
    assert model.target_projection.weight.grad is not None
    assert model.candidate_bias.weight.grad is not None
    assert model.value_head[0].weight.grad is not None
