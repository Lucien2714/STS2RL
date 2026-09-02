"""Full-map DAG tokenization and map-action reference tests."""

from __future__ import annotations

import pytest

from sts2rl.actions import GameAction
from sts2rl.encoder import (
    MAP_NUMERIC_FIELDS,
    EntityReference,
    GameTokenizer,
    GameVocabulary,
    TokenizationError,
)
from sts2rl.env import GameObservation


@pytest.fixture(scope="module")
def vocabulary() -> GameVocabulary:
    return GameVocabulary.from_bundled_data()


@pytest.fixture(scope="module")
def tokenizer(vocabulary: GameVocabulary) -> GameTokenizer:
    return GameTokenizer(vocabulary)


def _player() -> dict[str, object]:
    return {
        "character": "The Ironclad",
        "hp": 70,
        "max_hp": 80,
        "relics": [],
        "potions": [],
        "status": [],
    }


def _branching_map() -> dict[str, object]:
    return {
        "state_type": "map",
        "player": _player(),
        "map": {
            "current_position": {"col": 0, "row": 0, "type": "Start"},
            "visited": [{"col": 0, "row": 0, "type": "Start"}],
            "next_options": [
                {"index": 7, "col": 0, "row": 1, "type": "Monster"},
                {"index": 9, "col": 1, "row": 1, "type": "Monster"},
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
                {
                    "col": 0,
                    "row": 2,
                    "type": "Elite",
                    "children": [[0, 3]],
                },
                {
                    "col": 1,
                    "row": 2,
                    "type": "Event",
                    "children": [[1, 3]],
                },
                {
                    "col": 0,
                    "row": 3,
                    "type": "Shop",
                    "children": [[0, 4]],
                },
                {
                    "col": 1,
                    "row": 3,
                    "type": "RestSite",
                    "children": [[0, 4]],
                },
            ],
            "boss": {"col": 0, "row": 4, "id": "BOSS_A"},
            "bosses": [{"col": 0, "row": 4, "id": "BOSS_A"}],
        },
    }


def _column(name: str) -> int:
    return MAP_NUMERIC_FIELDS.index(name)


def test_full_map_merges_boss_and_builds_topological_parent_child_edges(
    tokenizer: GameTokenizer,
):
    game_map = tokenizer.tokenize_state(
        GameObservation(_branching_map())
    ).game_map

    assert game_map is not None
    assert game_map.node_categorical.shape == (8, 1)
    assert game_map.edge_index.shape == (2, 8)
    assert game_map.topological_order.tolist() == list(range(8))
    assert game_map.current_index == 0
    assert game_map.candidate_indices.tolist() == [1, 2]
    assert game_map.boss_indices.tolist() == [7]
    assert game_map.reachable_mask.tolist() == [
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]


def test_same_immediate_node_types_keep_different_complete_future_subgraphs(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    game_map = tokenizer.tokenize_state(
        GameObservation(_branching_map())
    ).game_map
    assert game_map is not None

    monster = vocabulary.lookup("map_node_types", "monster")
    elite = vocabulary.lookup("map_node_types", "elite")
    event = vocabulary.lookup("map_node_types", "event")
    shop = vocabulary.lookup("map_node_types", "shop")
    rest = vocabulary.lookup("map_node_types", "restsite")
    boss = vocabulary.lookup("map_node_types", "boss")

    first, second = game_map.candidate_type_counts
    assert first[monster].item() == second[monster].item() == 1
    assert first[elite].item() == 1
    assert first[shop].item() == 1
    assert first[event].item() == first[rest].item() == 0
    assert second[event].item() == 1
    assert second[rest].item() == 1
    assert second[elite].item() == second[shop].item() == 0
    assert first[boss].item() == second[boss].item() == 1
    assert not first.equal(second)


def test_map_flags_and_distances_include_shortest_and_longest_boss_paths(
    tokenizer: GameTokenizer,
):
    state = {
        "state_type": "map",
        "player": _player(),
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
                    "children": [[0, 4], [0, 2]],
                },
                {"col": 0, "row": 2, "type": "Shop", "children": [[0, 3]]},
                {"col": 0, "row": 3, "type": "Elite", "children": [[0, 4]]},
                {"col": 2, "row": 0, "type": "Event", "children": []},
            ],
            "bosses": [{"col": 0, "row": 4}],
        },
    }

    game_map = tokenizer.tokenize_state(GameObservation(state)).game_map
    assert game_map is not None
    candidate = game_map.candidate_indices[0].item()
    unreachable = 1

    assert game_map.node_numeric[candidate, _column("min_distance_to_boss")].item() == 1
    assert game_map.node_numeric[candidate, _column("max_distance_to_boss")].item() == 3
    assert game_map.node_numeric_mask[candidate, _column("min_distance_to_boss")]
    assert not game_map.reachable_mask[unreachable]
    assert not game_map.node_numeric_mask[
        unreachable, _column("min_distance_to_boss")
    ]


def test_choose_map_node_resolves_next_option_before_referencing_coordinate_node(
    tokenizer: GameTokenizer,
):
    state = _branching_map()
    actions = [
        GameAction("choose_map_node", index=9),
        GameAction("choose_map_node", index=7),
    ]

    decision = tokenizer.tokenize_decision(GameObservation(state), actions)

    assert decision.actions[0].target == EntityReference("map_node", 2)
    assert decision.actions[1].target == EntityReference("map_node", 1)
    assert decision.actions[0].source is None


@pytest.mark.parametrize(
    ("map_data", "message"),
    [
        (
            {
                "nodes": [
                    {"col": 0, "row": 0, "children": []},
                    {"col": 0, "row": 0, "children": []},
                ]
            },
            "duplicate map node coordinate",
        ),
        (
            {
                "nodes": [
                    {"col": 0, "row": 0, "children": [[1, 1]]},
                ]
            },
            "child coordinate",
        ),
        (
            {
                "nodes": [{"col": 0, "row": 0, "children": []}],
                "next_options": [{"index": 0, "col": 1, "row": 1}],
            },
            "candidate coordinate",
        ),
        (
            {
                "nodes": [
                    {"col": 0, "row": 0, "children": [[0, 1]]},
                    {"col": 0, "row": 1, "children": [[0, 0]]},
                ]
            },
            "contains a cycle",
        ),
        (
            {
                "nodes": [{"col": 0, "row": 0, "children": []}],
                "next_options": [
                    {"index": 0, "col": 0, "row": 0},
                    {"index": 1, "col": 0, "row": 0},
                ],
            },
            "duplicate candidate coordinate",
        ),
    ],
)
def test_invalid_map_graphs_fail_loudly(
    tokenizer: GameTokenizer,
    map_data: dict[str, object],
    message: str,
):
    state = {"state_type": "map", "player": _player(), "map": map_data}

    with pytest.raises(TokenizationError, match=message) as raised:
        tokenizer.tokenize_state(GameObservation(state))

    assert "state_type='map'" in str(raised.value)


def test_unresolved_and_duplicate_map_action_handles_fail_loudly(
    tokenizer: GameTokenizer,
):
    state = _branching_map()
    state["map"]["next_options"][1]["index"] = 7  # type: ignore[index]

    with pytest.raises(TokenizationError, match="unique map_candidate"):
        tokenizer.tokenize_decision(
            GameObservation(state),
            [GameAction("choose_map_node", index=7)],
        )
