"""Semantic action-reference tests for ``GameTokenizer``."""

from __future__ import annotations

import pytest

from sts2rl.actions import GameAction
from sts2rl.agents import LegalActionProvider
from sts2rl.encoder import (
    ACTION_NUMERIC_FIELDS,
    EntityReference,
    GameTokenizer,
    GameVocabulary,
    TokenizationError,
)
from sts2rl.env import GameObservation


@pytest.fixture(scope="module")
def tokenizer() -> GameTokenizer:
    return GameTokenizer(GameVocabulary.from_bundled_data())


def _player(**changes: object) -> dict[str, object]:
    player: dict[str, object] = {
        "character": "The Ironclad",
        "hp": 70,
        "max_hp": 80,
        "relics": [],
        "potions": [],
        "hand": [],
        "draw_pile": [],
        "discard_pile": [],
        "exhaust_pile": [],
        "status": [],
    }
    player.update(changes)
    return player


def _card(index: int, *, target_type: str = "AnyEnemy") -> dict[str, object]:
    return {
        "index": index,
        "id": "UPPERCUT",
        "type": "Attack",
        "cost": "2",
        "rarity": "Uncommon",
        "target_type": target_type,
        "can_play": True,
    }


def test_combat_actions_reference_hand_potions_and_live_enemy(
    tokenizer: GameTokenizer,
):
    state = {
        "state_type": "monster",
        "player": _player(
            hand=[_card(10), _card(20, target_type="Self")],
            potions=[
                {
                    "id": "SWIFT_POTION",
                    "slot": 4,
                    "can_use_in_combat": True,
                    "target_type": "AnyEnemy",
                }
            ],
        ),
        "battle": {
            "turn": "player",
            "is_play_phase": True,
            "enemies": [
                {
                    "entity_id": "ARCHITECT_0",
                    "name": "The Architect",
                    "hp": 10,
                    "max_hp": 10,
                    "status": [],
                    "intents": [],
                }
            ],
        },
    }
    candidates = LegalActionProvider().require_candidates(state)

    decision = tokenizer.tokenize_decision(GameObservation(state), candidates)

    # The belt is not full, so discarding the potion is not offered.
    assert [action.source for action in decision.actions] == [
        EntityReference("card", 0),
        EntityReference("card", 1),
        EntityReference("potion", 0),
        None,
    ]
    assert [action.target for action in decision.actions] == [
        EntityReference("enemy", 0),
        None,
        EntityReference("enemy", 0),
        None,
    ]
    assert all(action.numeric.shape == (len(ACTION_NUMERIC_FIELDS),) for action in decision.actions)
    assert all(not action.numeric_mask.any().item() for action in decision.actions)


@pytest.mark.parametrize(
    ("state", "action", "expected"),
    [
        (
            {
                "state_type": "hand_select",
                "player": _player(),
                "hand_select": {"cards": [_card(3)], "can_confirm": False},
            },
            GameAction("combat_select_card", card_index=3),
            EntityReference("card", 0),
        ),
        (
            {
                "state_type": "rewards",
                "player": _player(),
                "rewards": {"items": [{"index": 6, "type": "gold"}]},
            },
            GameAction("claim_reward", index=6),
            EntityReference("reward", 0),
        ),
        (
            {
                "state_type": "card_reward",
                "player": _player(),
                "card_reward": {"cards": [_card(8)]},
            },
            GameAction("select_card_reward", card_index=8),
            EntityReference("card", 0),
        ),
        (
            {
                "state_type": "event",
                "player": _player(),
                "event": {
                    "event_id": "ABYSSAL_BATHS",
                    "options": [{"index": 2, "title": "Immerse"}],
                },
            },
            GameAction("choose_event_option", index=2),
            EntityReference("event_option", 0),
        ),
        (
            {
                "state_type": "rest_site",
                "player": _player(),
                "rest_site": {"options": [{"index": 4, "id": "rest"}]},
            },
            GameAction("choose_rest_option", index=4),
            EntityReference("rest_option", 0),
        ),
        (
            {
                "state_type": "shop",
                "player": _player(),
                "shop": {"items": [{"index": 5, "category": "card"}]},
            },
            GameAction("shop_purchase", index=5),
            EntityReference("shop_item", 0),
        ),
        (
            {
                "state_type": "fake_merchant",
                "player": _player(),
                "fake_merchant": {
                    "shop": {"items": [{"index": 7, "category": "relic"}]}
                },
            },
            GameAction("shop_purchase", index=7),
            EntityReference("shop_item", 0),
        ),
        (
            {
                "state_type": "card_select",
                "player": _player(),
                "card_select": {"cards": [_card(12)]},
            },
            GameAction("select_card", index=12),
            EntityReference("card", 0),
        ),
        (
            {
                "state_type": "bundle_select",
                "player": _player(),
                "bundle_select": {
                    "bundles": [{"index": 9, "cards": [_card(0)]}]
                },
            },
            GameAction("select_bundle", index=9),
            EntityReference("bundle", 0),
        ),
        (
            {
                "state_type": "relic_select",
                "player": _player(),
                "relic_select": {
                    "relics": [{"index": 11, "id": "BLACK_STAR"}]
                },
            },
            GameAction("select_relic", index=11),
            EntityReference("relic", 0),
        ),
        (
            {
                "state_type": "treasure",
                "player": _player(),
                "treasure": {"relics": [{"index": 13, "id": "VAJRA"}]},
            },
            GameAction("claim_treasure_relic", index=13),
            EntityReference("relic", 0),
        ),
    ],
)
def test_indexed_actions_resolve_to_screen_semantic_entities(
    tokenizer: GameTokenizer,
    state: dict[str, object],
    action: GameAction,
    expected: EntityReference,
):
    decision = tokenizer.tokenize_decision(GameObservation(state), [action])

    assert decision.actions[0].source == expected
    assert decision.actions[0].target is None
    assert not decision.actions[0].numeric_mask.any().item()


@pytest.mark.parametrize(
    "action_type",
    [
        "end_turn",
        "combat_confirm_selection",
        "skip_card_reward",
        "proceed",
        "advance_dialogue",
        "confirm_selection",
        "cancel_selection",
        "confirm_bundle_selection",
        "cancel_bundle_selection",
        "skip_relic_selection",
        "crystal_sphere_proceed",
    ],
)
def test_parameterless_actions_have_no_entity_references(
    tokenizer: GameTokenizer,
    action_type: str,
):
    state = {"state_type": "unknown", "player": _player()}

    action = tokenizer.tokenize_decision(
        GameObservation(state), [GameAction(action_type)]
    ).actions[0]

    assert action.source is None
    assert action.target is None
    assert not action.numeric_mask.any().item()


def test_crystal_actions_link_tools_and_cells_and_keep_semantic_coordinates(
    tokenizer: GameTokenizer,
):
    state = {
        "state_type": "crystal_sphere",
        "player": _player(),
        "crystal_sphere": {
            "grid_width": 9,
            "grid_height": 11,
            "cells": [{"x": 4, "y": 7, "is_clickable": True}],
            "clickable_cells": [{"x": 4, "y": 7}],
            "tool": "small",
            "can_use_big_tool": True,
            "can_use_small_tool": True,
        },
    }
    # The action space plays the sphere by rule and returns a single move, so
    # the candidates are built here: this is a tokenizer test.
    candidates = (
        GameAction("crystal_sphere_set_tool", tool="big"),
        GameAction("crystal_sphere_set_tool", tool="small"),
        GameAction("crystal_sphere_click_cell", x=4, y=7),
    )

    decision = tokenizer.tokenize_decision(GameObservation(state), candidates)

    assert decision.actions[0].source == EntityReference("crystal_tool", 0)
    assert decision.actions[1].source == EntityReference("crystal_tool", 1)
    click = decision.actions[2]
    assert click.source == EntityReference("crystal_tool", 1)
    assert click.target == EntityReference("crystal_cell", 0)
    assert click.numeric.tolist() == [4.0, 7.0]
    assert click.numeric_mask.tolist() == [True, True]


@pytest.mark.parametrize(
    "action",
    [
        GameAction("play_card", card_index=999),
        GameAction("play_card"),
        GameAction("not_a_real_action"),
        GameAction("menu_select", option="singleplayer"),
        GameAction("choose_map_node", index=0),
    ],
)
def test_unresolved_or_deferred_actions_raise_contextual_error(
    tokenizer: GameTokenizer,
    action: GameAction,
):
    state = {"state_type": "monster", "player": _player()}

    with pytest.raises(TokenizationError) as raised:
        tokenizer.tokenize_decision(GameObservation(state), [action])

    message = str(raised.value)
    assert "state_type='monster'" in message
    assert action.action_type in message


def test_duplicate_runtime_handles_are_rejected_as_ambiguous(
    tokenizer: GameTokenizer,
):
    state = {
        "state_type": "card_select",
        "player": _player(),
        "card_select": {"cards": [_card(2), _card(2)]},
    }

    with pytest.raises(TokenizationError, match="unique selection_card"):
        tokenizer.tokenize_decision(
            GameObservation(state),
            [GameAction("select_card", index=2)],
        )


def test_candidate_order_is_preserved(tokenizer: GameTokenizer):
    state = {"state_type": "rewards", "player": _player()}
    candidates = [GameAction("proceed"), GameAction("skip_card_reward")]

    decision = tokenizer.tokenize_decision(GameObservation(state), candidates)

    expected = [
        tokenizer.vocabulary.lookup("action_types", action.action_type)
        for action in candidates
    ]
    assert [action.action_type.item() for action in decision.actions] == expected


def _complete_map_state() -> dict[str, object]:
    return {
        "state_type": "map",
        "player": _player(),
        "map": {
            "current_position": {"col": 0, "row": 0},
            "visited": [{"col": 0, "row": 0}],
            "next_options": [{"index": 4, "col": 1, "row": 1}],
            "nodes": [
                {
                    "col": 0,
                    "row": 0,
                    "type": "Start",
                    "children": [[1, 1]],
                },
                {
                    "col": 1,
                    "row": 1,
                    "type": "Monster",
                    "children": [[0, 2]],
                },
            ],
            "bosses": [{"col": 0, "row": 2}],
        },
    }


def _legal_state_cases() -> dict[str, dict[str, object]]:
    """Representative complete payload for every learnable screen family."""
    combat = {
        "player": _player(hand=[_card(0)]),
        "battle": {
            "turn": "player",
            "is_play_phase": True,
            "enemies": [
                {
                    "entity_id": "ARCHITECT_0",
                    "id": "THE_ARCHITECT",
                    "hp": 20,
                    "max_hp": 20,
                    "status": [],
                    "intents": [],
                }
            ],
        },
    }
    return {
        state_type: {"state_type": state_type, **combat}
        for state_type in ("monster", "elite", "boss")
    } | {
        "hand_select": {
            "state_type": "hand_select",
            "player": _player(),
            "hand_select": {"cards": [_card(2)], "can_confirm": True},
        },
        "rewards": {
            "state_type": "rewards",
            "player": _player(),
            "rewards": {
                "items": [{"index": 1, "type": "gold", "amount": 25}],
                "can_proceed": True,
            },
        },
        "card_reward": {
            "state_type": "card_reward",
            "player": _player(),
            "card_reward": {"cards": [_card(3)], "can_skip": True},
        },
        "map": _complete_map_state(),
        "event": {
            "state_type": "event",
            "player": _player(),
            "event": {
                "event_id": "ABYSSAL_BATHS",
                "options": [
                    {"index": 4, "title": "Immerse", "is_locked": False}
                ],
            },
        },
        "event_dialogue": {
            "state_type": "event",
            "player": _player(),
            "event": {"in_dialogue": True, "options": []},
        },
        "rest_site": {
            "state_type": "rest_site",
            "player": _player(),
            "rest_site": {
                "options": [{"index": 5, "id": "rest", "is_enabled": True}],
                "can_proceed": True,
            },
        },
        "shop": {
            "state_type": "shop",
            "player": _player(),
            "shop": {
                "items": [
                    {
                        "index": 6,
                        "category": "card",
                        "id": "UPPERCUT",
                        "can_afford": True,
                        "is_stocked": True,
                    }
                ],
                "can_proceed": True,
            },
        },
        "fake_merchant": {
            "state_type": "fake_merchant",
            "player": _player(),
            "fake_merchant": {
                "shop": {
                    "items": [
                        {
                            "index": 7,
                            "category": "relic",
                            "id": "BLACK_STAR",
                            "can_afford": True,
                            "is_stocked": True,
                        }
                    ]
                }
            },
        },
        "treasure": {
            "state_type": "treasure",
            "player": _player(),
            "treasure": {
                "relics": [{"index": 8, "id": "VAJRA"}],
                "can_proceed": True,
            },
        },
        "card_select": {
            "state_type": "card_select",
            "player": _player(),
            "card_select": {
                "cards": [_card(9)],
                "can_confirm": True,
                "can_cancel": True,
            },
        },
        "bundle_select": {
            "state_type": "bundle_select",
            "player": _player(),
            "bundle_select": {
                "bundles": [{"index": 10, "cards": [_card(0)]}],
                "can_confirm": True,
                "can_cancel": True,
            },
        },
        "relic_select": {
            "state_type": "relic_select",
            "player": _player(),
            "relic_select": {
                "relics": [{"index": 11, "id": "BLACK_STAR"}],
                "can_skip": True,
            },
        },
        "crystal_sphere": {
            "state_type": "crystal_sphere",
            "player": _player(),
            "crystal_sphere": {
                "grid_width": 9,
                "grid_height": 11,
                "cells": [{"x": 4, "y": 7, "is_clickable": True}],
                "clickable_cells": [{"x": 4, "y": 7}],
                "tool": "big",
                "can_use_big_tool": True,
                "can_use_small_tool": True,
                "can_proceed": True,
            },
        },
    }


@pytest.mark.parametrize(
    ("case_name", "state"),
    _legal_state_cases().items(),
)
def test_every_legal_screen_fixture_tokenizes_its_complete_candidate_set(
    tokenizer: GameTokenizer,
    case_name: str,
    state: dict[str, object],
):
    del case_name
    candidates = LegalActionProvider().require_candidates(state)
    observation = GameObservation(state)

    decision = tokenizer.tokenize_decision(observation, candidates)

    assert len(decision.actions) == len(candidates)
