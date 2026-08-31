"""Dynamic structured action-space contract tests."""

import pytest

from sts2rl.agents import LegalActionProvider, NoLegalActionsError


def payloads(state):
    return [action.to_dict() for action in LegalActionProvider().candidates(state)]


def test_combat_candidates_expand_enemy_targets_and_filter_unplayable_cards():
    state = {
        "state_type": "monster",
        "battle": {
            "turn": "player",
            "is_play_phase": True,
            "enemies": [
                {"entity_id": "A", "hp": 10},
                {"entity_id": "B", "hp": 0},
            ],
        },
        "player": {
            "hand": [
                {"index": 0, "can_play": True, "target_type": "AnyEnemy"},
                {"index": 1, "can_play": True, "target_type": "AllEnemies"},
                {"index": 2, "can_play": False, "target_type": "None"},
            ],
            "potions": [
                {
                    "slot": 0,
                    "can_use_in_combat": True,
                    "target_type": "AnyEnemy",
                }
            ],
        },
    }

    assert payloads(state) == [
        {"type": "play_card", "card_index": 0, "target": "A"},
        {"type": "play_card", "card_index": 1},
        {"type": "use_potion", "slot": 0, "target": "A"},
        {"type": "discard_potion", "slot": 0},
        {"type": "end_turn"},
    ]


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            {
                "state_type": "hand_select",
                "hand_select": {"cards": [{"index": 2}], "can_confirm": True},
            },
            [
                {"type": "combat_select_card", "card_index": 2},
                {"type": "combat_confirm_selection"},
            ],
        ),
        (
            {
                "state_type": "rewards",
                "rewards": {"items": [{"index": 1}], "can_proceed": True},
                "player": {"potions": []},
            },
            [
                {"type": "claim_reward", "index": 1},
                {"type": "proceed"},
            ],
        ),
        (
            {
                "state_type": "card_reward",
                "card_reward": {"cards": [{"index": 0}], "can_skip": True},
            },
            [
                {"type": "select_card_reward", "card_index": 0},
                {"type": "skip_card_reward"},
            ],
        ),
        (
            {"state_type": "map", "map": {"next_options": [{"index": 4}]}},
            [{"type": "choose_map_node", "index": 4}],
        ),
        (
            {
                "state_type": "event",
                "event": {
                    "in_dialogue": False,
                    "options": [
                        {"index": 0, "is_locked": True},
                        {"index": 1, "is_locked": False},
                    ],
                },
            },
            [{"type": "choose_event_option", "index": 1}],
        ),
        (
            {
                "state_type": "rest_site",
                "rest_site": {
                    "options": [
                        {"index": 0, "is_enabled": False},
                        {"index": 1, "is_enabled": True},
                    ],
                    "can_proceed": True,
                },
            },
            [
                {"type": "choose_rest_option", "index": 1},
                {"type": "proceed"},
            ],
        ),
        (
            {
                "state_type": "shop",
                "shop": {
                    "items": [
                        {"index": 2, "is_stocked": True, "can_afford": True},
                        {"index": 3, "is_stocked": True, "can_afford": False},
                    ],
                    "can_proceed": True,
                },
            },
            [
                {"type": "shop_purchase", "index": 2},
                {"type": "proceed"},
            ],
        ),
        (
            {
                "state_type": "treasure",
                "treasure": {"relics": [{"index": 0}], "can_proceed": True},
            },
            [
                {"type": "claim_treasure_relic", "index": 0},
                {"type": "proceed"},
            ],
        ),
        (
            {
                "state_type": "card_select",
                "card_select": {
                    "cards": [{"index": 3}],
                    "can_confirm": True,
                    "can_cancel": True,
                },
            },
            [
                {"type": "select_card", "index": 3},
                {"type": "confirm_selection"},
                {"type": "cancel_selection"},
            ],
        ),
        (
            {
                "state_type": "bundle_select",
                "bundle_select": {
                    "bundles": [{"index": 2}],
                    "can_confirm": True,
                    "can_cancel": True,
                },
            },
            [
                {"type": "select_bundle", "index": 2},
                {"type": "confirm_bundle_selection"},
                {"type": "cancel_bundle_selection"},
            ],
        ),
        (
            {
                "state_type": "relic_select",
                "relic_select": {"relics": [{"index": 1}], "can_skip": True},
            },
            [
                {"type": "select_relic", "index": 1},
                {"type": "skip_relic_selection"},
            ],
        ),
        (
            {
                "state_type": "crystal_sphere",
                "crystal_sphere": {
                    "can_use_big_tool": True,
                    "can_use_small_tool": False,
                    "tool": "big",
                    "clickable_cells": [{"x": 4, "y": 7}],
                    "can_proceed": True,
                },
            },
            [
                {"type": "crystal_sphere_set_tool", "tool": "big"},
                {"type": "crystal_sphere_click_cell", "x": 4, "y": 7},
                {"type": "crystal_sphere_proceed"},
            ],
        ),
    ],
)
def test_full_run_state_candidates(state, expected):
    assert payloads(state) == expected


def test_dialogue_only_exposes_advance_action():
    assert payloads(
        {"state_type": "event", "event": {"in_dialogue": True, "options": []}}
    ) == [{"type": "advance_dialogue"}]


def test_full_potion_belt_excludes_new_potion_and_exposes_discard_actions():
    state = {
        "state_type": "rewards",
        "rewards": {
            "items": [
                {"index": 0, "type": "gold"},
                {"index": 1, "type": "potion"},
            ],
            "can_proceed": True,
        },
        "player": {
            "potions": [{"slot": 0}],
            "max_potion_slots": 1,
        },
    }

    assert payloads(state) == [
        {"type": "claim_reward", "index": 0},
        {"type": "proceed"},
        {"type": "discard_potion", "slot": 0},
    ]


def test_crystal_sphere_requires_selecting_tool_before_clicking_cell():
    state = {
        "state_type": "crystal_sphere",
        "crystal_sphere": {
            "tool": "none",
            "can_use_big_tool": True,
            "can_use_small_tool": True,
            "clickable_cells": [{"x": 4, "y": 7}],
        },
    }

    assert payloads(state) == [
        {"type": "crystal_sphere_set_tool", "tool": "big"},
        {"type": "crystal_sphere_set_tool", "tool": "small"},
    ]


@pytest.mark.parametrize("state_type", ["menu", "game_over", "unknown", "overlay"])
def test_non_automatable_states_raise_explicitly(state_type):
    with pytest.raises(NoLegalActionsError, match="No safe legal actions"):
        LegalActionProvider().require_candidates({"state_type": state_type})
