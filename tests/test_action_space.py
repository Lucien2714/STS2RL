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
            # Leaving is not offered while a reward is still there to take.
            [
                {"type": "claim_reward", "index": 1},
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

    # The gold is still claimable, so leaving is not on offer yet.
    assert payloads(state) == [
        {"type": "claim_reward", "index": 0},
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


def test_shop_proceed_follows_can_proceed_now_that_it_means_what_it_says():
    """The mod used to report false while the inventory overlay was open."""
    unaffordable = {
        "state_type": "shop",
        "player": {"gold": 28, "potions": []},
        "shop": {
            "can_proceed": True,
            "inventory_open": True,
            "items": [
                {"index": 0, "category": "card", "price": 149,
                 "is_stocked": True, "can_afford": False},
            ],
        },
    }

    assert [a.to_dict() for a in LegalActionProvider().candidates(unaffordable)] == [
        {"type": "proceed"}
    ]

    not_ready = {
        "state_type": "shop",
        "player": {"gold": 28, "potions": []},
        "shop": {"can_proceed": False, "items": [], "error": "inventory not ready"},
    }

    assert LegalActionProvider().candidates(not_ready) == ()


def test_an_opening_treasure_chest_is_not_skipped_by_a_proceed_fallback():
    state = {
        "state_type": "treasure",
        "treasure": {"message": "Opening chest..."},
    }

    assert LegalActionProvider().candidates(state) == ()


def test_a_selected_card_is_not_offered_again():
    """Re-picking a selected card is rejected and wastes the step."""
    state = {
        "state_type": "card_select",
        "card_select": {
            "screen_type": "transform",
            "selected_count": 1,
            "min_select": 2,
            "max_select": 2,
            "cards": [
                {"index": 0, "id": "STRIKE_IRONCLAD", "is_selected": True},
                {"index": 1, "id": "DEFEND_IRONCLAD", "is_selected": False},
            ],
            "can_confirm": False,
            "can_cancel": True,
        },
    }

    actions = [a.to_dict() for a in LegalActionProvider().candidates(state)]

    assert {"type": "select_card", "index": 1} in actions
    assert {"type": "select_card", "index": 0} not in actions


def test_no_further_picks_once_the_maximum_is_selected():
    state = {
        "state_type": "card_select",
        "card_select": {
            "screen_type": "transform",
            "selected_count": 2,
            "min_select": 2,
            "max_select": 2,
            "cards": [
                {"index": 0, "id": "STRIKE_IRONCLAD", "is_selected": True},
                {"index": 1, "id": "DEFEND_IRONCLAD", "is_selected": True},
                {"index": 2, "id": "BASH", "is_selected": False},
            ],
            "can_confirm": True,
            "can_cancel": True,
        },
    }

    actions = [a.to_dict() for a in LegalActionProvider().candidates(state)]

    assert actions == [{"type": "confirm_selection"}, {"type": "cancel_selection"}]


def test_hand_selection_also_skips_already_picked_cards():
    state = {
        "state_type": "hand_select",
        "hand_select": {
            "mode": "simple_select",
            "selected_count": 1,
            "min_select": 2,
            "max_select": 2,
            "cards": [
                {"index": 0, "id": "STRIKE_IRONCLAD", "is_selected": True},
                {"index": 1, "id": "DEFEND_IRONCLAD", "is_selected": False},
            ],
            "can_confirm": False,
        },
    }

    actions = [a.to_dict() for a in LegalActionProvider().candidates(state)]

    assert actions == [{"type": "combat_select_card", "card_index": 1}]


def test_selection_without_the_new_fields_still_offers_every_card():
    """Older builds report neither is_selected nor the counts."""
    state = {
        "state_type": "card_select",
        "card_select": {
            "screen_type": "transform",
            "cards": [{"index": 0}, {"index": 1}],
            "can_cancel": True,
        },
    }

    actions = [a.to_dict() for a in LegalActionProvider().candidates(state)]

    assert {"type": "select_card", "index": 0} in actions
    assert {"type": "select_card", "index": 1} in actions


def test_an_open_bundle_preview_offers_only_confirm_and_cancel():
    """select_bundle is rejected while a preview is open."""
    state = {
        "state_type": "bundle_select",
        "bundle_select": {
            "bundles": [{"index": 0}, {"index": 1}],
            "preview_showing": True,
            "can_confirm": True,
            "can_cancel": True,
        },
    }

    actions = [a.to_dict() for a in LegalActionProvider().candidates(state)]

    assert actions == [
        {"type": "confirm_bundle_selection"},
        {"type": "cancel_bundle_selection"},
    ]


def _player_with_potions(count: int, capacity: int | None) -> dict:
    player: dict = {
        "hand": [],
        "potions": [
            {"slot": slot, "can_use_in_combat": False} for slot in range(count)
        ],
    }
    if capacity is not None:
        player["max_potion_slots"] = capacity
    return player


def _combat(count: int, capacity: int | None) -> dict:
    return {
        "state_type": "monster",
        "battle": {"turn": "player", "is_play_phase": True, "enemies": []},
        "player": _player_with_potions(count, capacity),
    }


def test_a_potion_is_not_discardable_while_the_belt_has_room():
    """Throwing one away buys nothing, so it is not a choice worth offering."""
    actions = payloads(_combat(count=1, capacity=3))

    assert not [a for a in actions if a["type"] == "discard_potion"]


def test_a_full_belt_may_discard_to_make_room():
    actions = payloads(_combat(count=3, capacity=3))

    assert [a["slot"] for a in actions if a["type"] == "discard_potion"] == [0, 1, 2]


def test_an_unreported_capacity_does_not_invent_discards():
    """Without max_potion_slots the belt cannot be known to be full."""
    actions = payloads(_combat(count=2, capacity=None))

    assert not [a for a in actions if a["type"] == "discard_potion"]


def test_the_rewards_screen_follows_the_same_rule():
    full = {
        "state_type": "rewards",
        "rewards": {"items": [{"index": 0, "type": "gold"}], "can_proceed": True},
        "player": _player_with_potions(2, 2),
    }
    roomy = {**full, "player": _player_with_potions(1, 2)}

    assert [a for a in payloads(full) if a["type"] == "discard_potion"]
    assert not [a for a in payloads(roomy) if a["type"] == "discard_potion"]


def test_the_shop_follows_the_same_rule():
    full = {
        "state_type": "shop",
        "shop": {"items": [], "can_proceed": True},
        "player": _player_with_potions(2, 2),
    }
    roomy = {**full, "player": _player_with_potions(1, 2)}

    assert [a for a in payloads(full) if a["type"] == "discard_potion"]
    assert not [a for a in payloads(roomy) if a["type"] == "discard_potion"]


def test_leaving_a_reward_screen_is_not_offered_while_rewards_remain():
    """proceed abandons the screen and is a step cheaper than claiming.

    Offered side by side, a traced policy took proceed on 27 of 28 reward
    screens and never once reached the card-reward screen behind them.
    """
    state = {
        "state_type": "rewards",
        "rewards": {
            "items": [
                {"index": 0, "type": "gold"},
                {"index": 1, "type": "relic"},
                {"index": 2, "type": "card"},
            ],
            "can_proceed": True,
        },
        "player": {"potions": []},
    }

    actions = payloads(state)

    assert [a["type"] for a in actions] == ["claim_reward"] * 3


def test_an_empty_reward_screen_may_be_left():
    state = {
        "state_type": "rewards",
        "rewards": {"items": [], "can_proceed": True},
        "player": {"potions": []},
    }

    assert payloads(state) == [{"type": "proceed"}]


def test_a_screen_holding_only_an_unclaimable_potion_may_be_left():
    """A full belt blocks the claim, so refusing to leave would strand it."""
    state = {
        "state_type": "rewards",
        "rewards": {"items": [{"index": 0, "type": "potion"}], "can_proceed": True},
        "player": {"potions": [{"slot": 0}], "max_potion_slots": 1},
    }

    actions = payloads(state)

    assert {"type": "proceed"} in actions
    assert not [a for a in actions if a["type"] == "claim_reward"]


def test_the_card_choice_still_lives_on_its_own_screen():
    """Claiming a card opens card_reward, where skipping is a real decision."""
    state = {
        "state_type": "card_reward",
        "card_reward": {"cards": [{"index": 0}, {"index": 1}], "can_skip": True},
    }

    actions = payloads(state)

    assert {"type": "skip_card_reward"} in actions
    assert len([a for a in actions if a["type"] == "select_card_reward"]) == 2
