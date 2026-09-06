"""State-only tokenizer tests for structured game observations."""

from __future__ import annotations

import math

import pytest
import torch

from sts2rl.encoder import (
    ENTITY_CATEGORICAL_FIELDS,
    ENTITY_NUMERIC_FIELDS,
    EntityReference,
    GLOBAL_CATEGORICAL_FIELDS,
    GLOBAL_NUMERIC_FIELDS,
    GameTokenizer,
    GameVocabulary,
    PAD_INDEX,
    UNKNOWN_INDEX,
)
from sts2rl.env import GameObservation


@pytest.fixture(scope="module")
def vocabulary() -> GameVocabulary:
    return GameVocabulary.from_bundled_data()


@pytest.fixture(scope="module")
def tokenizer(vocabulary: GameVocabulary) -> GameTokenizer:
    return GameTokenizer(vocabulary)


def _column(fields: tuple[str, ...], name: str) -> int:
    return fields.index(name)


def _card(**changes: object) -> dict[str, object]:
    card: dict[str, object] = {
        "id": "UPPERCUT",
        "name": "Uppercut",
        "type": "Attack",
        "cost": "2",
        "star_cost": None,
        "target_type": "AnyEnemy",
        "rarity": "Uncommon",
        "is_upgraded": False,
        "current_upgrade_level": 0,
        "description": "This text is deliberately ignored.",
    }
    card.update(changes)
    return card


def _base_player(**changes: object) -> dict[str, object]:
    player: dict[str, object] = {
        "character": "The Ironclad",
        "hp": 72,
        "max_hp": 80,
        "block": 0,
        "gold": 99,
        "energy": 0,
        "max_energy": 3,
        "stars": None,
        "status": [],
        "relics": [],
        "potions": [],
        "hand": [],
        "draw_pile": [],
        "discard_pile": [],
        "exhaust_pile": [],
    }
    player.update(changes)
    return player


def test_globals_come_from_the_single_state_the_api_returns(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    state = {
        "state_type": "monster",
        "run": {"act": 1, "floor": 12, "ascension": 3},
        "player": _base_player(),
        "battle": {"enemies": []},
    }

    tokenized = tokenizer.tokenize_state(GameObservation(state))

    assert tokenized.global_categorical.tolist() == [
        vocabulary.lookup("state_types", "monster"),
        vocabulary.lookup("characters", "The Ironclad"),
    ]
    floor = _column(GLOBAL_NUMERIC_FIELDS, "floor")
    energy = _column(GLOBAL_NUMERIC_FIELDS, "energy")
    assert tokenized.global_numeric[floor].item() == pytest.approx(math.log1p(12))
    assert tokenized.global_numeric[energy].item() == 0
    assert tokenized.global_numeric_mask[energy].item() is True
    assert tokenized.game_map is None


def test_card_zones_group_only_unordered_observationally_equal_copies(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    state = {
        "state_type": "monster",
        "player": _base_player(
            hand=[_card(index=8), _card(index=3)],
            draw_pile=[_card(), _card(), _card(cost="1")],
            discard_pile=[_card()],
            exhaust_pile=[_card()],
        ),
        "battle": {"enemies": []},
    }

    tokenized = tokenizer.tokenize_state(GameObservation(state))
    cards = tokenized.entities["card"]
    zone_col = _column(ENTITY_CATEGORICAL_FIELDS["card"], "card_zone")
    position_col = _column(ENTITY_NUMERIC_FIELDS["card"], "position")
    count_col = _column(ENTITY_NUMERIC_FIELDS["card"], "copy_count")
    hand_zone = vocabulary.lookup("card_zones", "hand")
    draw_zone = vocabulary.lookup("card_zones", "draw")
    hand_rows = cards.categorical[:, zone_col] == hand_zone
    draw_rows = cards.categorical[:, zone_col] == draw_zone

    assert hand_rows.sum().item() == 2
    assert cards.numeric[hand_rows, position_col].tolist() == [0.0, 1.0]
    assert cards.numeric_mask[hand_rows, position_col].tolist() == [True, True]
    assert draw_rows.sum().item() == 2
    assert cards.numeric_mask[draw_rows, position_col].tolist() == [False, False]
    assert sorted(cards.numeric[draw_rows, count_col].tolist()) == pytest.approx(
        sorted([math.log1p(1), math.log1p(2)])
    )


def test_combat_entities_preserve_positions_and_owner_references(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    state = {
        "state_type": "monster",
        "player": _base_player(
            status=[{"id": "STRENGTH", "type": "Buff", "amount": 2}],
            pets=[
                {
                    "id": "OSTY",
                    "alive": True,
                    "hp": 9,
                    "max_hp": 12,
                    "block": 1,
                    "status": [
                        {"id": "STRENGTH", "type": "Buff", "amount": 1}
                    ],
                }
            ],
            orbs=[
                {"name": "Lightning", "passive_val": 3, "evoke_val": 8},
                {"id": "FROST_ORB", "passive_val": 2, "evoke_val": 5},
            ],
        ),
        "battle": {
            "enemies": [
                {
                    "entity_id": "ARCHITECT_0",
                    "name": "The Architect",
                    "hp": 100,
                    "max_hp": 120,
                    "block": 4,
                    "status": [
                        {"id": "STRENGTH", "type": "Buff", "amount": 3}
                    ],
                    "intents": [
                        {"type": "Attack", "label": "11"},
                        {"type": "Buff", "label": "?"},
                    ],
                }
            ]
        },
    }

    tokenized = tokenizer.tokenize_state(GameObservation(state))

    assert tokenized.entities["player"].entity_count == 1
    assert tokenized.entities["pet"].active.tolist() == [True]
    assert tokenized.entities["pet"].active_mask.tolist() == [True]
    assert tokenized.entities["enemy"].categorical[0, 0].item() == vocabulary.lookup(
        "monsters", "ARCHITECT"
    )
    assert tokenized.entities["orb"].numeric[:, 2].tolist() == [0.0, 1.0]
    assert tokenized.entities["power"].owners == (
        EntityReference("player", 0),
        EntityReference("pet", 0),
        EntityReference("enemy", 0),
    )
    assert [owner.kind for owner in tokenized.entities["intent"].owners] == [
        "enemy",
        "enemy",
    ]
    label = _column(ENTITY_NUMERIC_FIELDS["intent"], "label")
    assert tokenized.entities["intent"].numeric_mask[:, label].tolist() == [True, False]


def test_relic_activity_is_explicit_and_unknown_is_not_guessed(
    tokenizer: GameTokenizer,
):
    state = {
        "state_type": "map",
        "player": _base_player(
            relics=[
                {"id": "BURNING_BLOOD", "counter": None},
                {"id": "BLACK_STAR", "active": False},
                {"id": "VAJRA", "used": False},
            ]
        ),
        "map": {"nodes": []},
    }

    relics = tokenizer.tokenize_state(GameObservation(state)).entities["relic"]

    assert relics.active.tolist() == [False, False, True]
    assert relics.active_mask.tolist() == [False, True, True]


def test_rewards_shop_event_and_rest_options_have_semantic_ids(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    state = {
        "state_type": "event",
        "player": _base_player(),
        "rewards": {
            "items": [
                {"index": 7, "type": "gold", "gold_amount": 25},
                {"index": 8, "type": "potion", "potion_id": "SWIFT_POTION"},
            ]
        },
        "shop": {
            "items": [
                {
                    "index": 9,
                    "category": "card",
                    "price": 75,
                    "is_stocked": True,
                    "can_afford": False,
                    "on_sale": False,
                    "card_id": "UPPERCUT",
                    "card_type": "Attack",
                    "card_rarity": "Uncommon",
                    "card_cost": "X",
                }
            ]
        },
        "event": {
            "event_id": "ABYSSAL_BATHS",
            "options": [
                {"index": 4, "title": "Immerse", "is_locked": False},
                {"index": 5, "title": "Runtime-only option", "is_locked": True},
            ],
        },
        "rest_site": {
            "options": [
                {"index": 2, "id": "rest", "is_enabled": True},
                {"index": 3, "id": "modded-option", "is_enabled": False},
            ]
        },
    }

    tokenized = tokenizer.tokenize_state(GameObservation(state))
    rewards = tokenized.entities["reward"]
    shop = tokenized.entities["shop_item"]
    events = tokenized.entities["event_option"]
    rest = tokenized.entities["rest_option"]

    assert rewards.categorical[1, 1].item() == vocabulary.lookup(
        "potions", "SWIFT_POTION"
    )
    assert shop.categorical[0, 1].item() == vocabulary.lookup("cards", "UPPERCUT")
    card_cost = _column(ENTITY_NUMERIC_FIELDS["shop_item"], "card_cost")
    assert shop.numeric[0, card_cost].item() == -1
    assert events.categorical[0, 1].item() == vocabulary.event_option_index(
        "ABYSSAL_BATHS", "Immerse"
    )
    assert events.categorical[1, 1].item() == UNKNOWN_INDEX
    assert events.active.tolist() == [True, False]
    assert rest.categorical[1, 0].item() == UNKNOWN_INDEX
    assert rest.active.tolist() == [True, False]


def test_selection_cards_and_bundles_remain_individual_and_link_children(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    state = {
        "state_type": "bundle_select",
        "player": _base_player(),
        "card_select": {
            "screen_type": "upgrade",
            "cards": [_card(index=2), _card(index=5)],
            "selected_cards": [{"index": 5}],
        },
        "card_reward": {"cards": [_card(index=11)]},
        "bundle_select": {
            "bundles": [
                {"index": 7, "card_count": 2, "cards": [_card(), _card()]}
            ]
        },
        "relic_select": {"relics": [{"id": "BLACK_STAR", "rarity": "Rare"}]},
        "treasure": {"relics": [{"id": "VAJRA", "rarity": "Common"}]},
    }

    tokenized = tokenizer.tokenize_state(GameObservation(state))
    cards = tokenized.entities["card"]
    bundles = tokenized.entities["bundle"]
    card_zone = _column(ENTITY_CATEGORICAL_FIELDS["card"], "card_zone")

    assert cards.entity_count == 5
    assert (
        cards.categorical[:, card_zone]
        == vocabulary.lookup("card_zones", "bundle")
    ).sum().item() == 2
    assert bundles.children == ((
        EntityReference("card", 3),
        EntityReference("card", 4),
    ),)
    assert tokenized.entities["relic"].entity_count == 2


def test_crystal_cells_and_tools_preserve_grid_semantics(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    state = {
        "state_type": "crystal_sphere",
        "player": _base_player(),
        "crystal_sphere": {
            "grid_width": 3,
            "grid_height": 2,
            "cells": [
                {
                    "x": 0,
                    "y": 0,
                    "is_hidden": True,
                    "is_clickable": True,
                    "is_highlighted": False,
                    "is_hovered": False,
                },
                {"x": 2, "y": 1, "is_hidden": False, "is_clickable": False},
            ],
            "revealed_items": [
                {
                    "item_type": "CrystalSphereGold",
                    "x": 2,
                    "y": 1,
                    "width": 1,
                    "height": 1,
                    "is_good": True,
                }
            ],
            "tool": "small",
            "can_use_big_tool": False,
            "can_use_small_tool": True,
        },
    }

    tokenized = tokenizer.tokenize_state(GameObservation(state))
    cells = tokenized.entities["crystal_cell"]
    tools = tokenized.entities["crystal_tool"]

    assert cells.entity_count == 2
    assert cells.numeric[1, 2:4].tolist() == [1.0, 1.0]
    assert cells.categorical[1, 0].item() == vocabulary.lookup(
        "crystal_item_types", "CrystalSphereGold"
    )
    assert cells.active.tolist() == [True, False]
    assert tools.active.tolist() == [False, True]
    assert tools.numeric[:, 1].tolist() == [0.0, 1.0]


def test_empty_state_has_all_well_shaped_cpu_entity_batches(
    tokenizer: GameTokenizer,
):
    tokenized = tokenizer.tokenize_state(GameObservation({"state_type": "menu"}))

    assert tokenized.global_categorical.shape == (len(GLOBAL_CATEGORICAL_FIELDS),)
    assert tokenized.global_numeric.shape == (len(GLOBAL_NUMERIC_FIELDS),)
    assert tokenized.global_numeric_mask.any().item() is False
    assert set(tokenized.entities) == set(ENTITY_CATEGORICAL_FIELDS)
    for kind, batch in tokenized.entities.items():
        assert batch.categorical.shape == (0, len(ENTITY_CATEGORICAL_FIELDS[kind]))
        assert batch.numeric.shape == (0, len(ENTITY_NUMERIC_FIELDS[kind]))
        assert batch.categorical.device.type == "cpu"


def test_free_text_does_not_change_tokenized_state(tokenizer: GameTokenizer):
    first = {
        "state_type": "monster",
        "player": _base_player(hand=[_card(description="first description")]),
        "battle": {
            "enemies": [
                {
                    "name": "The Architect",
                    "hp": 10,
                    "max_hp": 10,
                    "description": "first enemy description",
                }
            ]
        },
    }
    second = {
        "state_type": "monster",
        "player": _base_player(hand=[_card(description="unrelated prose")]),
        "battle": {
            "enemies": [
                {
                    "name": "The Architect",
                    "hp": 10,
                    "max_hp": 10,
                    "description": "different enemy prose",
                }
            ]
        },
    }

    tokenized_first = tokenizer.tokenize_state(GameObservation(first))
    tokenized_second = tokenizer.tokenize_state(GameObservation(second))

    assert torch.equal(tokenized_first.global_categorical, tokenized_second.global_categorical)
    assert torch.equal(
        tokenized_first.entities["card"].categorical,
        tokenized_second.entities["card"].categorical,
    )
    assert torch.equal(
        tokenized_first.entities["enemy"].numeric,
        tokenized_second.entities["enemy"].numeric,
    )


def _player_detail(cards, **deck_changes):
    deck = {
        "count": sum(c.get("quantity", 1) for c in cards),
        "unique_count": len(cards),
        "upgraded_count": sum(
            c.get("quantity", 1) for c in cards if c.get("upgrade_level")
        ),
        "cards": cards,
    }
    deck.update(deck_changes)
    return {"in_run": True, "deck": deck}


def _deck_card(**changes):
    card = {
        "id": "STRIKE_IRONCLAD",
        "name": "Strike",
        "type": "Attack",
        "rarity": "Basic",
        "cost": "1",
        "star_cost": None,
        "description": "Deal 6 damage.",
        "is_upgraded": False,
        "upgrade_level": 0,
        "max_upgrade_level": 1,
        "is_upgradable": True,
        "enchantment": None,
        "affliction": None,
        "quantity": 5,
    }
    card.update(changes)
    return card


def test_master_deck_becomes_deck_zone_rows_with_reported_quantities(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    observation = GameObservation(
        {"state_type": "map", "player": _base_player()},
        _player_detail(
            [
                _deck_card(),
                _deck_card(
                    id="BASH", name="Bash", cost="2", quantity=1,
                    upgrade_level=1, is_upgraded=True, is_upgradable=False,
                ),
            ]
        ),
    )

    tokenized = tokenizer.tokenize_state(observation)
    cards = tokenized.entities["card"]
    zone = _column(ENTITY_CATEGORICAL_FIELDS["card"], "card_zone")
    ident = _column(ENTITY_CATEGORICAL_FIELDS["card"], "card_id")
    count = _column(ENTITY_NUMERIC_FIELDS["card"], "copy_count")
    level = _column(ENTITY_NUMERIC_FIELDS["card"], "upgrade_level")
    deck_rows = cards.categorical[:, zone] == vocabulary.lookup("card_zones", "deck")

    assert deck_rows.sum().item() == 2
    assert vocabulary.lookup("cards", "STRIKE_IRONCLAD") in (
        cards.categorical[deck_rows, ident].tolist()
    )
    assert sorted(cards.numeric[deck_rows, count].tolist()) == pytest.approx(
        sorted([math.log1p(5), math.log1p(1)])
    )
    assert sorted(cards.numeric[deck_rows, level].tolist()) == [0.0, 1.0]


def test_deck_totals_reach_the_global_features(tokenizer: GameTokenizer):
    observation = GameObservation(
        {"state_type": "map", "player": _base_player()},
        _player_detail([_deck_card(quantity=7, upgrade_level=1)]),
    )

    tokenized = tokenizer.tokenize_state(observation)
    total = _column(GLOBAL_NUMERIC_FIELDS, "deck_count")
    upgraded = _column(GLOBAL_NUMERIC_FIELDS, "deck_upgraded_count")

    assert tokenized.global_numeric[total].item() == pytest.approx(math.log1p(7))
    assert tokenized.global_numeric[upgraded].item() == pytest.approx(math.log1p(7))


def test_enchantment_and_affliction_are_distinguished_from_absence(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    observation = GameObservation(
        {"state_type": "map", "player": _base_player()},
        _player_detail(
            [
                _deck_card(quantity=1),
                _deck_card(
                    id="BASH",
                    quantity=1,
                    enchantment={"id": "ADROIT", "name": "Adroit"},
                    affliction={"id": "SOMETHING", "name": "Something"},
                ),
            ]
        ),
    )

    tokenized = tokenizer.tokenize_state(observation)
    cards = tokenized.entities["card"]
    zone = _column(ENTITY_CATEGORICAL_FIELDS["card"], "card_zone")
    ench = _column(ENTITY_CATEGORICAL_FIELDS["card"], "enchantment_id")
    afflicted = _column(ENTITY_NUMERIC_FIELDS["card"], "is_afflicted")
    deck_rows = cards.categorical[:, zone] == vocabulary.lookup("card_zones", "deck")

    assert sorted(cards.categorical[deck_rows, ench].tolist()) == sorted(
        [PAD_INDEX, vocabulary.lookup("enchantments", "ADROIT")]
    )
    assert sorted(cards.numeric[deck_rows, afflicted].tolist()) == [0.0, 1.0]


def test_a_missing_deck_leaves_deck_columns_masked(tokenizer: GameTokenizer):
    tokenized = tokenizer.tokenize_state(
        GameObservation({"state_type": "map", "player": _base_player()})
    )
    total = _column(GLOBAL_NUMERIC_FIELDS, "deck_count")

    assert tokenized.global_numeric_mask[total].item() is False
    assert tokenized.entities["card"].entity_count == 0


def test_battle_round_reaches_the_global_features(tokenizer: GameTokenizer):
    state = {
        "state_type": "monster",
        "run": {"act": 1, "floor": 3},
        "player": _base_player(),
        "battle": {"round": 4, "enemies": []},
    }

    tokenized = tokenizer.tokenize_state(GameObservation(state))
    column = _column(GLOBAL_NUMERIC_FIELDS, "battle_round")

    assert tokenized.global_numeric[column].item() == 4.0
    assert tokenized.global_numeric_mask[column].item() is True


def test_battle_round_is_missing_outside_combat(tokenizer: GameTokenizer):
    tokenized = tokenizer.tokenize_state(
        GameObservation({"state_type": "map", "player": _base_player()})
    )
    column = _column(GLOBAL_NUMERIC_FIELDS, "battle_round")

    assert tokenized.global_numeric_mask[column].item() is False


def test_statuses_resolve_through_the_suffixed_ids_the_api_sends(
    tokenizer: GameTokenizer,
    vocabulary: GameVocabulary,
):
    """Live statuses arrive as STRENGTH_POWER, which the table does not hold."""
    state = {
        "state_type": "monster",
        "player": _base_player(
            status=[{"id": "STRENGTH_POWER", "name": "Strength",
                     "type": "Buff", "amount": 2}],
        ),
        "battle": {
            "enemies": [
                {
                    "entity_id": "JAW_WORM_0",
                    "name": "Jaw Worm",
                    "hp": 40,
                    "status": [{"id": "VULNERABLE_POWER", "name": "Vulnerable",
                                "type": "Debuff", "amount": 2}],
                    "intents": [{"type": "DebuffStrong", "label": ""}],
                }
            ]
        },
    }

    tokenized = tokenizer.tokenize_state(GameObservation(state))
    powers = tokenized.entities["power"]
    intents = tokenized.entities["intent"]
    pid = _column(ENTITY_CATEGORICAL_FIELDS["power"], "power_id")
    iid = _column(ENTITY_CATEGORICAL_FIELDS["intent"], "intent_id")

    assert sorted(powers.categorical[:, pid].tolist()) == sorted(
        [vocabulary.lookup("powers", "STRENGTH"),
         vocabulary.lookup("powers", "VULNERABLE")]
    )
    assert UNKNOWN_INDEX not in powers.categorical[:, pid].tolist()
    assert intents.categorical[0, iid].item() == vocabulary.lookup(
        "intents", "DEBUFF_STRONG"
    )
