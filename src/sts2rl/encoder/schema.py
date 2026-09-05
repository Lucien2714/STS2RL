"""Field and vocabulary schema shared by the tokenizer and the encoders.

Both layers must agree on which columns exist, in what order, and which
vocabulary each one indexes.  Keeping the field name and its vocabulary table
in one tuple makes that agreement structural: there is no second table to fall
out of step with, and no positional ``zip`` that would silently drop a column.

This module imports nothing from ``sts2rl`` so the torch-free tokenizer and the
trainable encoders can both depend on it without depending on each other.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


GLOBAL_CATEGORICAL: tuple[tuple[str, str], ...] = (
    ("state_type", "state_types"),
    ("character", "characters"),
)

ACTION_NUMERIC_FIELDS = ("x", "y")

MAP_CATEGORICAL_FIELDS = ("node_type",)
MAP_NUMERIC_FIELDS = (
    "col",
    "row",
    "is_current",
    "is_visited",
    "is_candidate",
    "is_boss",
    "is_reachable",
    "min_distance_to_boss",
    "max_distance_to_boss",
)

GLOBAL_NUMERIC_FIELDS = (
    "act",
    "floor",
    "ascension",
    "hp",
    "max_hp",
    "hp_ratio",
    "block",
    "gold",
    "energy",
    "max_energy",
    "energy_ratio",
    "stars",
    "max_potion_slots",
    "deck_count",
    "draw_pile_count",
    "discard_pile_count",
    "exhaust_pile_count",
    "orb_slots",
    "orb_empty_slots",
)

ENTITY_CATEGORICAL: Mapping[str, tuple[tuple[str, str], ...]] = MappingProxyType(
    {
        "player": (
            ("character", "characters"),
            ("owner_type", "owner_types"),
        ),
        "card": (
            ("card_id", "cards"),
            ("card_type", "card_types"),
            ("rarity", "rarities"),
            ("card_zone", "card_zones"),
            ("entity_zone", "entity_zones"),
            ("target_type", "target_types"),
            ("enchantment_id", "enchantments"),
            ("selection_type", "selection_types"),
        ),
        "relic": (
            ("relic_id", "relics"),
            ("rarity", "rarities"),
            ("entity_zone", "entity_zones"),
        ),
        "potion": (
            ("potion_id", "potions"),
            ("target_type", "target_types"),
            ("entity_zone", "entity_zones"),
        ),
        "orb": (
            ("orb_id", "orbs"),
            ("entity_zone", "entity_zones"),
        ),
        "pet": (
            ("monster_id", "monsters"),
            ("owner_type", "owner_types"),
            ("entity_zone", "entity_zones"),
        ),
        "enemy": (
            ("monster_id", "monsters"),
            ("owner_type", "owner_types"),
            ("entity_zone", "entity_zones"),
        ),
        "power": (
            ("power_id", "powers"),
            ("power_type", "power_types"),
            ("owner_type", "owner_types"),
            ("entity_zone", "entity_zones"),
        ),
        "intent": (
            ("intent_id", "intents"),
            ("entity_zone", "entity_zones"),
        ),
        "reward": (
            ("reward_type", "reward_types"),
            ("potion_id", "potions"),
            ("relic_id", "relics"),
            ("card_id", "cards"),
            ("entity_zone", "entity_zones"),
        ),
        "shop_item": (
            ("shop_category", "shop_categories"),
            ("card_id", "cards"),
            ("relic_id", "relics"),
            ("potion_id", "potions"),
            ("card_type", "card_types"),
            ("rarity", "rarities"),
            ("target_type", "target_types"),
            ("entity_zone", "entity_zones"),
        ),
        "event_option": (
            ("event_id", "events"),
            ("event_option", "event_options"),
            ("entity_zone", "entity_zones"),
        ),
        "rest_option": (
            ("rest_option", "rest_options"),
            ("entity_zone", "entity_zones"),
        ),
        "bundle": (
            ("selection_type", "selection_types"),
            ("entity_zone", "entity_zones"),
        ),
        "crystal_cell": (
            ("item_type", "crystal_item_types"),
            ("entity_zone", "entity_zones"),
        ),
        "crystal_tool": (
            ("tool", "crystal_tools"),
            ("entity_zone", "entity_zones"),
        ),
    }
)

ENTITY_NUMERIC_FIELDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "player": GLOBAL_NUMERIC_FIELDS[3:],
        "card": (
            "cost",
            "star_cost",
            "upgrade_level",
            "max_upgrade_level",
            "copy_count",
            "position",
            "can_play",
            "is_upgradable",
            "is_enchanted",
            "selected",
        ),
        "relic": ("counter",),
        "potion": ("can_use_in_combat",),
        "orb": ("passive", "evoke", "position"),
        "pet": ("hp", "max_hp", "hp_ratio", "block", "position"),
        "enemy": ("hp", "max_hp", "hp_ratio", "block", "position"),
        "power": ("amount",),
        "intent": ("label", "position"),
        "reward": ("gold_amount",),
        "shop_item": (
            "price",
            "is_stocked",
            "can_afford",
            "on_sale",
            "card_cost",
            "card_star_cost",
        ),
        "event_option": (
            "is_locked",
            "is_proceed",
            "was_chosen",
        ),
        "rest_option": ("is_enabled",),
        "bundle": ("card_count",),
        "crystal_cell": (
            "x",
            "y",
            "x_ratio",
            "y_ratio",
            "is_hidden",
            "is_clickable",
            "is_highlighted",
            "is_hovered",
            "is_good",
            "item_width",
            "item_height",
        ),
        "crystal_tool": ("can_use", "selected"),
    }
)

ENTITY_KINDS = tuple(ENTITY_CATEGORICAL)

GLOBAL_CATEGORICAL_FIELDS = tuple(field for field, _ in GLOBAL_CATEGORICAL)

ENTITY_CATEGORICAL_FIELDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        kind: tuple(field for field, _ in columns)
        for kind, columns in ENTITY_CATEGORICAL.items()
    }
)
