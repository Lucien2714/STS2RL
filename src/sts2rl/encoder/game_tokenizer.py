"""Deterministic conversion from game observations to structured state tokens.

This stage intentionally handles state only.  Candidate-action references and
the full map DAG are added by later tokenizer stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from sts2rl.encoder.numeric import (
    NumericFeature,
    linear_feature,
    pack_numeric,
    ratio_feature,
    signed_log_feature,
)
from sts2rl.encoder.tokens import (
    EntityReference,
    TokenizedEntityBatch,
    TokenizedState,
)
from sts2rl.encoder.vocabulary import GameVocabulary
from sts2rl.env.types import GameObservation


GLOBAL_CATEGORICAL_FIELDS = ("state_type", "character")
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

ENTITY_CATEGORICAL_FIELDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "player": ("character", "owner_type"),
        "card": (
            "card_id",
            "card_type",
            "rarity",
            "card_zone",
            "entity_zone",
            "target_type",
            "enchantment_id",
            "selection_type",
        ),
        "relic": ("relic_id", "rarity", "entity_zone"),
        "potion": ("potion_id", "target_type", "entity_zone"),
        "orb": ("orb_id", "entity_zone"),
        "pet": ("monster_id", "owner_type", "entity_zone"),
        "enemy": ("monster_id", "owner_type", "entity_zone"),
        "power": ("power_id", "power_type", "owner_type", "entity_zone"),
        "intent": ("intent_id", "entity_zone"),
        "reward": (
            "reward_type",
            "potion_id",
            "relic_id",
            "card_id",
            "entity_zone",
        ),
        "shop_item": (
            "shop_category",
            "card_id",
            "relic_id",
            "potion_id",
            "card_type",
            "rarity",
            "target_type",
            "entity_zone",
        ),
        "event_option": ("event_id", "event_option", "entity_zone"),
        "rest_option": ("rest_option", "entity_zone"),
        "bundle": ("selection_type", "entity_zone"),
        "crystal_cell": ("item_type", "entity_zone"),
        "crystal_tool": ("tool", "entity_zone"),
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

ENTITY_KINDS = tuple(ENTITY_CATEGORICAL_FIELDS)


@dataclass
class _EntityRows:
    categorical: list[list[int]] = field(default_factory=list)
    numeric: list[list[float]] = field(default_factory=list)
    numeric_mask: list[list[bool]] = field(default_factory=list)
    active: list[bool] = field(default_factory=list)
    active_mask: list[bool] = field(default_factory=list)
    owners: list[EntityReference | None] = field(default_factory=list)
    children: list[tuple[EntityReference, ...]] = field(default_factory=list)

    def append(
        self,
        categorical: Sequence[int],
        numeric: Sequence[NumericFeature],
        *,
        activity: tuple[bool, bool] = (False, False),
        owner: EntityReference | None = None,
        children: Sequence[EntityReference] = (),
    ) -> int:
        values, mask = pack_numeric(numeric)
        index = len(self.categorical)
        self.categorical.append(list(categorical))
        self.numeric.append(values.tolist())
        self.numeric_mask.append(mask.tolist())
        self.active.append(activity[0])
        self.active_mask.append(activity[1])
        self.owners.append(owner)
        self.children.append(tuple(children))
        return index


class GameTokenizer:
    """Create deterministic, CPU-resident state tokens from an observation."""

    def __init__(self, vocabulary: GameVocabulary):
        self.vocabulary = vocabulary

    def tokenize_state(self, observation: GameObservation) -> TokenizedState:
        """Tokenize global values and all non-map entities in one snapshot."""
        if not isinstance(observation, GameObservation):
            raise TypeError("observation must be a GameObservation")

        state = observation.raw_state
        detail = observation.player_detail or {}
        raw_player = _mapping(state.get("player"))
        detail_player = _mapping(detail.get("player"))
        player = {**detail_player, **raw_player}
        run = {**_mapping(detail.get("run")), **_mapping(state.get("run"))}

        global_categorical = torch.tensor(
            [
                self.vocabulary.lookup("state_types", _text(state.get("state_type"))),
                self.vocabulary.lookup("characters", _text(player.get("character"))),
            ],
            dtype=torch.long,
        )
        global_numeric, global_numeric_mask = pack_numeric(
            self._global_numeric(run, player)
        )

        rows = {kind: _EntityRows() for kind in ENTITY_KINDS}
        player_ref = self._add_player(rows, player)
        self._add_cards(rows, state, player, detail_player)
        self._add_inventory(rows, player)
        self._add_combat_entities(rows, state, player, player_ref)
        self._add_screen_entities(rows, state)

        return TokenizedState(
            global_categorical=global_categorical,
            global_numeric=global_numeric,
            global_numeric_mask=global_numeric_mask,
            entities=self._build_batches(rows),
            game_map=None,
        )

    def _global_numeric(
        self,
        run: Mapping[str, object],
        player: Mapping[str, object],
    ) -> tuple[NumericFeature, ...]:
        return (
            linear_feature(run.get("act")),
            signed_log_feature(run.get("floor")),
            linear_feature(run.get("ascension")),
            signed_log_feature(player.get("hp")),
            signed_log_feature(player.get("max_hp")),
            ratio_feature(player.get("hp"), player.get("max_hp")),
            signed_log_feature(player.get("block")),
            signed_log_feature(player.get("gold")),
            linear_feature(player.get("energy")),
            linear_feature(player.get("max_energy")),
            ratio_feature(player.get("energy"), player.get("max_energy")),
            linear_feature(player.get("stars")),
            linear_feature(player.get("max_potion_slots")),
            signed_log_feature(player.get("deck_count")),
            signed_log_feature(player.get("draw_pile_count")),
            signed_log_feature(player.get("discard_pile_count")),
            signed_log_feature(player.get("exhaust_pile_count")),
            linear_feature(player.get("orb_slots")),
            linear_feature(player.get("orb_empty_slots")),
        )

    def _add_player(
        self,
        rows: dict[str, _EntityRows],
        player: Mapping[str, object],
    ) -> EntityReference | None:
        if not player:
            return None
        index = rows["player"].append(
            [
                self.vocabulary.lookup("characters", _text(player.get("character"))),
                self.vocabulary.lookup("owner_types", "player"),
            ],
            self._global_numeric({}, player)[3:],
            activity=_activity(player),
        )
        return EntityReference("player", index)

    def _add_cards(
        self,
        rows: dict[str, _EntityRows],
        state: Mapping[str, object],
        player: Mapping[str, object],
        detail_player: Mapping[str, object],
    ) -> None:
        self._add_grouped_cards(rows, _records(detail_player.get("deck")), "deck")
        self._add_individual_cards(rows, _records(player.get("hand")), "hand")
        for source, zone in (
            ("draw_pile", "draw"),
            ("discard_pile", "discard"),
            ("exhaust_pile", "exhaust"),
        ):
            self._add_grouped_cards(rows, _records(player.get(source)), zone)

        hand_select = _mapping(state.get("hand_select"))
        self._add_individual_cards(
            rows,
            _records(hand_select.get("cards")),
            "selection",
            selection_type=_text(hand_select.get("mode")),
            selected_indices=_selected_indices(hand_select),
        )
        card_reward = _mapping(state.get("card_reward"))
        self._add_individual_cards(
            rows,
            _records(card_reward.get("cards")),
            "reward",
        )
        card_select = _mapping(state.get("card_select"))
        self._add_individual_cards(
            rows,
            _records(card_select.get("cards")),
            "selection",
            selection_type=_text(card_select.get("screen_type")),
            selected_indices=_selected_indices(card_select),
        )

    def _add_grouped_cards(
        self,
        rows: dict[str, _EntityRows],
        cards: list[Mapping[str, object]],
        zone: str,
    ) -> None:
        groups: dict[tuple[object, ...], tuple[Mapping[str, object], int]] = {}
        for card in cards:
            signature = _card_signature(card)
            previous = groups.get(signature)
            groups[signature] = (card, 1 if previous is None else previous[1] + 1)
        for signature in sorted(groups, key=lambda item: tuple(map(str, item))):
            card, count = groups[signature]
            self._append_card(rows, card, zone, copy_count=count)

    def _add_individual_cards(
        self,
        rows: dict[str, _EntityRows],
        cards: list[Mapping[str, object]],
        zone: str,
        *,
        selection_type: str | None = None,
        selected_indices: set[int] | None = None,
    ) -> list[EntityReference]:
        references = []
        for position, card in enumerate(cards):
            selected = None
            raw_index = _integer(card.get("index"))
            if selected_indices is not None and raw_index is not None:
                selected = raw_index in selected_indices
            index = self._append_card(
                rows,
                card,
                zone,
                copy_count=1,
                position=position if zone == "hand" else None,
                selection_type=selection_type,
                selected=selected,
            )
            references.append(EntityReference("card", index))
        return references

    def _append_card(
        self,
        rows: dict[str, _EntityRows],
        card: Mapping[str, object],
        zone: str,
        *,
        copy_count: int,
        position: int | None = None,
        selection_type: str | None = None,
        selected: bool | None = None,
    ) -> int:
        enchanted = card.get("is_enchanted")
        enchantment = _mapping(card.get("enchantment"))
        enchantment_id = _text(enchantment.get("id") or enchantment.get("name"))
        if enchanted is True and enchantment_id is None:
            enchantment_id = "<present-but-unknown>"
        return rows["card"].append(
            [
                self.vocabulary.lookup("cards", _identity(card)),
                self.vocabulary.lookup("card_types", _text(card.get("type"))),
                self.vocabulary.lookup("rarities", _text(card.get("rarity"))),
                self.vocabulary.lookup("card_zones", zone),
                self.vocabulary.lookup("entity_zones", zone),
                self.vocabulary.lookup("target_types", _text(card.get("target_type"))),
                self.vocabulary.lookup("enchantments", enchantment_id),
                self.vocabulary.lookup("selection_types", selection_type),
            ],
            [
                _cost_feature(card.get("cost")),
                _cost_feature(card.get("star_cost")),
                _upgrade_feature(card),
                linear_feature(card.get("max_upgrade_level")),
                signed_log_feature(copy_count),
                linear_feature(position),
                _bool_feature(card.get("can_play")),
                _bool_feature(card.get("is_upgradable")),
                _bool_feature(enchanted),
                _bool_feature(selected),
            ],
            activity=_activity(card, positive=("active", "is_active", "can_play")),
        )

    def _add_inventory(
        self,
        rows: dict[str, _EntityRows],
        player: Mapping[str, object],
    ) -> None:
        for relic in _records(player.get("relics")):
            self._append_relic(rows, relic, "inventory")
        for potion in _records(player.get("potions")):
            rows["potion"].append(
                [
                    self.vocabulary.lookup("potions", _identity(potion)),
                    self.vocabulary.lookup(
                        "target_types", _text(potion.get("target_type"))
                    ),
                    self.vocabulary.lookup("entity_zones", "inventory"),
                ],
                [_bool_feature(potion.get("can_use_in_combat"))],
                activity=_activity(potion),
            )
        for position, orb in enumerate(_records(player.get("orbs"))):
            rows["orb"].append(
                [
                    self.vocabulary.lookup("orbs", _identity(orb)),
                    self.vocabulary.lookup("entity_zones", "battle"),
                ],
                [
                    signed_log_feature(orb.get("passive_val")),
                    signed_log_feature(orb.get("evoke_val")),
                    linear_feature(position),
                ],
                activity=_activity(orb),
            )

    def _append_relic(
        self,
        rows: dict[str, _EntityRows],
        relic: Mapping[str, object],
        zone: str,
    ) -> int:
        return rows["relic"].append(
            [
                self.vocabulary.lookup("relics", _identity(relic)),
                self.vocabulary.lookup("rarities", _text(relic.get("rarity"))),
                self.vocabulary.lookup("entity_zones", zone),
            ],
            [signed_log_feature(relic.get("counter"))],
            activity=_activity(relic),
        )

    def _add_combat_entities(
        self,
        rows: dict[str, _EntityRows],
        state: Mapping[str, object],
        player: Mapping[str, object],
        player_ref: EntityReference | None,
    ) -> None:
        if player_ref is not None:
            self._add_powers(
                rows,
                _records(player.get("status")),
                player_ref,
                "player",
            )

        pet_refs: list[tuple[EntityReference, Mapping[str, object]]] = []
        for position, pet in enumerate(_records(player.get("pets"))):
            index = rows["pet"].append(
                [
                    self.vocabulary.lookup("monsters", _identity(pet)),
                    self.vocabulary.lookup("owner_types", "pet"),
                    self.vocabulary.lookup("entity_zones", "battle"),
                ],
                _combatant_numeric(pet, position),
                activity=_activity(pet, positive=("alive", "active", "is_active")),
            )
            pet_refs.append((EntityReference("pet", index), pet))
        for reference, pet in pet_refs:
            self._add_powers(rows, _records(pet.get("status")), reference, "pet")

        battle = _mapping(state.get("battle"))
        enemy_refs: list[tuple[EntityReference, Mapping[str, object]]] = []
        for position, enemy in enumerate(_records(battle.get("enemies"))):
            index = rows["enemy"].append(
                [
                    self.vocabulary.lookup("monsters", _identity(enemy, include_entity=False)),
                    self.vocabulary.lookup("owner_types", "enemy"),
                    self.vocabulary.lookup("entity_zones", "battle"),
                ],
                _combatant_numeric(enemy, position),
                activity=_activity(enemy, positive=("alive", "active", "is_active")),
            )
            enemy_refs.append((EntityReference("enemy", index), enemy))
        for reference, enemy in enemy_refs:
            self._add_powers(rows, _records(enemy.get("status")), reference, "enemy")
            for position, intent in enumerate(_records(enemy.get("intents"))):
                rows["intent"].append(
                    [
                        self.vocabulary.lookup("intents", _text(intent.get("type"))),
                        self.vocabulary.lookup("entity_zones", "battle"),
                    ],
                    [linear_feature(intent.get("label")), linear_feature(position)],
                    owner=reference,
                )

    def _add_powers(
        self,
        rows: dict[str, _EntityRows],
        powers: list[Mapping[str, object]],
        owner: EntityReference,
        owner_type: str,
    ) -> None:
        for power in powers:
            rows["power"].append(
                [
                    self.vocabulary.lookup("powers", _identity(power)),
                    self.vocabulary.lookup("power_types", _text(power.get("type"))),
                    self.vocabulary.lookup("owner_types", owner_type),
                    self.vocabulary.lookup("entity_zones", "battle"),
                ],
                [signed_log_feature(power.get("amount"))],
                activity=_activity(power),
                owner=owner,
            )

    def _add_screen_entities(
        self,
        rows: dict[str, _EntityRows],
        state: Mapping[str, object],
    ) -> None:
        self._add_rewards(rows, state)
        self._add_shop(rows, state)
        self._add_event(rows, state)
        self._add_rest(rows, state)
        self._add_bundle(rows, state)
        self._add_relic_selections(rows, state)
        self._add_crystal_sphere(rows, state)

    def _add_rewards(
        self, rows: dict[str, _EntityRows], state: Mapping[str, object]
    ) -> None:
        rewards = _mapping(state.get("rewards"))
        for reward in _records(rewards.get("items")):
            reward_type = _text(reward.get("type"))
            rows["reward"].append(
                [
                    self.vocabulary.lookup("reward_types", reward_type),
                    self.vocabulary.lookup("potions", _text(reward.get("potion_id"))),
                    self.vocabulary.lookup("relics", _text(reward.get("relic_id"))),
                    self.vocabulary.lookup("cards", _text(reward.get("card_id"))),
                    self.vocabulary.lookup("entity_zones", "reward"),
                ],
                [signed_log_feature(reward.get("gold_amount"))],
                activity=_activity(reward),
            )

    def _add_shop(
        self, rows: dict[str, _EntityRows], state: Mapping[str, object]
    ) -> None:
        shop = _mapping(state.get("shop"))
        if not shop:
            shop = _mapping(_mapping(state.get("fake_merchant")).get("shop"))
        for item in _records(shop.get("items")):
            rows["shop_item"].append(
                [
                    self.vocabulary.lookup("shop_categories", _text(item.get("category"))),
                    self.vocabulary.lookup("cards", _text(item.get("card_id"))),
                    self.vocabulary.lookup("relics", _text(item.get("relic_id"))),
                    self.vocabulary.lookup("potions", _text(item.get("potion_id"))),
                    self.vocabulary.lookup("card_types", _text(item.get("card_type"))),
                    self.vocabulary.lookup(
                        "rarities",
                        _text(item.get("card_rarity") or item.get("relic_rarity")),
                    ),
                    self.vocabulary.lookup("target_types", _text(item.get("target_type"))),
                    self.vocabulary.lookup("entity_zones", "shop"),
                ],
                [
                    signed_log_feature(item.get("price", item.get("cost"))),
                    _bool_feature(item.get("is_stocked")),
                    _bool_feature(item.get("can_afford")),
                    _bool_feature(item.get("on_sale")),
                    _cost_feature(item.get("card_cost")),
                    _cost_feature(item.get("card_star_cost")),
                ],
                activity=_activity(item, positive=("is_stocked",)),
            )

    def _add_event(
        self, rows: dict[str, _EntityRows], state: Mapping[str, object]
    ) -> None:
        event = _mapping(state.get("event"))
        event_id = _text(event.get("event_id"))
        for option in _records(event.get("options")):
            locked = option.get("is_locked")
            rows["event_option"].append(
                [
                    self.vocabulary.lookup("events", event_id),
                    self.vocabulary.event_option_index(event_id, _text(option.get("title"))),
                    self.vocabulary.lookup("entity_zones", "event"),
                ],
                [
                    _bool_feature(locked),
                    _bool_feature(option.get("is_proceed")),
                    _bool_feature(option.get("was_chosen")),
                ],
                activity=(not locked, True) if isinstance(locked, bool) else (False, False),
            )

    def _add_rest(
        self, rows: dict[str, _EntityRows], state: Mapping[str, object]
    ) -> None:
        rest = _mapping(state.get("rest_site"))
        for option in _records(rest.get("options")):
            enabled = option.get("is_enabled")
            rows["rest_option"].append(
                [
                    self.vocabulary.lookup("rest_options", _identity(option)),
                    self.vocabulary.lookup("entity_zones", "rest"),
                ],
                [_bool_feature(enabled)],
                activity=(enabled, True) if isinstance(enabled, bool) else (False, False),
            )

    def _add_bundle(
        self, rows: dict[str, _EntityRows], state: Mapping[str, object]
    ) -> None:
        selection = _mapping(state.get("bundle_select"))
        for bundle in _records(selection.get("bundles")):
            children = self._add_individual_cards(
                rows,
                _records(bundle.get("cards")),
                "bundle",
                selection_type="bundle",
            )
            rows["bundle"].append(
                [
                    self.vocabulary.lookup("selection_types", "bundle"),
                    self.vocabulary.lookup("entity_zones", "bundle"),
                ],
                [linear_feature(bundle.get("card_count", len(children)))],
                children=children,
            )

    def _add_relic_selections(
        self, rows: dict[str, _EntityRows], state: Mapping[str, object]
    ) -> None:
        for relic in _records(_mapping(state.get("relic_select")).get("relics")):
            self._append_relic(rows, relic, "selection")
        for relic in _records(_mapping(state.get("treasure")).get("relics")):
            self._append_relic(rows, relic, "treasure")

    def _add_crystal_sphere(
        self, rows: dict[str, _EntityRows], state: Mapping[str, object]
    ) -> None:
        sphere = _mapping(state.get("crystal_sphere"))
        if not sphere:
            return
        width = sphere.get("grid_width")
        height = sphere.get("grid_height")
        revealed = {
            (_integer(item.get("x")), _integer(item.get("y"))): item
            for item in _records(sphere.get("revealed_items"))
        }
        for cell in _records(sphere.get("cells")):
            item = revealed.get((_integer(cell.get("x")), _integer(cell.get("y"))), {})
            item_type = _text(cell.get("item_type") or item.get("item_type"))
            is_good = cell.get("is_good", item.get("is_good"))
            rows["crystal_cell"].append(
                [
                    self.vocabulary.lookup("crystal_item_types", item_type),
                    self.vocabulary.lookup("entity_zones", "crystal"),
                ],
                [
                    linear_feature(cell.get("x")),
                    linear_feature(cell.get("y")),
                    _coordinate_ratio(cell.get("x"), width),
                    _coordinate_ratio(cell.get("y"), height),
                    _bool_feature(cell.get("is_hidden")),
                    _bool_feature(cell.get("is_clickable")),
                    _bool_feature(cell.get("is_highlighted")),
                    _bool_feature(cell.get("is_hovered")),
                    _bool_feature(is_good),
                    linear_feature(item.get("width")),
                    linear_feature(item.get("height")),
                ],
                activity=_activity(cell, positive=("is_clickable",)),
            )
        selected_tool = _text(sphere.get("tool"))
        for tool, can_use_key in (
            ("big", "can_use_big_tool"),
            ("small", "can_use_small_tool"),
        ):
            can_use = sphere.get(can_use_key)
            rows["crystal_tool"].append(
                [
                    self.vocabulary.lookup("crystal_tools", tool),
                    self.vocabulary.lookup("entity_zones", "crystal"),
                ],
                [
                    _bool_feature(can_use),
                    _bool_feature(
                        None if selected_tool is None else selected_tool == tool
                    ),
                ],
                activity=(can_use, True) if isinstance(can_use, bool) else (False, False),
            )

    def _build_batches(
        self, rows: Mapping[str, _EntityRows]
    ) -> dict[str, TokenizedEntityBatch]:
        result = {}
        for kind in ENTITY_KINDS:
            values = rows[kind]
            categorical_width = len(ENTITY_CATEGORICAL_FIELDS[kind])
            numeric_width = len(ENTITY_NUMERIC_FIELDS[kind])
            result[kind] = TokenizedEntityBatch(
                kind=kind,
                categorical=_matrix(values.categorical, categorical_width, torch.long),
                numeric=_matrix(values.numeric, numeric_width, torch.float32),
                numeric_mask=_matrix(values.numeric_mask, numeric_width, torch.bool),
                active=torch.tensor(values.active, dtype=torch.bool),
                active_mask=torch.tensor(values.active_mask, dtype=torch.bool),
                owners=tuple(values.owners),
                children=tuple(values.children),
            )
        return result


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _records(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _text(value: object) -> str | None:
    return str(value) if value is not None else None


def _identity(record: Mapping[str, object], *, include_entity: bool = True) -> str | None:
    keys = ("id", "name") if not include_entity else ("id", "name", "entity_id")
    for key in keys:
        value = _text(record.get(key))
        if value:
            return value
    return None


def _integer(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _bool_feature(value: object) -> NumericFeature:
    return linear_feature(int(value)) if isinstance(value, bool) else NumericFeature.missing()


def _cost_feature(value: object) -> NumericFeature:
    if isinstance(value, str) and value.strip().casefold() == "x":
        return linear_feature(-1)
    return linear_feature(value)


def _upgrade_feature(card: Mapping[str, object]) -> NumericFeature:
    level = card.get("current_upgrade_level")
    if level is not None:
        return linear_feature(level)
    upgraded = card.get("is_upgraded")
    return _bool_feature(upgraded)


def _coordinate_ratio(value: object, size: object) -> NumericFeature:
    parsed_size = _integer(size)
    if parsed_size is None:
        return NumericFeature.missing()
    return ratio_feature(value, parsed_size - 1)


def _activity(
    record: Mapping[str, object],
    *,
    positive: Sequence[str] = ("active", "is_active"),
) -> tuple[bool, bool]:
    for key in positive:
        value = record.get(key)
        if isinstance(value, bool):
            return value, True
    for key in ("used", "is_used", "consumed", "is_consumed"):
        value = record.get(key)
        if isinstance(value, bool):
            return not value, True
    return False, False


def _combatant_numeric(
    combatant: Mapping[str, object], position: int
) -> tuple[NumericFeature, ...]:
    return (
        signed_log_feature(combatant.get("hp")),
        signed_log_feature(combatant.get("max_hp")),
        ratio_feature(combatant.get("hp"), combatant.get("max_hp")),
        signed_log_feature(combatant.get("block")),
        linear_feature(position),
    )


def _selected_indices(selection: Mapping[str, object]) -> set[int]:
    return {
        index
        for card in _records(selection.get("selected_cards"))
        if (index := _integer(card.get("index"))) is not None
    }


def _card_signature(card: Mapping[str, object]) -> tuple[object, ...]:
    enchantment = _mapping(card.get("enchantment"))
    return (
        _identity(card),
        _text(card.get("type")),
        _text(card.get("cost")),
        _text(card.get("star_cost")),
        _text(card.get("target_type")),
        _text(card.get("rarity")),
        card.get("is_upgraded"),
        card.get("is_upgradable"),
        card.get("current_upgrade_level"),
        card.get("max_upgrade_level"),
        card.get("is_enchanted"),
        _identity(enchantment),
    )


def _matrix(rows: Sequence[Sequence[object]], width: int, dtype: torch.dtype) -> torch.Tensor:
    if not rows:
        return torch.empty((0, width), dtype=dtype)
    return torch.tensor(rows, dtype=dtype)
