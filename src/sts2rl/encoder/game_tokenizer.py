"""Deterministic conversion from complete observations to structured tokens.

``GameTokenizer`` handles non-map entities, the complete map DAG, and semantic
references from dynamic action candidates.  It owns no trainable parameters;
all learned representation work belongs to :class:`GameEncoder`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from types import MappingProxyType
from typing import Callable, Hashable, Mapping, Sequence

import torch

from sts2rl.actions import GameAction
from sts2rl.encoder.numeric import (
    NumericFeature,
    linear_feature,
    pack_numeric,
    ratio_feature,
    signed_log_feature,
)
from sts2rl.encoder.tokens import (
    EntityReference,
    TokenizedAction,
    TokenizedDecision,
    TokenizedEntityBatch,
    TokenizedMap,
    TokenizedState,
)
from sts2rl.encoder.vocabulary import GameVocabulary, UNKNOWN_INDEX
from sts2rl.env.types import GameObservation


GLOBAL_CATEGORICAL_FIELDS = ("state_type", "character")
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


class TokenizationError(ValueError):
    """Raised when a structured legal action cannot resolve to state entities."""


@dataclass
class _ReferenceRegistry:
    references: dict[tuple[str, Hashable], EntityReference] = field(
        default_factory=dict
    )
    ambiguous: set[tuple[str, Hashable]] = field(default_factory=set)

    def add(
        self,
        domain: str,
        handle: Hashable | None,
        reference: EntityReference,
    ) -> None:
        if handle is None:
            return
        key = (domain, handle)
        if key in self.references:
            self.references.pop(key)
            self.ambiguous.add(key)
        elif key not in self.ambiguous:
            self.references[key] = reference

    def resolve(self, domain: str, handle: Hashable) -> EntityReference | None:
        return self.references.get((domain, handle))


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
        state, _ = self._tokenize_state(observation)
        return state

    def tokenize_decision(
        self,
        observation: GameObservation,
        candidates: Sequence[GameAction],
    ) -> TokenizedDecision:
        """Tokenize a state and its ordered, structured legal-action set."""
        state, registry = self._tokenize_state(observation)
        actions = tuple(
            self._tokenize_action(observation.raw_state, action, registry)
            for action in candidates
        )
        return TokenizedDecision(state=state, actions=actions)

    def _tokenize_state(
        self, observation: GameObservation
    ) -> tuple[TokenizedState, _ReferenceRegistry]:
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
        registry = _ReferenceRegistry()
        player_ref = self._add_player(rows, player)
        self._add_cards(rows, state, player, detail_player, registry)
        self._add_inventory(rows, player, registry)
        self._add_combat_entities(rows, state, player, player_ref, registry)
        self._add_screen_entities(rows, state, registry)
        game_map = self._tokenize_map(state, registry)

        tokenized = TokenizedState(
            global_categorical=global_categorical,
            global_numeric=global_numeric,
            global_numeric_mask=global_numeric_mask,
            entities=self._build_batches(rows),
            game_map=game_map,
        )
        return tokenized, registry

    def _tokenize_action(
        self,
        state: Mapping[str, object],
        action: GameAction,
        registry: _ReferenceRegistry,
    ) -> TokenizedAction:
        action_type = action.action_type
        action_index = self.vocabulary.lookup("action_types", action_type)
        if action_index == UNKNOWN_INDEX:
            raise self._action_error(state, action, "unknown action type")
        if action_type == "menu_select":
            raise self._action_error(
                state,
                action,
                "menu options do not yet have semantic state entities",
            )

        params = action.params
        source: EntityReference | None = None
        target: EntityReference | None = None
        numeric = [NumericFeature.missing(), NumericFeature.missing()]

        source_specs = {
            "play_card": ("hand_card", "card_index", _integer),
            "use_potion": ("potion", "slot", _integer),
            "discard_potion": ("potion", "slot", _integer),
            "combat_select_card": ("hand_selection_card", "card_index", _integer),
            "claim_reward": ("reward", "index", _integer),
            "select_card_reward": ("reward_card", "card_index", _integer),
            "choose_event_option": ("event_option", "index", _integer),
            "choose_rest_option": ("rest_option", "index", _integer),
            "shop_purchase": ("shop_item", "index", _integer),
            "select_card": ("selection_card", "index", _integer),
            "select_bundle": ("bundle", "index", _integer),
            "select_relic": ("selection_relic", "index", _integer),
            "claim_treasure_relic": ("treasure_relic", "index", _integer),
            "crystal_sphere_set_tool": ("crystal_tool", "tool", _normalized_text),
        }
        spec = source_specs.get(action_type)
        if spec is not None:
            source = self._resolve_action_reference(
                state,
                action,
                registry,
                domain=spec[0],
                parameter=spec[1],
                normalize=spec[2],
            )

        if action_type in {"play_card", "use_potion"} and "target" in params:
            target = self._resolve_action_reference(
                state,
                action,
                registry,
                domain="enemy",
                parameter="target",
                normalize=_text,
            )

        if action_type == "choose_map_node":
            target = self._resolve_action_reference(
                state,
                action,
                registry,
                domain="map_candidate",
                parameter="index",
                normalize=_integer,
            )

        if action_type == "crystal_sphere_click_cell":
            x = _integer(params.get("x"))
            y = _integer(params.get("y"))
            if x is None or y is None:
                raise self._action_error(state, action, "invalid x/y cell coordinates")
            target = registry.resolve("crystal_cell", (x, y))
            if target is None:
                raise self._action_error(state, action, "cell coordinates are unresolved")
            numeric = [linear_feature(x), linear_feature(y)]
            sphere = _mapping(state.get("crystal_sphere"))
            selected_tool = _normalized_text(sphere.get("tool"))
            if selected_tool in {"big", "small"}:
                source = registry.resolve("crystal_tool", selected_tool)
                if source is None:
                    raise self._action_error(
                        state, action, "selected Crystal Sphere tool is unresolved"
                    )

        values, mask = pack_numeric(numeric)
        return TokenizedAction(
            action_type=torch.tensor(action_index, dtype=torch.long),
            numeric=values,
            numeric_mask=mask,
            source=source,
            target=target,
        )

    def _resolve_action_reference(
        self,
        state: Mapping[str, object],
        action: GameAction,
        registry: _ReferenceRegistry,
        *,
        domain: str,
        parameter: str,
        normalize: Callable[[object], Hashable | None],
    ) -> EntityReference:
        handle = normalize(action.params.get(parameter))
        if handle is None:
            raise self._action_error(
                state, action, f"missing or invalid {parameter!r} parameter"
            )
        reference = registry.resolve(domain, handle)
        if reference is None:
            raise self._action_error(
                state,
                action,
                f"{parameter!r} does not resolve to a unique {domain} entity",
            )
        return reference

    @staticmethod
    def _action_error(
        state: Mapping[str, object], action: GameAction, reason: str
    ) -> TokenizationError:
        return TokenizationError(
            "Cannot tokenize legal action "
            f"for state_type={state.get('state_type')!r}: "
            f"{action.to_dict()!r}: {reason}"
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
        registry: _ReferenceRegistry,
    ) -> None:
        self._add_grouped_cards(rows, _records(detail_player.get("deck")), "deck")
        self._add_individual_cards(
            rows,
            _records(player.get("hand")),
            "hand",
            registry=registry,
            reference_domain="hand_card",
        )
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
            registry=registry,
            reference_domain="hand_selection_card",
        )
        card_reward = _mapping(state.get("card_reward"))
        self._add_individual_cards(
            rows,
            _records(card_reward.get("cards")),
            "reward",
            registry=registry,
            reference_domain="reward_card",
        )
        card_select = _mapping(state.get("card_select"))
        self._add_individual_cards(
            rows,
            _records(card_select.get("cards")),
            "selection",
            selection_type=_text(card_select.get("screen_type")),
            selected_indices=_selected_indices(card_select),
            registry=registry,
            reference_domain="selection_card",
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
        registry: _ReferenceRegistry | None = None,
        reference_domain: str | None = None,
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
            reference = EntityReference("card", index)
            references.append(reference)
            if registry is not None and reference_domain is not None:
                registry.add(reference_domain, raw_index, reference)
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
        registry: _ReferenceRegistry,
    ) -> None:
        for relic in _records(player.get("relics")):
            self._append_relic(rows, relic, "inventory")
        for potion in _records(player.get("potions")):
            index = rows["potion"].append(
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
            registry.add(
                "potion",
                _integer(potion.get("slot")),
                EntityReference("potion", index),
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
        registry: _ReferenceRegistry,
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
            registry.add(
                "enemy",
                _text(enemy.get("entity_id")),
                EntityReference("enemy", index),
            )
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
        registry: _ReferenceRegistry,
    ) -> None:
        self._add_rewards(rows, state, registry)
        self._add_shop(rows, state, registry)
        self._add_event(rows, state, registry)
        self._add_rest(rows, state, registry)
        self._add_bundle(rows, state, registry)
        self._add_relic_selections(rows, state, registry)
        self._add_crystal_sphere(rows, state, registry)

    def _add_rewards(
        self,
        rows: dict[str, _EntityRows],
        state: Mapping[str, object],
        registry: _ReferenceRegistry,
    ) -> None:
        rewards = _mapping(state.get("rewards"))
        for reward in _records(rewards.get("items")):
            reward_type = _text(reward.get("type"))
            index = rows["reward"].append(
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
            registry.add(
                "reward",
                _integer(reward.get("index")),
                EntityReference("reward", index),
            )

    def _add_shop(
        self,
        rows: dict[str, _EntityRows],
        state: Mapping[str, object],
        registry: _ReferenceRegistry,
    ) -> None:
        shop = _mapping(state.get("shop"))
        if not shop:
            shop = _mapping(_mapping(state.get("fake_merchant")).get("shop"))
        for item in _records(shop.get("items")):
            index = rows["shop_item"].append(
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
            registry.add(
                "shop_item",
                _integer(item.get("index")),
                EntityReference("shop_item", index),
            )

    def _add_event(
        self,
        rows: dict[str, _EntityRows],
        state: Mapping[str, object],
        registry: _ReferenceRegistry,
    ) -> None:
        event = _mapping(state.get("event"))
        event_id = _text(event.get("event_id"))
        for option in _records(event.get("options")):
            locked = option.get("is_locked")
            index = rows["event_option"].append(
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
            registry.add(
                "event_option",
                _integer(option.get("index")),
                EntityReference("event_option", index),
            )

    def _add_rest(
        self,
        rows: dict[str, _EntityRows],
        state: Mapping[str, object],
        registry: _ReferenceRegistry,
    ) -> None:
        rest = _mapping(state.get("rest_site"))
        for option in _records(rest.get("options")):
            enabled = option.get("is_enabled")
            index = rows["rest_option"].append(
                [
                    self.vocabulary.lookup("rest_options", _identity(option)),
                    self.vocabulary.lookup("entity_zones", "rest"),
                ],
                [_bool_feature(enabled)],
                activity=(enabled, True) if isinstance(enabled, bool) else (False, False),
            )
            registry.add(
                "rest_option",
                _integer(option.get("index")),
                EntityReference("rest_option", index),
            )

    def _add_bundle(
        self,
        rows: dict[str, _EntityRows],
        state: Mapping[str, object],
        registry: _ReferenceRegistry,
    ) -> None:
        selection = _mapping(state.get("bundle_select"))
        for bundle in _records(selection.get("bundles")):
            children = self._add_individual_cards(
                rows,
                _records(bundle.get("cards")),
                "bundle",
                selection_type="bundle",
                registry=None,
            )
            index = rows["bundle"].append(
                [
                    self.vocabulary.lookup("selection_types", "bundle"),
                    self.vocabulary.lookup("entity_zones", "bundle"),
                ],
                [linear_feature(bundle.get("card_count", len(children)))],
                children=children,
            )
            registry.add(
                "bundle",
                _integer(bundle.get("index")),
                EntityReference("bundle", index),
            )

    def _add_relic_selections(
        self,
        rows: dict[str, _EntityRows],
        state: Mapping[str, object],
        registry: _ReferenceRegistry,
    ) -> None:
        for relic in _records(_mapping(state.get("relic_select")).get("relics")):
            index = self._append_relic(rows, relic, "selection")
            registry.add(
                "selection_relic",
                _integer(relic.get("index")),
                EntityReference("relic", index),
            )
        for relic in _records(_mapping(state.get("treasure")).get("relics")):
            index = self._append_relic(rows, relic, "treasure")
            registry.add(
                "treasure_relic",
                _integer(relic.get("index")),
                EntityReference("relic", index),
            )

    def _add_crystal_sphere(
        self,
        rows: dict[str, _EntityRows],
        state: Mapping[str, object],
        registry: _ReferenceRegistry,
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
            index = rows["crystal_cell"].append(
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
            x = _integer(cell.get("x"))
            y = _integer(cell.get("y"))
            registry.add(
                "crystal_cell",
                (x, y) if x is not None and y is not None else None,
                EntityReference("crystal_cell", index),
            )
        selected_tool = _text(sphere.get("tool"))
        for tool, can_use_key in (
            ("big", "can_use_big_tool"),
            ("small", "can_use_small_tool"),
        ):
            can_use = sphere.get(can_use_key)
            index = rows["crystal_tool"].append(
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
            registry.add(
                "crystal_tool",
                tool,
                EntityReference("crystal_tool", index),
            )

    def _tokenize_map(
        self,
        state: Mapping[str, object],
        registry: _ReferenceRegistry,
    ) -> TokenizedMap | None:
        raw_map = _mapping(state.get("map"))
        if state.get("state_type") != "map" and not raw_map:
            return None

        nodes_by_coord: dict[tuple[int, int], dict[str, object]] = {}
        for raw_node in _records(raw_map.get("nodes")):
            coord = self._require_coord(state, raw_node, "map node")
            if coord in nodes_by_coord:
                raise self._map_error(state, f"duplicate map node coordinate {coord}")
            nodes_by_coord[coord] = dict(raw_node)

        boss_coords: set[tuple[int, int]] = set()
        boss_records = _records(raw_map.get("bosses"))
        singular_boss = _mapping(raw_map.get("boss"))
        if singular_boss:
            boss_records.append(singular_boss)
        for boss in boss_records:
            coord = self._require_coord(state, boss, "boss node")
            boss_coords.add(coord)
            merged = dict(nodes_by_coord.get(coord, {}))
            merged.update(boss)
            merged["type"] = "boss"
            nodes_by_coord[coord] = merged

        ordered_coords = sorted(nodes_by_coord, key=lambda coord: (coord[1], coord[0]))
        index_by_coord = {
            coord: index for index, coord in enumerate(ordered_coords)
        }
        node_count = len(ordered_coords)
        adjacency: list[set[int]] = [set() for _ in range(node_count)]
        edges: set[tuple[int, int]] = set()
        for coord in ordered_coords:
            parent = index_by_coord[coord]
            node = nodes_by_coord[coord]
            children = node.get("children")
            if children is None:
                continue
            if not isinstance(children, list):
                raise self._map_error(
                    state, f"children for map node {coord} must be a list"
                )
            for raw_child in children:
                child_coord = self._require_coord(state, raw_child, "child node")
                child = index_by_coord.get(child_coord)
                if child is None:
                    raise self._map_error(
                        state,
                        f"child coordinate {child_coord} from {coord} is unresolved",
                    )
                edges.add((parent, child))
                adjacency[parent].add(child)

        topological = self._topological_order(state, adjacency, edges)
        current_index = self._resolve_optional_map_record(
            state,
            raw_map.get("current_position"),
            index_by_coord,
            "current position",
        )
        visited = self._resolve_map_records(
            state,
            raw_map.get("visited"),
            index_by_coord,
            "visited node",
        )

        candidate_indices: list[int] = []
        candidate_coords: set[tuple[int, int]] = set()
        for option in _records(raw_map.get("next_options")):
            coord = self._require_coord(state, option, "candidate node")
            if coord in candidate_coords:
                raise self._map_error(
                    state, f"duplicate candidate coordinate {coord}"
                )
            candidate_coords.add(coord)
            node_index = index_by_coord.get(coord)
            if node_index is None:
                raise self._map_error(
                    state, f"candidate coordinate {coord} is unresolved"
                )
            candidate_indices.append(node_index)
            registry.add(
                "map_candidate",
                _integer(option.get("index")),
                EntityReference("map_node", node_index),
            )

        boss_indices = {index_by_coord[coord] for coord in boss_coords}
        reachable = self._reachable_after_current(
            adjacency,
            candidate_indices,
            current_index,
            visited,
        )
        min_distance, max_distance = self._boss_distances(
            topological,
            adjacency,
            boss_indices,
        )

        node_categorical_rows: list[list[int]] = []
        node_numeric_rows: list[list[float]] = []
        node_numeric_masks: list[list[bool]] = []
        for index, coord in enumerate(ordered_coords):
            node = nodes_by_coord[coord]
            node_type = "boss" if index in boss_indices else _text(node.get("type"))
            node_categorical_rows.append(
                [self.vocabulary.lookup("map_node_types", node_type)]
            )
            numeric, mask = pack_numeric(
                [
                    linear_feature(coord[0]),
                    linear_feature(coord[1]),
                    _bool_feature(index == current_index),
                    _bool_feature(index in visited),
                    _bool_feature(index in candidate_indices),
                    _bool_feature(index in boss_indices),
                    _bool_feature(index in reachable),
                    linear_feature(min_distance[index]),
                    linear_feature(max_distance[index]),
                ]
            )
            node_numeric_rows.append(numeric.tolist())
            node_numeric_masks.append(mask.tolist())

        type_count_width = self.vocabulary.size("map_node_types")
        candidate_counts = torch.zeros(
            (len(candidate_indices), type_count_width), dtype=torch.float32
        )
        for candidate_row, candidate in enumerate(candidate_indices):
            for descendant in self._descendants(adjacency, [candidate]):
                type_index = node_categorical_rows[descendant][0]
                candidate_counts[candidate_row, type_index] += 1.0

        sorted_edges = sorted(edges)
        edge_index = (
            torch.tensor(sorted_edges, dtype=torch.long).transpose(0, 1).contiguous()
            if sorted_edges
            else torch.empty((2, 0), dtype=torch.long)
        )
        return TokenizedMap(
            node_categorical=_matrix(
                node_categorical_rows,
                len(MAP_CATEGORICAL_FIELDS),
                torch.long,
            ),
            node_numeric=_matrix(
                node_numeric_rows,
                len(MAP_NUMERIC_FIELDS),
                torch.float32,
            ),
            node_numeric_mask=_matrix(
                node_numeric_masks,
                len(MAP_NUMERIC_FIELDS),
                torch.bool,
            ),
            edge_index=edge_index,
            reachable_mask=torch.tensor(
                [index in reachable for index in range(node_count)],
                dtype=torch.bool,
            ),
            candidate_indices=torch.tensor(candidate_indices, dtype=torch.long),
            boss_indices=torch.tensor(sorted(boss_indices), dtype=torch.long),
            candidate_type_counts=candidate_counts,
            current_index=current_index,
        )

    def _require_coord(
        self,
        state: Mapping[str, object],
        value: object,
        label: str,
    ) -> tuple[int, int]:
        if isinstance(value, Mapping):
            col = _integer(value.get("col"))
            row = _integer(value.get("row"))
        elif isinstance(value, list | tuple) and len(value) == 2:
            col = _integer(value[0])
            row = _integer(value[1])
        else:
            col = row = None
        if col is None or row is None:
            raise self._map_error(state, f"{label} has invalid (col, row): {value!r}")
        return col, row

    def _resolve_optional_map_record(
        self,
        state: Mapping[str, object],
        value: object,
        index_by_coord: Mapping[tuple[int, int], int],
        label: str,
    ) -> int | None:
        if value is None:
            return None
        coord = self._require_coord(state, value, label)
        index = index_by_coord.get(coord)
        if index is None:
            raise self._map_error(state, f"{label} coordinate {coord} is unresolved")
        return index

    def _resolve_map_records(
        self,
        state: Mapping[str, object],
        values: object,
        index_by_coord: Mapping[tuple[int, int], int],
        label: str,
    ) -> set[int]:
        result = set()
        for value in _records(values):
            index = self._resolve_optional_map_record(
                state, value, index_by_coord, label
            )
            assert index is not None
            result.add(index)
        return result

    def _topological_order(
        self,
        state: Mapping[str, object],
        adjacency: Sequence[set[int]],
        edges: set[tuple[int, int]],
    ) -> list[int]:
        indegree = [0] * len(adjacency)
        for _, child in edges:
            indegree[child] += 1
        ready = [index for index, degree in enumerate(indegree) if degree == 0]
        heapq.heapify(ready)
        result = []
        while ready:
            parent = heapq.heappop(ready)
            result.append(parent)
            for child in sorted(adjacency[parent]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, child)
        if len(result) != len(adjacency):
            raise self._map_error(state, "map graph contains a cycle")
        return result

    @staticmethod
    def _descendants(
        adjacency: Sequence[set[int]], roots: Sequence[int]
    ) -> set[int]:
        visited: set[int] = set()
        stack = list(roots)
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(adjacency[node])
        return visited

    def _reachable_after_current(
        self,
        adjacency: Sequence[set[int]],
        candidates: Sequence[int],
        current: int | None,
        visited: set[int],
    ) -> set[int]:
        if candidates:
            roots = list(candidates)
        elif current is not None:
            roots = list(adjacency[current])
        else:
            indegree = [0] * len(adjacency)
            for children in adjacency:
                for child in children:
                    indegree[child] += 1
            roots = [index for index, degree in enumerate(indegree) if degree == 0]
        reachable = self._descendants(adjacency, roots)
        reachable.difference_update(visited)
        if current is not None:
            reachable.discard(current)
        return reachable

    @staticmethod
    def _boss_distances(
        topological: Sequence[int],
        adjacency: Sequence[set[int]],
        bosses: set[int],
    ) -> tuple[list[int | None], list[int | None]]:
        minimum: list[int | None] = [None] * len(adjacency)
        maximum: list[int | None] = [None] * len(adjacency)
        for node in reversed(topological):
            if node in bosses:
                minimum[node] = maximum[node] = 0
                continue
            child_minimums = [
                minimum[child]
                for child in adjacency[node]
                if minimum[child] is not None
            ]
            child_maximums = [
                maximum[child]
                for child in adjacency[node]
                if maximum[child] is not None
            ]
            if child_minimums:
                minimum[node] = 1 + min(child_minimums)
                maximum[node] = 1 + max(child_maximums)
        return minimum, maximum

    @staticmethod
    def _map_error(
        state: Mapping[str, object], reason: str
    ) -> TokenizationError:
        return TokenizationError(
            f"Cannot tokenize map for state_type={state.get('state_type')!r}: {reason}"
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


def _normalized_text(value: object) -> str | None:
    text = _text(value)
    return text.strip().casefold() if text is not None else None


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
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    try:
        return int(value)  # type: ignore[arg-type]
    except (OverflowError, TypeError, ValueError):
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
