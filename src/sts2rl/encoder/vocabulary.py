"""Deterministic categorical vocabularies for structured game encoding."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, Iterable, Mapping

from sts2rl.data.loader import DEFAULT_DATA_DIR, normalize_data_type


PAD_INDEX = 0
UNKNOWN_INDEX = 1
PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unknown>"


def normalize_token(value: str) -> str:
    """Normalize a categorical value for case-insensitive lookup."""
    return value.strip().casefold()


@dataclass(frozen=True)
class TokenVocabulary:
    """An immutable token-to-index table with stable special indices."""

    tokens: tuple[str, ...]
    _indices: Mapping[str, int] = field(repr=False, compare=False)

    @classmethod
    def from_tokens(cls, tokens: Iterable[str]) -> TokenVocabulary:
        """Build a table sorted by normalized token instead of source order."""
        canonical_by_key: dict[str, str] = {}
        for token in sorted(str(token) for token in tokens):
            key = normalize_token(token)
            if key and key not in {PAD_TOKEN, UNKNOWN_TOKEN}:
                canonical_by_key.setdefault(key, token)

        ordered = tuple(
            canonical_by_key[key] for key in sorted(canonical_by_key)
        )
        indexed_tokens = (PAD_TOKEN, UNKNOWN_TOKEN, *ordered)
        indices = {
            normalize_token(token): index
            for index, token in enumerate(indexed_tokens)
        }
        return cls(indexed_tokens, MappingProxyType(indices))

    def lookup(self, token: str | None) -> int:
        """Return PAD for missing values and UNKNOWN for unseen values."""
        if token is None:
            return PAD_INDEX
        return self._indices.get(normalize_token(str(token)), UNKNOWN_INDEX)

    def token(self, index: int) -> str:
        """Return the canonical token stored at an index."""
        return self.tokens[index]

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class EventOptionVocabulary:
    """Stable indices for event options identified by event and visible title."""

    pairs: tuple[tuple[str, str], ...]
    _indices: Mapping[tuple[str, str], int] = field(repr=False, compare=False)

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[tuple[str, str]],
    ) -> EventOptionVocabulary:
        """Build a deterministic table of normalized event/title pairs."""
        canonical_by_key: dict[tuple[str, str], tuple[str, str]] = {}
        for event_id, title in sorted(
            (str(event_id), str(title)) for event_id, title in pairs
        ):
            key = (normalize_token(event_id), normalize_token(title))
            if all(key):
                canonical_by_key.setdefault(key, (event_id, title))

        ordered = tuple(
            canonical_by_key[key] for key in sorted(canonical_by_key)
        )
        indexed_pairs = (
            (PAD_TOKEN, PAD_TOKEN),
            (UNKNOWN_TOKEN, UNKNOWN_TOKEN),
            *ordered,
        )
        indices = {
            (normalize_token(event_id), normalize_token(title)): index
            for index, (event_id, title) in enumerate(indexed_pairs)
        }
        return cls(indexed_pairs, MappingProxyType(indices))

    def lookup(self, event_id: str | None, title: str | None) -> int:
        """Look up a visible event option without depending on letter case."""
        if event_id is None or title is None:
            return PAD_INDEX
        key = (normalize_token(str(event_id)), normalize_token(str(title)))
        return self._indices.get(key, UNKNOWN_INDEX)

    def pair(self, index: int) -> tuple[str, str]:
        """Return the canonical event and title stored at an index."""
        return self.pairs[index]

    def __len__(self) -> int:
        return len(self.pairs)


@dataclass(frozen=True)
class GameVocabulary:
    """All deterministic ID and fixed-category vocabularies used by encoders."""

    FIXED_TOKENS: ClassVar[Mapping[str, tuple[str, ...]]] = MappingProxyType(
        {
            "state_types": (
                "menu",
                "unknown",
                "monster",
                "elite",
                "boss",
                "hand_select",
                "rewards",
                "card_reward",
                "map",
                "event",
                "rest_site",
                "shop",
                "fake_merchant",
                "treasure",
                "card_select",
                "bundle_select",
                "relic_select",
                "crystal_sphere",
                "game_over",
                "overlay",
                "player_detail",
            ),
            "action_types": (
                "menu_select",
                "play_card",
                "use_potion",
                "discard_potion",
                "end_turn",
                "combat_select_card",
                "combat_confirm_selection",
                "claim_reward",
                "select_card_reward",
                "skip_card_reward",
                "proceed",
                "choose_event_option",
                "advance_dialogue",
                "choose_rest_option",
                "shop_purchase",
                "choose_map_node",
                "select_card",
                "confirm_selection",
                "cancel_selection",
                "select_bundle",
                "confirm_bundle_selection",
                "cancel_bundle_selection",
                "select_relic",
                "skip_relic_selection",
                "claim_treasure_relic",
                "crystal_sphere_set_tool",
                "crystal_sphere_click_cell",
                "crystal_sphere_proceed",
            ),
            "card_types": (
                "attack",
                "skill",
                "power",
                "status",
                "curse",
                "quest",
            ),
            "rarities": (
                "ancient",
                "ancient relic",
                "basic",
                "common",
                "common relic",
                "curse",
                "uncommon",
                "uncommon relic",
                "rare",
                "rare relic",
                "event",
                "event relic",
                "none",
                "quest",
                "relic",
                "shop",
                "shop relic",
                "starter",
                "starter relic",
                "status",
                "token",
            ),
            "map_node_types": (
                "start",
                "monster",
                "elite",
                "restsite",
                "shop",
                "event",
                "treasure",
                "boss",
                "unknown",
            ),
            "target_types": (
                "none",
                "self",
                "anyenemy",
                "allenemies",
                "randomenemy",
                "anyally",
                "allallies",
                "anyplayer",
                "allplayers",
            ),
            "entity_types": (
                "state",
                "player",
                "card",
                "relic",
                "potion",
                "orb",
                "pet",
                "enemy",
                "power",
                "intent",
                "reward",
                "shop_item",
                "event_option",
                "rest_option",
                "bundle",
                "map_node",
                "crystal_cell",
                "crystal_tool",
            ),
            "card_zones": (
                "deck",
                "hand",
                "draw",
                "discard",
                "exhaust",
                "reward",
                "selection",
                "bundle",
                "shop",
            ),
            "entity_zones": (
                "battle",
                "bundle",
                "crystal",
                "deck",
                "discard",
                "draw",
                "event",
                "exhaust",
                "hand",
                "inventory",
                "rest",
                "reward",
                "selection",
                "shop",
                "treasure",
            ),
            "power_types": ("buff", "debuff"),
            "owner_types": ("player", "enemy", "pet"),
            "reward_types": (
                "gold",
                "potion",
                "relic",
                "card",
                "special_card",
                "card_removal",
            ),
            "shop_categories": ("card", "relic", "potion", "card_removal"),
            "selection_types": (
                "transform",
                "upgrade",
                "select",
                "simple_select",
                "choose",
                "bundle",
            ),
            "rest_options": (
                "dig",
                "lift",
                "recall",
                "rest",
                "smith",
                "toke",
            ),
            "crystal_item_types": ("CrystalSphereGold",),
            "crystal_tools": ("none", "big", "small"),
        }
    )

    tables: Mapping[str, TokenVocabulary]
    event_options: EventOptionVocabulary
    aliases: Mapping[str, Mapping[str, int]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_bundled_data(
        cls,
        data_dir: str | Path = DEFAULT_DATA_DIR,
    ) -> GameVocabulary:
        """Load sorted ID tables and event options from bundled JSON data."""
        data_path = Path(data_dir)
        raw_tables: dict[str, list[dict[str, object]]] = {}
        id_tokens: dict[str, list[str]] = {}

        for file_path in sorted(data_path.glob("*.json")):
            with file_path.open("r", encoding="utf-8") as file:
                raw_data = json.load(file)
            if not isinstance(raw_data, list):
                continue

            records = [item for item in raw_data if isinstance(item, dict)]
            raw_tables[file_path.stem] = records
            id_tokens[file_path.stem] = [
                str(item["id"]) for item in records if "id" in item
            ]

        table_tokens = {**id_tokens, **cls.FIXED_TOKENS}
        tables = {
            name: TokenVocabulary.from_tokens(tokens)
            for name, tokens in sorted(table_tokens.items())
        }
        event_options = EventOptionVocabulary.from_pairs(
            cls._event_option_pairs(raw_tables.get("events", []))
        )
        aliases: dict[str, Mapping[str, int]] = {}
        for name, records in raw_tables.items():
            table = tables[name]
            alias_candidates: dict[str, set[int]] = {}
            for record in records:
                item_id = record.get("id")
                if item_id is None:
                    continue
                index = table.lookup(str(item_id))
                for alias in (item_id, record.get("name")):
                    if alias is not None:
                        alias_candidates.setdefault(
                            normalize_token(str(alias)), set()
                        ).add(index)
            table_aliases = {
                alias: next(iter(indices))
                for alias, indices in alias_candidates.items()
                if len(indices) == 1
            }
            aliases[name] = MappingProxyType(table_aliases)
        return cls(
            MappingProxyType(tables),
            event_options,
            MappingProxyType(aliases),
        )

    def table(self, name: str) -> TokenVocabulary:
        """Return a data or fixed-category table, accepting data aliases."""
        normalized_name = normalize_data_type(name)
        try:
            return self.tables[normalized_name]
        except KeyError as exc:
            raise KeyError(f"Unknown vocabulary table: {normalized_name}") from exc

    def lookup(self, table_name: str, token: str | None) -> int:
        """Look up an ID or bundled display-name alias in a named table."""
        normalized_name = normalize_data_type(table_name)
        result = self.table(normalized_name).lookup(token)
        if result != UNKNOWN_INDEX or token is None:
            return result
        return self.aliases.get(normalized_name, {}).get(
            normalize_token(str(token)),
            UNKNOWN_INDEX,
        )

    def size(self, table_name: str) -> int:
        """Return the embedding table size including PAD and UNKNOWN."""
        return len(self.table(table_name))

    def event_option_index(
        self,
        event_id: str | None,
        title: str | None,
    ) -> int:
        """Return a stable index for a visible event option."""
        return self.event_options.lookup(event_id, title)

    @staticmethod
    def _event_option_pairs(
        events: Iterable[Mapping[str, object]],
    ) -> Iterable[tuple[str, str]]:
        for event in events:
            event_id = event.get("id")
            if event_id is None:
                continue

            yield from GameVocabulary._options_in(event_id, event.get("options"))
            pages = event.get("pages")
            if isinstance(pages, list):
                for page in pages:
                    if isinstance(page, dict):
                        yield from GameVocabulary._options_in(
                            event_id,
                            page.get("options"),
                        )

    @staticmethod
    def _options_in(
        event_id: object,
        options: object,
    ) -> Iterable[tuple[str, str]]:
        if not isinstance(options, list):
            return
        for option in options:
            if isinstance(option, dict) and option.get("title") is not None:
                yield str(event_id), str(option["title"])
