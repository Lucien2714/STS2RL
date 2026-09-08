"""Deterministic categorical vocabularies for structured game encoding."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, Iterable, Mapping, Sequence

from sts2rl.data.loader import DEFAULT_DATA_DIR, normalize_data_type


PAD_INDEX = 0
UNKNOWN_INDEX = 1
PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unknown>"


_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def normalize_token(value: str) -> str:
    """Normalize a categorical value for case-insensitive lookup."""
    return value.strip().casefold()


def parse_text_key(text_key: str) -> tuple[str | None, str | None, str | None]:
    """Split an event option's ``text_key`` into event, page and option ids.

    ``TRASH_HEAP.pages.INITIAL.options.DIVE_IN`` names the event, the page the
    option is offered on, and the option itself; a key without a ``pages``
    segment names an option offered outside any page.  A key of any other shape
    yields ``None`` for the parts it does not carry, so an unfamiliar spelling
    degrades to ``<unknown>`` rather than being sliced into nonsense.
    """
    parts = str(text_key).split(".")
    event = parts[0].strip() if parts and parts[0].strip() else None
    return event, _segment_after(parts, "pages"), _segment_after(parts, "options")


def _segment_after(parts: Sequence[str], marker: str) -> str | None:
    for position, part in enumerate(parts[:-1]):
        if normalize_token(part) == marker:
            return parts[position + 1].strip() or None
    return None


def snake_variant(value: str) -> str:
    """Return the underscore-separated reading of a CamelCase value.

    This is a second spelling tried only when the plain one misses, never the
    canonical form: tables hold both ``restsite`` and ``DEBUFF_STRONG``, so
    rewriting every token this way would fix one and break the other.
    """
    return normalize_token(_CAMEL_BOUNDARY.sub("_", value.strip()))


# Spellings the game sends that no normalization can reach from the table's id.
#
# Two mechanisms already absorb most of the difference: identifiers are tried in
# order (id, then display name), and a miss is retried as the CamelCase reading,
# which turns ``DebuffStrong`` into ``DEBUFF_STRONG``.  Neither can help when the
# game uses a *different word*: the intent is ``StatusCard`` where the table
# holds ``STATUS``, and the display-name fallback cannot rescue it either,
# because ``DEBUFF``, ``DEBUFF_STRONG`` and ``STATUS`` all present as
# "Strategic", so that alias is ambiguous and deliberately dropped.
#
# Keep this list short and only for spellings confirmed against a running game.
# It belongs here rather than in the bundled JSON, which is copied wholesale
# from spire-codex and would lose the edit on the next refresh.
API_SPELLINGS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {"intents": MappingProxyType({"StatusCard": "STATUS"})}
)


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

        ordered = tuple(canonical_by_key[key] for key in sorted(canonical_by_key))
        indexed_tokens = (PAD_TOKEN, UNKNOWN_TOKEN, *ordered)
        indices = {
            normalize_token(token): index for index, token in enumerate(indexed_tokens)
        }
        return cls(indexed_tokens, MappingProxyType(indices))

    def lookup(self, token: str | None) -> int:
        """Return PAD for missing values and UNKNOWN for unseen values.

        A miss is retried against the CamelCase reading, because the game's
        runtime enums are CamelCase where the bundled tables are snake_case:
        the API reports an intent as ``DebuffStrong`` for ``DEBUFF_STRONG``.
        """
        if token is None:
            return PAD_INDEX
        text = str(token)
        index = self._indices.get(normalize_token(text))
        if index is not None:
            return index
        return self._indices.get(snake_variant(text), UNKNOWN_INDEX)

    def token(self, index: int) -> str:
        """Return the canonical token stored at an index."""
        return self.tokens[index]

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class EventOptionVocabulary:
    """Stable indices for event options identified by event and option id.

    The API identifies an option by ``text_key``
    (``TRASH_HEAP.pages.INITIAL.options.DIVE_IN``), which is what the option
    *is*; ``title`` is display text that changes with the player's language and
    is not always distinctive — eleven events title every locked option
    "Locked", so a title-keyed table cannot tell "Requires 100 Gold" from "You
    have no Exhaust cards".

    The page is deliberately not part of the key.  ABYSSAL_BATHS offers
    ``LINGER`` on twelve pages, and one row that sees all twelve is worth more
    than twelve rows that see one each; the page travels as its own column so
    the model still knows how far into the event it is.

    ``title`` stays as a fallback for a ``text_key`` the game left unset, but
    only where it names exactly one option.  An ambiguous title is dropped
    rather than guessed, the same rule the display-name aliases follow.
    """

    pairs: tuple[tuple[str, str], ...]
    _indices: Mapping[tuple[str, str], int] = field(repr=False, compare=False)
    _titles: Mapping[tuple[str, str], int] = field(repr=False, compare=False)

    @classmethod
    def from_options(
        cls,
        options: Iterable[tuple[object, object, object]],
    ) -> EventOptionVocabulary:
        """Build a deterministic table from (event, option, title) records."""
        canonical_by_key: dict[tuple[str, str], tuple[str, str]] = {}
        keys_by_title: dict[tuple[str, str], set[tuple[str, str]]] = {}
        for event_id, option_id, title in sorted(
            (
                str(event_id),
                str(option_id),
                "" if title is None else str(title),
            )
            for event_id, option_id, title in options
            if event_id is not None and option_id is not None
        ):
            key = (normalize_token(event_id), normalize_token(option_id))
            if not all(key):
                continue
            canonical_by_key.setdefault(key, (event_id, option_id))
            if title.strip():
                keys_by_title.setdefault(
                    (normalize_token(event_id), normalize_token(title)), set()
                ).add(key)

        ordered = tuple(canonical_by_key[key] for key in sorted(canonical_by_key))
        indexed_pairs = (
            (PAD_TOKEN, PAD_TOKEN),
            (UNKNOWN_TOKEN, UNKNOWN_TOKEN),
            *ordered,
        )
        indices = {
            (normalize_token(event_id), normalize_token(option_id)): index
            for index, (event_id, option_id) in enumerate(indexed_pairs)
        }
        titles = {
            title_key: indices[next(iter(keys))]
            for title_key, keys in keys_by_title.items()
            if len(keys) == 1
        }
        return cls(indexed_pairs, MappingProxyType(indices), MappingProxyType(titles))

    def lookup(
        self,
        event_id: str | None,
        text_key: str | None = None,
        title: str | None = None,
    ) -> int:
        """Resolve an option by its text key, falling back to its title.

        The text key carries its own event id, which is preferred over the one
        the screen reports: they agree, and the key is the identity.
        """
        if text_key is not None:
            key_event, _, option_id = parse_text_key(text_key)
            resolved_event = key_event if key_event is not None else event_id
            if resolved_event is not None and option_id is not None:
                index = self._indices.get(
                    (
                        normalize_token(str(resolved_event)),
                        normalize_token(str(option_id)),
                    )
                )
                if index is not None:
                    return index
        if event_id is not None and title is not None:
            index = self._titles.get(
                (normalize_token(str(event_id)), normalize_token(str(title)))
            )
            if index is not None:
                return index
        if text_key is None and title is None:
            return PAD_INDEX
        return UNKNOWN_INDEX

    def pair(self, index: int) -> tuple[str, str]:
        """Return the canonical event and option stored at an index."""
        return self.pairs[index]

    def titles(self) -> Mapping[tuple[str, str], int]:
        """Return the unambiguous (event, title) fallbacks, for fingerprinting."""
        return self._titles

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

    FINGERPRINT_VERSION: ClassVar[int] = 3

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

        # ``event_pages`` is derived rather than read: pages are nested inside
        # the events table, and the API never names the page directly -- it is
        # read back out of each option's text key.
        table_tokens = {
            **id_tokens,
            "event_pages": list(cls._event_page_ids(raw_tables.get("events", []))),
            **cls.FIXED_TOKENS,
        }
        tables = {
            name: TokenVocabulary.from_tokens(tokens)
            for name, tokens in sorted(table_tokens.items())
        }
        event_options = EventOptionVocabulary.from_options(
            cls._event_option_records(raw_tables.get("events", []))
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
            # Explicit spellings win: they are the ones no rule can derive.
            for spelling, target in API_SPELLINGS.get(name, {}).items():
                index = table.lookup(target)
                if index == UNKNOWN_INDEX:
                    raise ValueError(
                        f"API_SPELLINGS maps {spelling!r} onto {target!r}, "
                        f"which is not in the {name!r} table"
                    )
                table_aliases[normalize_token(spelling)] = index
            aliases[name] = MappingProxyType(table_aliases)
        return cls(
            MappingProxyType(tables),
            event_options,
            MappingProxyType(aliases),
        )

    def table(self, name: str) -> TokenVocabulary:
        """Return a data or fixed-category table, accepting data aliases."""
        normalized_name = normalize_data_type(name)
        return self.tables[normalized_name]

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

    def lookup_first(self, table_name: str, candidates: Iterable[object]) -> int:
        """Return the index of the first candidate that resolves.

        The API does not always name a thing the way the bundled data does:
        statuses arrive as ``STRENGTH_POWER`` where the table holds
        ``STRENGTH``, while the display name still matches.  Trying the
        identifiers in order recovers those instead of discarding the entity as
        unknown.  All-missing yields PAD; present-but-unrecognized yields
        UNKNOWN.
        """
        seen_any = False
        for candidate in candidates:
            if candidate is None:
                continue
            seen_any = True
            index = self.lookup(table_name, candidate)
            if index != UNKNOWN_INDEX:
                return index
        return UNKNOWN_INDEX if seen_any else PAD_INDEX

    def size(self, table_name: str) -> int:
        """Return the embedding table size including PAD and UNKNOWN.

        ``event_options`` is keyed by (event, option) pairs rather than by a
        single token, so it is resolved here instead of by every caller.
        """
        if normalize_data_type(table_name) == "event_options":
            return len(self.event_options)
        return len(self.table(table_name))

    def event_option_index(
        self,
        event_id: str | None,
        text_key: str | None = None,
        title: str | None = None,
    ) -> int:
        """Return a stable index for an event option, keyed by its text key."""
        return self.event_options.lookup(event_id, text_key, title)

    def fingerprint(self) -> str:
        """Return a stable digest of every model-visible categorical index.

        Checkpoints use the digest to reject a vocabulary whose embedding rows
        would have different meanings, even when all table sizes still match.
        """
        payload = {
            "version": self.FINGERPRINT_VERSION,
            "tables": {
                name: [normalize_token(token) for token in table.tokens]
                for name, table in sorted(self.tables.items())
            },
            "api_spellings": {
                name: dict(sorted(spellings.items()))
                for name, spellings in sorted(API_SPELLINGS.items())
            },
            "event_options": [
                [normalize_token(event_id), normalize_token(option_id)]
                for event_id, option_id in self.event_options.pairs
            ],
            "event_option_titles": sorted(
                [event_id, title, index]
                for (event_id, title), index in self.event_options.titles().items()
            ),
            "aliases": {
                name: sorted((alias, index) for alias, index in aliases.items())
                for name, aliases in sorted(self.aliases.items())
            },
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    @staticmethod
    def _event_option_records(
        events: Iterable[Mapping[str, object]],
    ) -> Iterable[tuple[object, object, object]]:
        for event in events:
            event_id = event.get("id")
            if event_id is None:
                continue

            yield from GameVocabulary._options_in(event_id, event.get("options"))
            for page in GameVocabulary._pages_in(event):
                yield from GameVocabulary._options_in(event_id, page.get("options"))

    @staticmethod
    def _options_in(
        event_id: object,
        options: object,
    ) -> Iterable[tuple[object, object, object]]:
        if not isinstance(options, list):
            return
        for option in options:
            if isinstance(option, dict) and option.get("id") is not None:
                yield event_id, option["id"], option.get("title")

    @staticmethod
    def _event_page_ids(events: Iterable[Mapping[str, object]]) -> Iterable[str]:
        for event in events:
            for page in GameVocabulary._pages_in(event):
                page_id = page.get("id")
                if page_id is not None:
                    yield str(page_id)

    @staticmethod
    def _pages_in(event: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
        pages = event.get("pages")
        if not isinstance(pages, list):
            return
        for page in pages:
            if isinstance(page, dict):
                yield page
