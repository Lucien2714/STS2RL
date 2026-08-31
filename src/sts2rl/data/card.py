"""Card models and identity helpers for live STS2MCP card payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from sts2rl.data.loader import (
    DataIdMap,
    get_card_index,
    get_data_index_or_default,
)

UNKNOWN_CARD_ID = "UNKNOWN_CARD"

# Row 0 of each learned embedding table is reserved for unknown/none, so real
# DataIdMap indices shift up by one. Defined here (torch-free) because both the
# learned embedding in models/ and the torch-free featurizer in encoders/ must
# agree on it exactly — the embedding rows are keyed by these numbers.
EMBEDDING_RESERVED_ROWS = 1


def card_factor_indices(card: "Card | CardIdentity | dict") -> tuple[int, int, int]:
    """Resolve a card to ``(card_index, enchantment_index, upgraded)`` table rows.

    ``card_index`` / ``enchantment_index`` are embedding-table rows (0 =
    unknown/none); ``upgraded`` is 0 or 1.
    """
    identity = coerce_card_identity(card)
    card_index = get_card_index(identity.card_id, default=-1) + EMBEDDING_RESERVED_ROWS
    enchantment_index = (
        get_data_index_or_default("enchantments", identity.enchantment_id, -1)
        + EMBEDDING_RESERVED_ROWS
    )
    return card_index, enchantment_index, 1 if identity.upgrade_level > 0 else 0


def coerce_card_identity(card: "Card | CardIdentity | dict") -> "CardIdentity":
    """Return the CardIdentity for a Card, CardIdentity, or raw card dict."""
    if isinstance(card, CardIdentity):
        return card
    if isinstance(card, Card):
        return card.identity
    if isinstance(card, dict):
        return CardIdentity.from_raw(card)
    raise TypeError(f"Unsupported card type for encoding: {type(card).__name__}")


@dataclass(frozen=True)
class CardIdentity:
    """Stable policy identity for a card version.

    A card's policy identity intentionally includes persistent card modifiers
    that change card value, such as upgrades and enchantments. Transient combat
    state such as current cost or can_play remains on Card.
    """

    card_id: str
    upgrade_level: int = 0
    enchantment_id: str | None = None

    @property
    def key(self) -> str:
        """Return a compact action-safe key for this card identity."""
        parts = [self.card_id]
        if self.upgrade_level == 1:
            parts.append("+")
        elif self.upgrade_level > 1:
            parts.append(f"+{self.upgrade_level}")
        if self.enchantment_id:
            parts.append(f"|enchanted:{self.enchantment_id}")
        return "".join(parts)

    @classmethod
    def from_raw(cls, card: dict[str, Any]) -> "CardIdentity":
        """Build a stable identity from a raw card dictionary."""
        card_id = normalize_card_id(card.get("id") or card.get("name"))
        upgrade_level = parse_upgrade_level(card)
        enchantment_id = normalize_enchantment_id(card.get("enchantment"))
        if not bool(card.get("is_enchanted", enchantment_id is not None)):
            enchantment_id = None
        return cls(
            card_id=card_id,
            upgrade_level=upgrade_level,
            enchantment_id=enchantment_id,
        )


@dataclass
class Card:
    """Normalized card object for static data, deck detail, and combat hand cards."""

    id: str
    name: str = ""
    type: str = ""
    cost: Any = 0
    star_cost: Any = None
    description: str = ""
    rarity: str = ""
    is_upgraded: bool = False
    is_upgradable: bool = False
    current_upgrade_level: int = 0
    max_upgrade_level: int = 0
    is_enchanted: bool = False
    enchantment: Any = None
    keywords: list[Any] = field(default_factory=list)
    index: int | None = None
    target_type: str = ""
    can_play: bool | None = None
    unplayable_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def identity(self) -> CardIdentity:
        """Return the stable policy identity for this card."""
        raw = self.raw or {
            "id": self.id,
            "is_upgraded": self.is_upgraded,
            "current_upgrade_level": self.current_upgrade_level,
            "is_enchanted": self.is_enchanted,
            "enchantment": self.enchantment,
        }
        return CardIdentity.from_raw(raw)

    @property
    def identity_key(self) -> str:
        """Return the string key used by card-id based policy actions."""
        return self.identity.key

    def cost_for_energy(self, energy: int = 0) -> int:
        """Return numeric cost, resolving X-cost cards to available energy."""
        return parse_card_cost(self.cost, energy)

    def is_playable_with_energy(self, energy: int) -> bool:
        """Return whether the card should be considered playable."""
        if self.can_play is not None and not self.can_play:
            return False
        if self.can_play is None and self.type.lower() in {"status", "curse"}:
            return False
        if self.unplayable_reason:
            return False
        return self.cost_for_energy(energy) <= energy

    @classmethod
    def from_raw(cls, card: dict[str, Any]) -> "Card":
        """Create a normalized Card from a raw STS2MCP/static card payload."""
        target_type = card.get("target_type", card.get("target", ""))
        return cls(
            id=normalize_card_id(card.get("id") or card.get("name")),
            name=str(card.get("name", "")),
            type=str(card.get("type", "")),
            cost=card.get("cost", 0),
            star_cost=card.get("star_cost"),
            description=str(card.get("description", "")),
            rarity=str(card.get("rarity", "")),
            is_upgraded=bool(card.get("is_upgraded", card.get("upgraded", False))),
            is_upgradable=bool(card.get("is_upgradable", False)),
            current_upgrade_level=parse_upgrade_level(card),
            max_upgrade_level=parse_int(card.get("max_upgrade_level"), 0),
            is_enchanted=bool(card.get("is_enchanted", card.get("enchantment") is not None)),
            enchantment=card.get("enchantment"),
            keywords=list(card.get("keywords", [])),
            index=parse_optional_int(card.get("index")),
            target_type=str(target_type),
            can_play=card.get("can_play"),
            unplayable_reason=card.get("unplayable_reason"),
            raw=dict(card),
        )


class CardManager:
    """Manage normalized cards and resolve stable identities back to hand indices."""

    def __init__(self, cards: Iterable[dict[str, Any] | Card] = ()) -> None:
        self.cards = [card if isinstance(card, Card) else Card.from_raw(card) for card in cards]
        self.by_identity: dict[CardIdentity, list[Card]] = {}
        for card in self.cards:
            self.by_identity.setdefault(card.identity, []).append(card)

    @classmethod
    def from_state_hand(cls, raw_state: dict[str, Any]) -> "CardManager":
        """Build a manager for the current combat hand."""
        player = raw_state.get("player", {})
        hand = player.get("hand", raw_state.get("hand", []))
        return cls(hand)

    @classmethod
    def from_catalog(cls) -> "CardManager":
        """Build a manager for every bundled base card."""
        DataIdMap.ensure_loaded()
        cards = DataIdMap.data_dir / "cards.json"
        # DataStore is intentionally not used here so the manager preserves the
        # original list-shaped card payloads without relying on private maps.
        import json

        with cards.open("r", encoding="utf-8") as file:
            return cls(json.load(file))

    def identity_keys(self) -> list[str]:
        """Return sorted unique identity keys managed by this instance."""
        return sorted(identity.key for identity in self.by_identity)

    def matching_cards(self, identity: CardIdentity | str) -> list[Card]:
        """Return cards matching an identity object or identity key."""
        identity = self._coerce_identity(identity)
        return list(self.by_identity.get(identity, []))

    def find_hand_index(
        self,
        identity: CardIdentity | str,
        *,
        energy: int = 0,
        require_playable: bool = True,
    ) -> int | None:
        """Return the best current hand index for a stable card identity."""
        candidates = self.matching_cards(identity)
        if require_playable:
            candidates = [card for card in candidates if card.is_playable_with_energy(energy)]
        if not candidates:
            return None

        candidates.sort(
            key=lambda card: (
                card.cost_for_energy(energy),
                -card.current_upgrade_level,
                card.index if card.index is not None else 999,
            )
        )
        return candidates[0].index

    def resolve_play_action(
        self,
        identity: CardIdentity | str,
        *,
        energy: int = 0,
        target: str | None = None,
        target_index: int | None = None,
    ) -> dict[str, Any] | None:
        """Create a play_card action by resolving identity to a hand index."""
        card_index = self.find_hand_index(identity, energy=energy)
        if card_index is None:
            return None
        action: dict[str, Any] = {
            "type": "play_card",
            "card_index": card_index,
            "card_identity": self._coerce_identity(identity).key,
        }
        if target is not None:
            action["target"] = target
        if target_index is not None:
            action["target_index"] = target_index
        return action

    def _coerce_identity(self, identity: CardIdentity | str) -> CardIdentity:
        if isinstance(identity, CardIdentity):
            return identity
        for candidate in self.by_identity:
            if candidate.key == identity:
                return candidate
        return parse_identity_key(identity)


def parse_identity_key(identity_key: str) -> CardIdentity:
    """Parse a CardIdentity key produced by CardIdentity.key."""
    base, _, enchantment_part = identity_key.partition("|enchanted:")
    upgrade_level = 0
    if "+" in base:
        card_id, _, upgrade_part = base.partition("+")
        upgrade_level = parse_int(upgrade_part, 1)
    else:
        card_id = base
    enchantment_id = enchantment_part or None
    return CardIdentity(
        card_id=normalize_card_id(card_id),
        upgrade_level=upgrade_level,
        enchantment_id=normalize_enchantment_id(enchantment_id),
    )


def normalize_card_id(value: object) -> str:
    """Normalize a card id/name for identity keys."""
    if value is None:
        return UNKNOWN_CARD_ID
    return str(value).strip().replace(" ", "_").upper() or UNKNOWN_CARD_ID


def normalize_enchantment_id(value: object) -> str | None:
    """Normalize an enchantment object, id, or display name."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("id") or value.get("name")
    if value is None:
        return None
    normalized = str(value).strip().replace(" ", "_").upper()
    return normalized or None


def parse_upgrade_level(card: dict[str, Any]) -> int:
    """Return a normalized upgrade level from raw card fields."""
    explicit_level = parse_optional_int(card.get("current_upgrade_level"))
    if explicit_level is not None:
        return max(0, explicit_level)
    return 1 if bool(card.get("is_upgraded", card.get("upgraded", False))) else 0


def parse_card_cost(value: object, energy: int = 0, default: int = 0) -> int:
    """Parse card cost, treating X-cost cards as the current energy."""
    if isinstance(value, str) and value.strip().upper() == "X":
        return max(0, parse_int(energy, 0))
    return parse_int(value, default)


def parse_optional_int(value: object) -> int | None:
    """Parse an optional integer value."""
    if value is None:
        return None
    return parse_int(value, 0)


def parse_int(value: object, default: int = 0) -> int:
    """Parse an integer-like value without raising for malformed data."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list, tuple, set)):
        return default
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
    try:
        return int(value)
    except (OverflowError, TypeError, ValueError):
        return default
