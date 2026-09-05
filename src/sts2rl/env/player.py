"""Local player model built from bundled data and live STS2MCP detail payloads."""

from sts2rl.data.loader import (
    DataStore,
    DataIdMap,
    CardInstance,
    get_potion_index,
    get_relic_index,
    load_card,
)


CHARACTER_MAP = {
    0: "IRONCLAD",
    1: "SILENT",
    2: "REGENT",
    3: "NECROBINDER",
    4: "DEFECT",
}

CHARACTER_DETAIL_MAP = {
    "IRONCLAD": "IRONCLAD",
    "THE_IRONCLAD": "IRONCLAD",
    "SILENT": "SILENT",
    "THE_SILENT": "SILENT",
    "REGENT": "REGENT",
    "THE_REGENT": "REGENT",
    "NECROBINDER": "NECROBINDER",
    "THE_NECROBINDER": "NECROBINDER",
    "DEFECT": "DEFECT",
    "THE_DEFECT": "DEFECT",
}


class Player:
    """Represent character resources, deck, relics, potions, and combat stats."""

    def __init__(self, character: int = 0):
        if character not in CHARACTER_MAP:
            raise ValueError(f"Invalid character index: {character}")

        self.character: int = character
        self.character_id: str = CHARACTER_MAP[character]

        character_data = DataStore.get("characters", self.character_id)

        self.max_hp: int = character_data.get("starting_hp", 0)
        self.current_hp: int = self.max_hp

        self.gold: int = character_data.get("starting_gold", 0)

        self.max_energy: int = character_data.get("max_energy", 3)
        self.current_energy: int = self.max_energy

        self.current_deck: list[CardInstance] = []
        self.current_relic: list[int] = []
        self.potion: list[int] = []
        self.deck_details: list[dict] = []
        self.relic_details: list[dict] = []
        self.potion_details: list[dict] = []
        self.status: list[dict] = []
        self.max_potion_slots: int = 3
        self.deck_count: int = 0
        self.player_detail: dict = {}

        # Defect's orb slots
        self.orb_slot: list[int] = []

        self.buffs: dict[int, int] = {}
        self.block: int = 0

        self.load_starting_deck(character_data)
        self.load_starting_relics(character_data)
        self.load_orb_slots(character_data)
        self.deck_count = len(self.current_deck)

    def load_starting_deck(self, character_data: dict) -> None:
        """Load starting deck card instances from character data."""
        starting_deck = character_data.get("starting_deck", [])

        for card_id in starting_deck:
            card_data = DataStore.get("cards", card_id)
            card_instance = load_card(card_data)
            self.current_deck.append(card_instance)

    def load_starting_relics(self, character_data: dict) -> None:
        """Load starting relic ids as numeric relic indices."""
        starting_relics = character_data.get("starting_relics", [])

        for relic_id in starting_relics:
            relic_index = DataIdMap.get_index("relics", relic_id)
            self.current_relic.append(relic_index)

    def load_orb_slots(self, character_data: dict) -> None:
        """Initialize character-specific orb slots when present."""
        orb_slots = character_data.get("orb_slots")

        if orb_slots is None:
            self.orb_slot = []
        else:
            self.orb_slot = [0 for _ in range(orb_slots)]

    def add_card(self, card_id: str) -> None:
        """Add a card instance to the player's current deck."""
        card_data = DataStore.get("cards", card_id)
        card_instance = load_card(card_data)
        self.current_deck.append(card_instance)

    def add_relic(self, relic_id: str) -> None:
        """Add a relic index to the player's current relic list."""
        relic_index = DataIdMap.get_index("relics", relic_id)
        self.current_relic.append(relic_index)

    def add_potion(self, potion_id: str) -> None:
        """Add a potion index to the player's potion list."""
        potion_index = DataIdMap.get_index("potions", potion_id)
        self.potion.append(potion_index)

    def update_from_detail(self, detail: dict) -> None:
        """Refresh player resources and inventory from a player-detail response."""
        if detail.get("status") != "ok":
            raise ValueError(f"Player detail response is not ok: {detail}")

        player = detail.get("player", {})
        self.player_detail = detail
        self.character_id = self._character_id_from_detail(
            player.get("character"),
            self.character_id,
        )
        self.max_hp = self._parse_int(player.get("max_hp"), self.max_hp)
        self.current_hp = self._parse_int(player.get("hp"), self.current_hp)
        self.gold = self._parse_int(player.get("gold"), self.gold)
        self.block = self._parse_int(player.get("block"), self.block)
        self.max_potion_slots = self._parse_int(
            player.get("max_potion_slots"),
            self.max_potion_slots,
        )
        self.status = list(player.get("status", []))

        self.deck_details = list(player.get("deck", []))
        self.deck_count = self._parse_int(
            player.get("deck_count"),
            len(self.deck_details),
        )
        self.current_deck = [
            self._card_instance_from_detail(card)
            for card in self.deck_details
        ]

        self.relic_details = list(player.get("relics", []))
        self.current_relic = [
            get_relic_index(relic.get("id") or relic.get("name"), default=-1)
            for relic in self.relic_details
        ]

        self.potion_details = list(player.get("potions", []))
        self.potion = [
            get_potion_index(potion.get("id") or potion.get("name"), default=-1)
            for potion in self.potion_details
        ]

    def take_damage(self, amount: int) -> None:
        """Apply incoming damage after block and update current HP."""
        damage_after_block = max(0, amount - self.block)
        self.block = max(0, self.block - amount)
        self.current_hp = max(0, self.current_hp - damage_after_block)

    def heal(self, amount: int) -> None:
        """Heal without exceeding maximum HP."""
        self.current_hp = min(self.max_hp, self.current_hp + amount)

    def gain_block(self, amount: int) -> None:
        """Increase current block."""
        self.block += amount

    def reset_block(self) -> None:
        """Clear block at the end of a turn or combat."""
        self.block = 0

    def reset_energy(self) -> None:
        """Restore current energy to the character maximum."""
        self.current_energy = self.max_energy

    def spend_energy(self, amount: int) -> bool:
        """Spend energy when available and report whether it succeeded."""
        if self.current_energy < amount:
            return False

        self.current_energy -= amount
        return True

    def add_buff(self, buff_id: int, amount: int) -> None:
        """Increase a buff stack by id."""
        self.buffs[buff_id] = self.buffs.get(buff_id, 0) + amount

    def remove_buff(self, buff_id: int) -> None:
        """Remove a buff by id when present."""
        if buff_id in self.buffs:
            del self.buffs[buff_id]

    def is_dead(self) -> bool:
        """Return whether the player has zero HP."""
        return self.current_hp <= 0

    def _card_instance_from_detail(self, card: dict) -> CardInstance:
        """Create a card instance from live deck detail payload."""
        return CardInstance(
            card_id=str(card.get("id") or card.get("name") or ""),
            regular_cost=self._parse_card_cost(card.get("cost"), 0),
            star_cost=self._parse_card_cost(card.get("star_cost"), 0),
            upgraded=bool(card.get("is_upgraded", False)),
        )

    def _character_id_from_detail(self, value: object, default: str) -> str:
        """Normalize display names or ids from player-detail into character ids."""
        if value is None:
            return default
        key = str(value).strip().replace(" ", "_").upper()
        return CHARACTER_DETAIL_MAP.get(key, default)

    def _parse_int(self, value: object, default: int = 0) -> int:
        """Parse an integer-like value with a safe default."""
        try:
            return int(value)
        except (OverflowError, TypeError, ValueError):
            return default

    def _parse_card_cost(self, value: object, default: int = 0) -> int:
        """Parse a card cost, using -1 for X-cost cards."""
        if isinstance(value, str) and value.strip().upper() == "X":
            return -1
        return self._parse_int(value, default)
