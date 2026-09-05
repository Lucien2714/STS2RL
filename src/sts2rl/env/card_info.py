

from enum import StrEnum


class CardType(StrEnum):
    """Supported Slay the Spire card types."""

    UNKNOWN = ""
    ATTACK = "Attack"
    SKILL = "Skill"
    POWER = "Power"
    STATUS = "Status"
    CURSE = "Curse"
    QUEST = "Quest"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.value.casefold() == other.casefold()
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        return NotImplemented if result is NotImplemented else not result


class CardRarity(StrEnum):
    """Supported Slay the Spire card rarities."""

    ANCIENT = "Ancient"
    BASIC = "Basic"
    COMMON = "Common"
    UNCOMMON = "Uncommon"
    RARE = "Rare"
    CURSE = "Curse"
    EVENT = "Event"
    QUEST = "Quest"
    STATUS = "Status"
    TOKEN = "Token"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.value.casefold() == other.casefold()
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        return NotImplemented if result is NotImplemented else not result


class Card:
    def __init__(
        self,
        name: str = "",
        cost: int = 0,
        card_type: CardType = CardType.UNKNOWN,
        rarity: CardRarity = CardRarity.COMMON,
        upgrade_level: int = 0,
        enchanted: bool = False,
        enchanted_keyword: str = "", #TODO: add a proper class for enchanted keyword
    ):
        self.name = name
        self.cost = cost
        self.card_type = CardType(card_type)
        self.rarity = CardRarity(rarity)
        self.upgrade_level = upgrade_level
        self.enchanted = enchanted
        self.enchanted_keyword = enchanted_keyword

    def __repr__(self):
        return (
            f"Card(name={self.name!r}, cost={self.cost!r}, "
            f"card_type={self.card_type!r}, rarity={self.rarity!r}, "
            f"upgrade_level={self.upgrade_level!r}, enchanted={self.enchanted!r}, "
            f"enchanted_keyword={self.enchanted_keyword!r})"
        )
    
