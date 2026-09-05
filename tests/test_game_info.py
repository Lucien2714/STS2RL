import pytest

from sts2rl.env.card_info import Card, CardRarity, CardType


@pytest.mark.parametrize(
    ("card_type", "value"),
    [
        (CardType.ATTACK, "Attack"),
        (CardType.SKILL, "Skill"),
        (CardType.POWER, "Power"),
        (CardType.STATUS, "Status"),
        (CardType.CURSE, "Curse"),
        (CardType.QUEST, "Quest"),
    ],
)
def test_card_type_values(card_type, value):
    assert card_type.value == value


def test_card_uses_card_type_enum():
    card = Card(name="Strike", card_type=CardType.ATTACK)

    assert card.card_type is CardType.ATTACK


def test_card_coerces_matching_string_to_card_type():
    card = Card(card_type="Skill")

    assert card.card_type is CardType.SKILL


def test_card_defaults_to_unknown_type():
    assert Card().card_type is CardType.UNKNOWN


def test_card_type_equals_string_case_insensitively():
    assert CardType.ATTACK == "attack"
    assert CardType.ATTACK == "ATTACK"
    assert "attack" == CardType.ATTACK
    assert not CardType.ATTACK != "attack"
    assert CardType.ATTACK != "skill"


@pytest.mark.parametrize(
    ("rarity", "value"),
    [
        (CardRarity.ANCIENT, "Ancient"),
        (CardRarity.BASIC, "Basic"),
        (CardRarity.COMMON, "Common"),
        (CardRarity.UNCOMMON, "Uncommon"),
        (CardRarity.RARE, "Rare"),
        (CardRarity.CURSE, "Curse"),
        (CardRarity.EVENT, "Event"),
        (CardRarity.QUEST, "Quest"),
        (CardRarity.STATUS, "Status"),
        (CardRarity.TOKEN, "Token"),
    ],
)
def test_card_rarity_values(rarity, value):
    assert rarity.value == value


def test_card_uses_card_rarity_enum():
    card = Card(rarity=CardRarity.RARE)

    assert card.rarity is CardRarity.RARE


def test_card_coerces_matching_string_to_card_rarity():
    card = Card(rarity="Uncommon")

    assert card.rarity is CardRarity.UNCOMMON


def test_card_defaults_to_common_rarity():
    assert Card().rarity is CardRarity.COMMON


def test_card_rarity_equals_string_case_insensitively():
    assert CardRarity.COMMON == "common"
    assert CardRarity.COMMON == "COMMON"
    assert "common" == CardRarity.COMMON
    assert not CardRarity.COMMON != "common"
    assert CardRarity.COMMON != "rare"


def test_card_repr_includes_all_fields_and_enum_types():
    card = Card(
        name="Strike",
        cost=1,
        card_type=CardType.ATTACK,
        rarity=CardRarity.BASIC,
        upgrade_level=1,
        enchanted=True,
        enchanted_keyword="Flame",
    )

    assert repr(card) == (
        "Card(name='Strike', cost=1, card_type=<CardType.ATTACK: 'Attack'>, "
        "rarity=<CardRarity.BASIC: 'Basic'>, upgrade_level=1, enchanted=True, "
        "enchanted_keyword='Flame')"
    )
