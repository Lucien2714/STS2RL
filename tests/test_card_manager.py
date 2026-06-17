"""Tests for normalized card identities and hand-index resolution."""

from sts2rl.data.card import Card, CardIdentity, CardManager, parse_identity_key


def test_card_identity_from_live_payload():
    """Live card payloads should normalize into stable identity keys."""
    card = Card.from_raw(
        {
            "id": "DEFEND_IRONCLAD",
            "name": "Defend",
            "type": "Skill",
            "cost": "1",
            "star_cost": None,
            "description": "Gain 5 Block.",
            "rarity": "Basic",
            "is_upgraded": False,
            "is_upgradable": True,
            "current_upgrade_level": 0,
            "max_upgrade_level": 1,
            "is_enchanted": False,
            "enchantment": None,
            "keywords": [
                {
                    "name": "Block",
                    "description": "Until next turn, prevents damage.",
                }
            ],
            "index": 0,
            "target_type": "Self",
            "can_play": True,
            "unplayable_reason": None,
        }
    )

    assert card.identity == CardIdentity("DEFEND_IRONCLAD")
    assert card.identity_key == "DEFEND_IRONCLAD"
    assert card.index == 0
    assert card.is_playable_with_energy(1) is True


def test_upgraded_and_enchanted_cards_have_distinct_identity():
    """Persistent modifiers should make otherwise identical cards different."""
    base = Card.from_raw({"id": "DEFEND_IRONCLAD", "index": 0})
    upgraded = Card.from_raw(
        {
            "id": "DEFEND_IRONCLAD",
            "is_upgraded": True,
            "current_upgrade_level": 1,
            "index": 1,
        }
    )
    enchanted = Card.from_raw(
        {
            "id": "DEFEND_IRONCLAD",
            "is_upgraded": True,
            "current_upgrade_level": 1,
            "is_enchanted": True,
            "enchantment": {"id": "SWIFT", "name": "Swift"},
            "index": 2,
        }
    )

    assert base.identity_key == "DEFEND_IRONCLAD"
    assert upgraded.identity_key == "DEFEND_IRONCLAD+"
    assert enchanted.identity_key == "DEFEND_IRONCLAD+|enchanted:SWIFT"
    assert parse_identity_key(enchanted.identity_key) == enchanted.identity


def test_card_manager_resolves_identity_to_best_hand_index():
    """Duplicate identity cards should resolve to a playable low-cost hand card."""
    manager = CardManager(
        [
            {
                "id": "STRIKE_IRONCLAD",
                "cost": "2",
                "index": 0,
                "type": "Attack",
                "can_play": True,
            },
            {
                "id": "STRIKE_IRONCLAD",
                "cost": "1",
                "index": 4,
                "type": "Attack",
                "can_play": True,
            },
            {
                "id": "STRIKE_IRONCLAD",
                "cost": "0",
                "index": 1,
                "type": "Attack",
                "is_upgraded": True,
                "current_upgrade_level": 1,
                "can_play": True,
            },
        ]
    )

    assert manager.find_hand_index("STRIKE_IRONCLAD", energy=2) == 4
    assert manager.find_hand_index("STRIKE_IRONCLAD+", energy=2) == 1
    assert manager.resolve_play_action("STRIKE_IRONCLAD+", energy=2) == {
        "type": "play_card",
        "card_index": 1,
        "card_identity": "STRIKE_IRONCLAD+",
    }
