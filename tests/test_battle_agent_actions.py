"""Tests for battle-agent action masking."""

from sts2rl.agents.battle.dqn_agent import BattleDQNAgent


def test_status_card_with_can_play_is_valid_self_action():
    """Playable status cards such as Slimed should be legal actions."""
    agent = BattleDQNAgent()
    raw_state = {
        "state_type": "monster",
        "battle": {
            "turn": "player",
            "is_play_phase": True,
            "enemies": [
                {"entity_id": "ENEMY_0", "hp": 10, "max_hp": 10},
            ],
        },
        "player": {
            "energy": 3,
            "max_energy": 3,
            "hand": [
                {
                    "index": 0,
                    "id": "SLIMED",
                    "name": "Slimed",
                    "type": "Status",
                    "cost": "1",
                    "target_type": "None",
                    "can_play": True,
                    "unplayable_reason": None,
                },
            ],
            "potions": [],
            "status": [],
            "relics": [],
        },
    }

    action_keys = {
        candidate["action_key"]
        for candidate in agent.valid_action_candidates(raw_state)
    }

    assert "play_card:SLIMED:self" in action_keys


def test_status_card_without_can_play_stays_invalid():
    """Status cards without explicit backend playability remain filtered."""
    agent = BattleDQNAgent()
    raw_state = {
        "state_type": "monster",
        "battle": {
            "turn": "player",
            "is_play_phase": True,
            "enemies": [],
        },
        "player": {
            "energy": 3,
            "max_energy": 3,
            "hand": [
                {
                    "index": 0,
                    "id": "SOME_STATUS",
                    "name": "Some Status",
                    "type": "Status",
                    "cost": "1",
                    "target_type": "None",
                },
            ],
            "potions": [],
            "status": [],
            "relics": [],
        },
    }

    action_keys = {
        candidate["action_key"]
        for candidate in agent.valid_action_candidates(raw_state)
    }

    assert "play_card:SOME_STATUS:self" not in action_keys


def test_play_card_candidates_merge_duplicates_but_split_identity_versions():
    """Same identity cards merge, while upgraded/enchanted versions split."""
    agent = BattleDQNAgent()
    raw_state = {
        "state_type": "monster",
        "battle": {
            "turn": "player",
            "is_play_phase": True,
            "enemies": [],
        },
        "player": {
            "energy": 3,
            "max_energy": 3,
            "hand": [
                {
                    "index": 0,
                    "id": "DEFEND_IRONCLAD",
                    "type": "Skill",
                    "cost": "2",
                    "target_type": "Self",
                    "can_play": True,
                },
                {
                    "index": 1,
                    "id": "DEFEND_IRONCLAD",
                    "type": "Skill",
                    "cost": "1",
                    "target_type": "Self",
                    "can_play": True,
                },
                {
                    "index": 2,
                    "id": "DEFEND_IRONCLAD",
                    "type": "Skill",
                    "cost": "1",
                    "target_type": "Self",
                    "is_upgraded": True,
                    "current_upgrade_level": 1,
                    "can_play": True,
                },
                {
                    "index": 3,
                    "id": "DEFEND_IRONCLAD",
                    "type": "Skill",
                    "cost": "1",
                    "target_type": "Self",
                    "is_upgraded": True,
                    "current_upgrade_level": 1,
                    "is_enchanted": True,
                    "enchantment": {"id": "SWIFT", "name": "Swift"},
                    "can_play": True,
                },
            ],
            "potions": [],
            "status": [],
            "relics": [],
        },
    }

    play_candidates = [
        candidate["action"]
        for candidate in agent.valid_action_candidates(raw_state)
        if candidate["action"]["type"] == "play_card"
    ]

    assert [action["action_key"] for action in play_candidates] == [
        "play_card:DEFEND_IRONCLAD:self",
        "play_card:DEFEND_IRONCLAD+:self",
        "play_card:DEFEND_IRONCLAD+|enchanted:SWIFT:self",
    ]
    assert play_candidates[0]["card_index"] == 1
