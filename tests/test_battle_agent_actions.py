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

    mask = agent.valid_action_mask(raw_state)

    assert mask[agent.get_action_id("play_card_0_self")] is True


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

    mask = agent.valid_action_mask(raw_state)

    assert mask[agent.get_action_id("play_card_0_self")] is False
