"""Tests for battle-agent action masking."""

from sts2rl.agents.battle.dqn_agent import BattleDQNAgent
from sts2rl.env.rewards import BattleProgressReward


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


def test_non_battle_state_has_no_dqn_action_candidates():
    """Non-battle screens should not create fake bootstrap actions."""
    agent = BattleDQNAgent()
    raw_state = {
        "state_type": "treasure",
        "player": {
            "energy": 0,
            "max_energy": 3,
            "hand": [],
            "potions": [],
            "status": [],
            "relics": [],
        },
    }

    assert agent.valid_action_candidates(raw_state) == []
    assert agent.candidate_action_vectors(raw_state) == []


def test_non_player_play_phase_has_no_dqn_q_values():
    """Enemy turns should not expose playable DQN candidates."""
    agent = BattleDQNAgent()
    raw_state = {
        "state_type": "monster",
        "battle": {
            "turn": "enemy",
            "is_play_phase": False,
            "enemies": [
                {"entity_id": "ENEMY_0", "hp": 10, "max_hp": 10},
            ],
        },
        "player": {
            "energy": 3,
            "max_energy": 3,
            "hand": [],
            "potions": [],
            "status": [],
            "relics": [],
        },
    }

    assert agent.valid_action_candidates(raw_state) == []
    assert agent.current_q_values(raw_state)["available"] is False
    assert agent.choose_action(raw_state) == {"type": "proceed"}


def test_enemy_target_type_variants_generate_targeted_actions():
    """Target strings from MCP payloads should be normalized before masking."""
    agent = BattleDQNAgent()
    raw_state = {
        "state_type": "monster",
        "battle": {
            "turn": "player",
            "is_play_phase": True,
            "enemies": [
                {"entity_id": "ENEMY_0", "hp": 10, "max_hp": 10},
                {"entity_id": "ENEMY_1", "hp": 8, "max_hp": 8},
            ],
        },
        "player": {
            "energy": 3,
            "max_energy": 3,
            "hand": [
                {
                    "index": 0,
                    "id": "STRIKE_IRONCLAD",
                    "type": "Attack",
                    "cost": "1",
                    "target_type": "Any Enemy",
                    "can_play": True,
                },
                {
                    "index": 1,
                    "id": "BASH",
                    "type": "Attack",
                    "cost": "2",
                    "target_type": "single_enemy",
                    "can_play": True,
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

    assert "play_card:STRIKE_IRONCLAD:target:0" in action_keys
    assert "play_card:STRIKE_IRONCLAD:target:1" in action_keys
    assert "play_card:BASH:target:0" in action_keys
    assert "play_card:BASH:target:1" in action_keys


def test_rewards_after_battle_counts_as_win_reward():
    """Reward screens should terminate the battle reward as a win."""
    reward_model = BattleProgressReward()
    prev_state = {
        "state_type": "elite",
        "battle": {
            "enemies": [
                {"entity_id": "ENEMY_0", "hp": 1, "max_hp": 10},
            ],
        },
        "player": {
            "hp": 50,
            "max_hp": 80,
            "gold": 20,
        },
    }
    next_state = {
        "state_type": "rewards",
        "player": {
            "hp": 50,
            "max_hp": 80,
            "gold": 20,
        },
    }

    reward, details = reward_model.compute(
        prev_state,
        next_state,
        {"type": "play_card", "card_index": 0, "target": "ENEMY_0"},
    )

    assert details["result"] == "won"
    assert reward >= 200.0
