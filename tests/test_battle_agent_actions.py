"""Tests for battle-agent action masking."""

import torch

from sts2rl.agents.battle.dqn_agent import BattleDQNAgent
from sts2rl.agents.event.rule_based import EventPolicy
from sts2rl.agents.orchestrator import Agent
from sts2rl.env.constants import (
    BATTLE_GOLD_LOSS_PENALTY,
    BATTLE_HP_LOSS_PENALTY,
    BATTLE_LOSS_PENALTY,
    BATTLE_MAX_HP_LOSS_PENALTY,
    BATTLE_POTION_USE_PENALTY,
    BATTLE_UNUSED_ENERGY_PENALTY,
    BATTLE_WIN_REWARD,
    RUN_ACT_REWARD,
    RUN_FLOOR_REWARD,
    RUN_GAME_OVER_PENALTY,
)
from sts2rl.env.rewards import ScopedRewardModel


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
        candidate["action_key"] for candidate in agent.valid_action_candidates(raw_state)
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
        candidate["action_key"] for candidate in agent.valid_action_candidates(raw_state)
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
        candidate["action_key"] for candidate in agent.valid_action_candidates(raw_state)
    }

    assert "play_card:STRIKE_IRONCLAD:target:0" in action_keys
    assert "play_card:STRIKE_IRONCLAD:target:1" in action_keys
    assert "play_card:BASH:target:0" in action_keys
    assert "play_card:BASH:target:1" in action_keys


def test_in_battle_card_select_generates_per_card_candidates():
    """Battle-context card_select should expose each concrete card as a candidate."""
    agent = BattleDQNAgent()
    raw_state = {
        "state_type": "card_select",
        "in_battle": True,
        "card_select": {
            "screen_type": "simple_select",
            "prompt": "Choose a card to put on top of your Draw Pile.",
            "cards": [
                {"index": 0, "id": "STRIKE_IRONCLAD", "type": "Attack", "cost": "1"},
                {"index": 1, "id": "DEFEND_IRONCLAD", "type": "Skill", "cost": "1"},
            ],
            "selected_count": 0,
            "min_select": 1,
            "remaining_to_min": 1,
            "max_select": 1,
            "required_select_count": 1,
            "can_confirm": False,
            "can_cancel": False,
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

    action_keys = {
        candidate["action_key"] for candidate in agent.valid_action_candidates(raw_state)
    }

    assert action_keys == {"select_card:0", "select_card:1"}
    assert agent.action_key({"type": "select_card", "index": 1}, raw_state) == "select_card:1"
    assert len(agent.encode_action(raw_state, {"type": "select_card", "index": 1})) == (
        agent.action_feature_size
    )


def test_in_battle_card_select_routes_to_battle_agent():
    """Only battle-context card_select should route through the battle policy."""
    agent = Agent()

    def choose_card_select(raw_state, training=True):
        assert raw_state["state_type"] == "card_select"
        assert raw_state["in_battle"] is True
        return {"type": "select_card", "index": 1}

    agent.battle_agent.choose_action = choose_card_select
    raw_state = {
        "state_type": "card_select",
        "in_battle": True,
        "card_select": {
            "cards": [
                {"index": 0, "id": "STRIKE_IRONCLAD"},
                {"index": 1, "id": "DEFEND_IRONCLAD"},
            ],
            "selected_count": 0,
            "required_select_count": 1,
            "can_confirm": False,
        },
    }

    assert agent.choose_action(
        {
            "screen_type": "card_select",
            "raw_state": raw_state,
        }
    ) == {"type": "select_card", "index": 1}


def test_non_battle_card_select_still_uses_event_policy():
    """Non-battle card_select remains outside the battle agent."""
    policy = EventPolicy()
    agent = Agent()
    raw_state = {
        "state_type": "card_select",
        "in_battle": False,
        "card_select": {
            "cards": [
                {"index": 0, "id": "STRIKE_IRONCLAD"},
                {"index": 1, "id": "DEFEND_IRONCLAD"},
            ],
            "selected_count": 0,
            "required_select_count": 1,
            "can_confirm": False,
        },
    }

    assert agent.choose_action(
        {
            "screen_type": "card_select",
            "raw_state": raw_state,
        }
    ) == policy.choose_action(raw_state)


def test_rewards_after_battle_counts_as_win_reward():
    """Reward screens should terminate the battle reward as a win."""
    reward_model = ScopedRewardModel()
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
    assert details["win_reward"] == BATTLE_WIN_REWARD
    assert details["enemy_hp_lost"] == 1
    assert details["enemies_killed"] == 1
    assert details["enemy_damage_reward"] == 0.0
    assert details["enemy_kill_reward"] == 0.0
    assert details["battle_reward"] == BATTLE_WIN_REWARD
    assert details["run_reward"] == 0.0
    assert reward == details["total"] == BATTLE_WIN_REWARD


def test_battle_reward_applies_resource_penalties_to_battle_scope():
    """HP, potion, gold, and max HP penalties should stay in the battle scope."""
    reward_model = ScopedRewardModel()
    prev_state = {
        "state_type": "monster",
        "battle": {"enemies": [{"entity_id": "ENEMY_0", "hp": 1, "max_hp": 10}]},
        "player": {"hp": 50, "max_hp": 80, "gold": 20},
    }
    next_state = {
        "state_type": "rewards",
        "player": {"hp": 45, "max_hp": 75, "gold": 10},
    }

    reward, details = reward_model.compute(prev_state, next_state, {"type": "use_potion"})

    expected = (
        BATTLE_WIN_REWARD
        - 5 * BATTLE_HP_LOSS_PENALTY
        - BATTLE_POTION_USE_PENALTY
        - 10 * BATTLE_GOLD_LOSS_PENALTY
        - 5 * BATTLE_MAX_HP_LOSS_PENALTY
    )
    assert details["battle_reward"] == expected
    assert details["run_reward"] == 0.0
    assert reward == details["total"] == expected


def test_battle_loss_returns_negative_battle_terminal_reward():
    """Game over from battle should emit the battle loss reward and run death penalty."""
    reward_model = ScopedRewardModel()
    prev_state = {
        "state_type": "monster",
        "battle": {"enemies": [{"entity_id": "ENEMY_0", "hp": 10, "max_hp": 10}]},
        "player": {"hp": 1, "max_hp": 80, "gold": 20},
    }
    next_state = {
        "state_type": "game_over",
        "player": {"hp": 0, "max_hp": 80, "gold": 20},
    }

    reward, details = reward_model.compute(prev_state, next_state, {"type": "end_turn"})

    assert details["result"] == "lost"
    assert details["battle_reward"] == -BATTLE_LOSS_PENALTY - BATTLE_HP_LOSS_PENALTY
    assert details["run_reward"] == -RUN_GAME_OVER_PENALTY
    assert reward == details["battle_reward"] + details["run_reward"]


def test_unused_energy_no_longer_changes_training_reward():
    """Ending a turn with unused energy should be diagnostic-only by default."""
    reward_model = ScopedRewardModel()
    prev_state = {
        "state_type": "monster",
        "battle": {"enemies": [{"entity_id": "ENEMY_0", "hp": 10, "max_hp": 10}]},
        "player": {"hp": 50, "max_hp": 80, "gold": 20, "energy": 3},
    }
    next_state = {
        "state_type": "monster",
        "battle": {"enemies": [{"entity_id": "ENEMY_0", "hp": 10, "max_hp": 10}]},
        "player": {"hp": 50, "max_hp": 80, "gold": 20, "energy": 0},
    }

    reward, details = reward_model.compute(prev_state, next_state, {"type": "end_turn"})

    assert BATTLE_UNUSED_ENERGY_PENALTY == 0.0
    assert details["end_turn_energy_penalty"] == 0.0
    assert details["battle_reward"] == 0.0
    assert reward == 0.0


def test_run_reward_tracks_floor_act_and_game_over_separately():
    """Run-progress reward should not leak into battle reward."""
    reward_model = ScopedRewardModel()
    prev_state = {
        "state_type": "map",
        "run": {"floor": 15, "act": 1},
        "player": {"hp": 50, "max_hp": 80, "gold": 20},
    }
    next_state = {
        "state_type": "game_over",
        "run": {"floor": 17, "act": 2},
        "player": {"hp": 0, "max_hp": 80, "gold": 20},
    }

    reward, details = reward_model.compute(prev_state, next_state, {"type": "proceed"})

    expected_run = 2 * RUN_FLOOR_REWARD + RUN_ACT_REWARD - RUN_GAME_OVER_PENALTY
    assert details["battle_reward"] == 0.0
    assert details["run_reward"] == expected_run
    assert reward == details["total"] == expected_run


def test_dqn_greedy_breaks_ties_randomly_not_always_end_turn():
    """With equal candidate Q-values, greedy selection must not always pick the
    first candidate (end_turn); ties are broken randomly."""
    agent = BattleDQNAgent(hidden_size=16)
    agent.epsilon = 0.0  # pure greedy

    # Force every candidate to score identically by zeroing the output layer.
    with torch.no_grad():
        final_layer = agent.model.net[-1]
        final_layer.weight.zero_()
        final_layer.bias.zero_()

    raw_state = {
        "state_type": "monster",
        "battle": {
            "turn": "player",
            "is_play_phase": True,
            "enemies": [{"entity_id": "ENEMY_0", "hp": 10, "max_hp": 10}],
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
                    "target_type": "Enemy",
                    "can_play": True,
                },
            ],
            "potions": [],
            "status": [],
            "relics": [],
        },
    }

    candidate_keys = {c["action_key"] for c in agent.valid_action_candidates(raw_state)}
    assert "end_turn" in candidate_keys
    assert len(candidate_keys) >= 2

    chosen = set()
    for _ in range(60):
        action = agent.choose_action(raw_state, training=False)
        key = agent.action_key(action, raw_state)
        assert key in candidate_keys
        chosen.add(key)

    # Old behavior (first-index argmax) would always return end_turn.
    assert chosen != {"end_turn"}
    assert len(chosen) >= 2


def test_potion_candidates_include_discard():
    """Every held potion is discardable; only combat-usable potions can be used."""
    agent = BattleDQNAgent()
    raw_state = {
        "state_type": "monster",
        "battle": {
            "turn": "player",
            "is_play_phase": True,
            "enemies": [{"entity_id": "JAW_WORM_0", "hp": 20, "max_hp": 40}],
        },
        "player": {
            "energy": 3,
            "max_energy": 3,
            "hand": [],
            "potions": [
                {"slot": 0, "id": "FIRE_POTION", "target_type": "Enemy", "can_use_in_combat": True},
                {
                    "slot": 1,
                    "id": "FAIRY_POTION",
                    "target_type": "None",
                    "can_use_in_combat": False,
                },
            ],
            "status": [],
            "relics": [],
        },
    }

    action_keys = {
        candidate["action_key"] for candidate in agent.valid_action_candidates(raw_state)
    }

    # Both potions can be discarded, even the one that cannot be used in combat.
    assert "discard_potion:0" in action_keys
    assert "discard_potion:1" in action_keys
    # The enemy-target potion is usable; the non-combat potion is not.
    assert "use_potion:0:target:0" in action_keys
    assert "use_potion:1:self" not in action_keys

    assert agent.action_key({"type": "discard_potion", "slot": 1}, raw_state) == "discard_potion:1"
    assert len(agent.encode_action(raw_state, {"type": "discard_potion", "slot": 0})) == (
        agent.action_feature_size
    )
