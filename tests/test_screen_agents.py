"""Tests for trainable non-battle screen agents (candidate-action framework)."""

import pytest

from sts2rl.agents.candidate_ppo_agent import PPOCandidateAgent
from sts2rl.agents.orchestrator import (
    SCREEN_AGENTS,
    SCREEN_NAMES,
    Agent,
    create_screen_agent,
    create_screen_agents,
    screen_name_for_state,
)
from sts2rl.encoders.rest_encoder import RestEncoder
from sts2rl.env.rewards import ScopedRewardModel
from sts2rl.training.cli import training_reward_for_state


def rest_state(enabled=True) -> dict:
    """Rest-site state with rest + smith options (smith optionally disabled)."""
    return {
        "state_type": "rest_site",
        "rest_site": {
            "options": [
                {"index": 0, "id": "rest", "name": "Rest", "is_enabled": True},
                {"index": 1, "id": "smith", "name": "Smith", "is_enabled": enabled},
            ],
        },
        "player": {"hp": 30, "max_hp": 80, "gold": 50},
        "run": {"floor": 5, "act": 1},
    }


def make_rest_agent() -> PPOCandidateAgent:
    return PPOCandidateAgent(encoder=RestEncoder(), hidden_size=16, rollout_steps=2)


def test_rest_encoder_candidates_and_dims():
    """Enabled options become candidates; encoding widths match the agent dims."""
    agent = make_rest_agent()
    state = rest_state(enabled=True)

    keys = {c["action_key"] for c in agent.valid_action_candidates(state)}
    assert keys == {"choose_rest_option:0", "choose_rest_option:1"}

    # A disabled option is excluded.
    keys_disabled = {
        c["action_key"] for c in agent.valid_action_candidates(rest_state(enabled=False))
    }
    assert keys_disabled == {"choose_rest_option:0"}

    assert (
        agent.action_key({"type": "choose_rest_option", "index": 1}, state)
        == "choose_rest_option:1"
    )
    assert len(agent.encode_state(state)) == agent.state_size
    assert len(agent.encode_action(state, {"type": "choose_rest_option", "index": 0})) == (
        agent.action_feature_size
    )


def test_orchestrator_routes_rest_to_agent_and_trains():
    """The orchestrator uses the rest agent when candidates exist and trains on it."""
    rest_agent = make_rest_agent()
    agent = Agent(battle_agent_type="PPO", screen_agents={"rest": rest_agent})

    state = rest_state(enabled=True)
    action = agent.choose_action({"screen_type": "rest_site", "raw_state": state})
    assert action["type"] == "choose_rest_option"
    assert action["action_key"] in {"choose_rest_option:0", "choose_rest_option:1"}

    info = agent.train_from_step(state, action, reward=1.0, next_raw_state=state, done=False)
    assert info is not None
    assert info["screen"] == "rest"
    assert info["replay_size"] >= 1


def test_orchestrator_falls_back_when_no_candidates():
    """With no enabled options the rest agent yields to the rule-based policy."""
    rest_agent = make_rest_agent()
    agent = Agent(battle_agent_type="PPO", screen_agents={"rest": rest_agent})

    state = {
        "state_type": "rest_site",
        "rest_site": {"options": []},
        "player": {"hp": 30, "max_hp": 80},
        "run": {"floor": 5, "act": 1},
    }
    action = agent.choose_action({"screen_type": "rest_site", "raw_state": state})
    assert action["type"] == "proceed"
    # No agent-chosen transition -> nothing trained.
    assert agent.train_from_step(state, action, 0.0, state, False) is None


def test_create_screen_agents_registers_all_screens():
    """The screen-agent factory builds one PPO agent per registered screen."""
    agents = create_screen_agents("PPO")
    assert set(agents) == set(SCREEN_NAMES)
    for screen, agent in agents.items():
        assert agent.ACTION_SCHEMA == f"{screen}_ppo_v1"
        # Each agent is its dedicated per-screen class, not a bare candidate agent.
        assert type(agent) is SCREEN_AGENTS[screen]["PPO"]


@pytest.mark.parametrize("screen", sorted(SCREEN_NAMES))
def test_create_screen_agent_picks_dedicated_class(screen):
    """The single-screen factory returns the registered DQN/PPO class per screen."""
    for agent_type in ("DQN", "PPO"):
        agent = create_screen_agent(screen, agent_type)
        assert type(agent) is SCREEN_AGENTS[screen][agent_type]
        assert agent.ACTION_SCHEMA == f"{screen}_{agent_type.lower()}_v1"


# A representative raw state per screen with at least one real candidate.
SCREEN_STATES = {
    "map": {
        "state_type": "map",
        "map": {
            "next_options": [
                {"index": 0, "type": "Monster", "col": 2, "row": 3},
                {"index": 1, "type": "Shop", "col": 3, "row": 3},
            ]
        },
        "player": {"hp": 40, "max_hp": 80, "gold": 99},
        "run": {"floor": 3, "act": 1},
    },
    "reward": {
        "state_type": "rewards",
        "rewards": {
            "items": [{"index": 0, "type": "card"}, {"index": 1, "type": "gold"}],
            "can_proceed": True,
        },
        "player": {"hp": 40, "max_hp": 80, "gold": 99},
        "run": {"floor": 3, "act": 1},
    },
    "shop": {
        "state_type": "shop",
        "shop": {
            "items": [
                {
                    "index": 0,
                    "category": "card",
                    "price": 75,
                    "is_stocked": True,
                    "can_afford": True,
                    "card_id": "OFFERING",
                },
                {
                    "index": 5,
                    "category": "relic",
                    "price": 150,
                    "is_stocked": True,
                    "can_afford": False,
                    "relic_id": "VAJRA",
                },
            ],
            "can_proceed": True,
        },
        "player": {"hp": 40, "max_hp": 80, "gold": 99},
        "run": {"floor": 3, "act": 1},
    },
    "rest": {
        "state_type": "rest_site",
        "rest_site": {
            "options": [
                {"index": 0, "id": "rest", "is_enabled": True},
                {"index": 1, "id": "smith", "is_enabled": True},
            ]
        },
        "player": {"hp": 30, "max_hp": 80, "gold": 50},
        "run": {"floor": 5, "act": 1},
    },
    "event": {
        "state_type": "event",
        "event": {
            "event_id": "NEOW",
            "in_dialogue": False,
            "options": [
                {"index": 0, "title": "A", "is_locked": False, "is_proceed": False},
                {"index": 1, "title": "B", "is_locked": True},
            ],
        },
        "player": {"hp": 40, "max_hp": 80, "gold": 99},
        "run": {"floor": 0, "act": 1},
    },
}


@pytest.mark.parametrize("screen", sorted(SCREEN_NAMES))
def test_screen_encoder_dims_are_consistent(screen):
    """Each screen's encoded widths match the agent's declared dims."""
    agent = create_screen_agents("PPO")[screen]
    state = SCREEN_STATES[screen]
    candidates = agent.valid_action_candidates(state)
    assert candidates, screen
    assert len(agent.encode_state(state)) == agent.state_size
    for candidate in candidates:
        assert len(agent.encode_action(state, candidate["action"])) == agent.action_feature_size
        assert agent.action_key(candidate["action"], state) == candidate["action_key"]


@pytest.mark.parametrize("screen", sorted(SCREEN_NAMES))
def test_orchestrator_routes_and_trains_each_screen(screen):
    """The orchestrator routes each screen to its agent and trains the transition."""
    agents = create_screen_agents("PPO")
    agent = Agent(battle_agent_type="PPO", screen_agents=agents)
    state = SCREEN_STATES[screen]
    assert screen_name_for_state(state) == screen

    action = agent.choose_action({"screen_type": state["state_type"], "raw_state": state})
    candidate_keys = {c["action_key"] for c in agents[screen].valid_action_candidates(state)}
    assert agents[screen].action_key(action, state) in candidate_keys

    info = agent.train_from_step(state, action, reward=1.0, next_raw_state=state, done=False)
    assert info is not None and info["screen"] == screen


def test_training_reward_for_state_uses_battle_scope_for_battle_agent():
    """Battle-controlled transitions should train on battle reward only."""
    reward_model = ScopedRewardModel()
    prev_state = {
        "state_type": "monster",
        "battle": {"enemies": [{"entity_id": "ENEMY_0", "hp": 1, "max_hp": 10}]},
        "player": {"hp": 50, "max_hp": 80, "gold": 20},
        "run": {"floor": 1, "act": 1},
    }
    next_state = {
        "state_type": "rewards",
        "player": {"hp": 50, "max_hp": 80, "gold": 20},
        "run": {"floor": 2, "act": 1},
    }
    total, details = reward_model.compute(prev_state, next_state, {"type": "play_card"})

    assert details["battle_reward"] != details["run_reward"]
    assert training_reward_for_state(prev_state, total, details) == details["battle_reward"]


def test_training_reward_for_state_uses_run_scope_for_screen_agents():
    """Non-battle screen transitions should train on run reward only."""
    reward_model = ScopedRewardModel()
    prev_state = SCREEN_STATES["map"]
    next_state = {
        **prev_state,
        "run": {"floor": 4, "act": 1},
    }
    total, details = reward_model.compute(prev_state, next_state, {"type": "choose_map_node"})

    assert details["battle_reward"] == 0.0
    assert details["run_reward"] > 0.0
    assert training_reward_for_state(prev_state, total, details) == details["run_reward"]
