"""Tests for MCP card and hand selection counts."""

from sts2rl.agents.candidate_dqn_agent import DQNCandidateAgent
from sts2rl.agents.event.rule_based import EventPolicy
from sts2rl.agents.orchestrator import Agent, is_battle_policy_state


def test_card_select_picks_until_required_count_before_confirming():
    """Card-select should not confirm exact-count selections early."""
    policy = EventPolicy()
    state = {
        "state_type": "card_select",
        "card_select": {
            "prompt": "Choose 2 cards",
            "cards": [
                {"index": 0, "id": "STRIKE_IRONCLAD"},
                {"index": 1, "id": "DEFEND_IRONCLAD"},
            ],
            "selected_cards": [
                {"index": 0, "id": "STRIKE_IRONCLAD"},
            ],
            "selected_count": 1,
            "min_select": 2,
            "max_select": 2,
            "required_select_count": 2,
            "remaining_to_min": 1,
            "can_confirm": True,
            "can_cancel": False,
        },
    }

    assert policy.choose_action(state) == {"type": "select_card", "index": 1}


def test_card_select_confirms_after_required_count_is_met():
    """Card-select should continue after the exact required count is selected."""
    policy = EventPolicy()
    state = {
        "state_type": "card_select",
        "card_select": {
            "prompt": "Choose 2 cards",
            "cards": [
                {"index": 0, "id": "STRIKE_IRONCLAD"},
                {"index": 1, "id": "DEFEND_IRONCLAD"},
            ],
            "selected_cards": [
                {"index": 0, "id": "STRIKE_IRONCLAD"},
                {"index": 1, "id": "DEFEND_IRONCLAD"},
            ],
            "selected_count": 2,
            "min_select": 2,
            "max_select": 2,
            "required_select_count": 2,
            "remaining_to_min": 0,
            "can_confirm": True,
            "can_cancel": False,
        },
    }

    assert policy.choose_action(state) == {"type": "confirm_selection"}


def test_orchestrator_routes_hand_select_to_battle_agent():
    """Hand-select is a battle policy state."""
    agent = Agent()
    raw_state = {
        "state_type": "hand_select",
        "hand_select": {
            "cards": [
                {"index": 0, "id": "STRIKE_IRONCLAD"},
                {"index": 1, "id": "DEFEND_IRONCLAD"},
            ],
            "selected_cards": [
                {"index": 0, "id": "STRIKE_IRONCLAD"},
            ],
            "selected_count": 1,
            "min_select": 2,
            "max_select": 2,
            "required_select_count": 2,
            "remaining_to_min": 1,
            "can_confirm": True,
        },
    }

    assert is_battle_policy_state(raw_state) is True
    assert agent.choose_action(
        {
            "screen_type": "hand_select",
            "raw_state": raw_state,
        }
    ) == {"type": "combat_select_card", "card_index": 1, "action_key": "combat_select_card:1"}


def test_orchestrator_uses_battle_agent_for_hand_select():
    """Hand-select prompts should route through the trainable battle agent."""
    agent = Agent()

    def choose_hand_select(raw_state, training=True):
        assert raw_state["state_type"] == "hand_select"
        assert training is True
        return {"type": "combat_select_card", "card_index": 1}

    agent.battle_agent.choose_action = choose_hand_select
    raw_state = {
        "state_type": "hand_select",
        "hand_select": {
            "cards": [
                {"index": 0, "id": "STRIKE_IRONCLAD"},
                {"index": 1, "id": "DEFEND_IRONCLAD"},
            ],
            "selected_cards": [
                {"index": 0, "id": "STRIKE_IRONCLAD"},
            ],
            "selected_count": 1,
            "min_select": 2,
            "max_select": 2,
            "required_select_count": 2,
            "remaining_to_min": 1,
            "can_confirm": True,
        },
    }

    action = agent.choose_action(
        {
            "screen_type": "hand_select",
            "raw_state": raw_state,
        }
    )

    assert action == {"type": "combat_select_card", "card_index": 1}


def test_hand_select_steps_are_trained_as_battle_actions():
    """Prompt selections should run battle replay/training updates."""
    agent = Agent()
    raw_state = {
        "state_type": "hand_select",
        "hand_select": {
            "cards": [{"index": 0, "id": "STRIKE_IRONCLAD"}],
            "selected_count": 0,
            "required_select_count": 1,
            "can_confirm": False,
        },
    }

    training_info = agent.train_from_step(
        raw_state,
        {"type": "combat_select_card", "card_index": 0},
        0.0,
        raw_state,
        False,
    )

    assert training_info is not None
    assert training_info["action_type"] == "combat_select_card"
    assert len(agent.battle_agent.replay_buffer) == 1


def test_hand_select_candidates_confirm_only_after_required_count():
    """Hand-select action candidates should select up to the exact count."""
    battle_agent = DQNCandidateAgent()
    raw_state = {
        "state_type": "hand_select",
        "hand_select": {
            "cards": [
                {"index": 0, "id": "STRIKE_IRONCLAD"},
                {"index": 1, "id": "DEFEND_IRONCLAD"},
            ],
            "selected_cards": [
                {"index": 0, "id": "STRIKE_IRONCLAD"},
            ],
            "selected_count": 1,
            "min_select": 2,
            "max_select": 2,
            "required_select_count": 2,
            "remaining_to_min": 1,
            "can_confirm": True,
        },
    }

    action_keys = {
        candidate["action_key"] for candidate in battle_agent.valid_action_candidates(raw_state)
    }

    assert action_keys == {"combat_select_card:1"}

    raw_state["hand_select"]["selected_cards"].append({"index": 1, "id": "DEFEND_IRONCLAD"})
    raw_state["hand_select"]["selected_count"] = 2
    raw_state["hand_select"]["remaining_to_min"] = 0

    action_keys = {
        candidate["action_key"] for candidate in battle_agent.valid_action_candidates(raw_state)
    }

    assert action_keys == {"combat_confirm_selection"}
