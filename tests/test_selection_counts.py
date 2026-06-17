"""Tests for MCP card and hand selection counts."""

from sts2rl.agents.battle.dqn_agent import BattleDQNAgent
from sts2rl.agents.event.rule_based import EventPolicy
from sts2rl.agents.orchestrator import Agent, BATTLE_SCREEN_TYPES
from sts2rl.flow.battle_flow import advance_forced_hand_select_states


class ZeroReward:
    """Small reward-model double for forced-transition tests."""

    def compute(self, prev_state, next_state, action):
        return 0.0, {"type": "test", "total": 0.0}

    def action_error_reward(self, error):
        return 0.0, {"type": "action_error", "error": str(error), "total": 0.0}


class HandSelectGame:
    """Fake game that advances a two-card hand-select prompt."""

    def __init__(self):
        self.actions = []

    def step(self, action):
        self.actions.append(action)
        if len(self.actions) == 1:
            state = hand_select_state(selected_indices=[0], can_confirm=True)
        elif len(self.actions) == 2:
            state = hand_select_state(selected_indices=[0, 1], can_confirm=True)
        else:
            state = {"state_type": "monster"}
        return state, False, {"raw_state": state, "action_error": False}


def hand_select_state(selected_indices=None, can_confirm=False):
    """Build a hand-select state with a two-card exact requirement."""
    selected_indices = selected_indices or []
    cards = [
        {"index": 0, "id": "STRIKE_IRONCLAD"},
        {"index": 1, "id": "DEFEND_IRONCLAD"},
    ]
    selected_cards = [cards[index] for index in selected_indices]
    selected_count = len(selected_cards)
    return {
        "state_type": "hand_select",
        "hand_select": {
            "cards": cards,
            "selected_cards": selected_cards,
            "selected_count": selected_count,
            "min_select": 2,
            "max_select": 2,
            "required_select_count": 2,
            "remaining_to_min": max(0, 2 - selected_count),
            "can_confirm": can_confirm,
        },
    }


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


def test_auto_advance_finishes_hand_select_prompt_without_outer_loop_steps():
    """Hand-select auto-advance should select required cards and confirm."""
    agent = Agent()
    game = HandSelectGame()

    raw_state, reward, done, auto_steps = advance_forced_hand_select_states(
        game,
        agent,
        ZeroReward(),
        hand_select_state(),
    )

    assert raw_state == {"state_type": "monster"}
    assert reward == 0.0
    assert done is False
    assert game.actions == [
        {"type": "combat_select_card", "card_index": 0},
        {"type": "combat_select_card", "card_index": 1},
        {"type": "combat_confirm_selection"},
    ]
    assert len(auto_steps) == 3


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


def test_orchestrator_selects_hand_card_before_required_count():
    """The top-level forced transition should respect exact hand-select counts."""
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

    assert agent._forced_transition_action(raw_state) == {
        "type": "combat_select_card",
        "card_index": 1,
    }


def test_orchestrator_fast_selects_hand_cards_without_battle_dqn():
    """Hand-select prompts should be handled as fast forced transitions."""
    agent = Agent()

    def fail_battle_choice(*args, **kwargs):
        raise AssertionError("hand_select should not route through battle DQN")

    agent.battle_agent.choose_action = fail_battle_choice
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

    assert "hand_select" not in BATTLE_SCREEN_TYPES
    assert action == {"type": "combat_select_card", "card_index": 1}


def test_hand_select_steps_are_not_trained_as_battle_actions():
    """Prompt selections should not run DQN replay/training updates."""
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

    assert training_info is None


def test_hand_select_candidates_confirm_only_after_required_count():
    """Hand-select action candidates should select up to the exact count."""
    battle_agent = BattleDQNAgent()
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
        candidate["action_key"]
        for candidate in battle_agent.valid_action_candidates(raw_state)
    }

    assert action_keys == {"combat_select_card:1"}

    raw_state["hand_select"]["selected_cards"].append(
        {"index": 1, "id": "DEFEND_IRONCLAD"}
    )
    raw_state["hand_select"]["selected_count"] = 2
    raw_state["hand_select"]["remaining_to_min"] = 0

    action_keys = {
        candidate["action_key"]
        for candidate in battle_agent.valid_action_candidates(raw_state)
    }

    assert action_keys == {"combat_confirm_selection"}
