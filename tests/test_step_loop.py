"""The step loop shared by training and evaluation."""

import pytest

from sts2rl.agents.orchestrator import Agent
from sts2rl.flow.step_loop import StepError, apply_action, decide_action

COMBAT_STATE = {
    "state_type": "monster",
    "in_battle": True,
    "battle": {
        "turn": "player",
        "is_play_phase": True,
        "enemies": [{"entity_id": "JAW_WORM_0", "id": "JAW_WORM", "hp": 40, "max_hp": 44}],
    },
    "player": {
        "hp": 70,
        "max_hp": 80,
        "energy": 0,
        "max_energy": 3,
        "hand": [],
        "potions": [],
        "status": [],
        "relics": [],
        "draw_pile": [],
        "discard_pile": [],
        "exhaust_pile": [],
    },
}


class ZeroReward:
    """Reward model double that scores every transition as zero."""

    def compute(self, prev_state, next_state, action):
        return 0.0, {"type": "test", "total": 0.0, "battle_reward": 0.0, "run_reward": 0.0}

    def action_error_reward(self, error):
        return 0.0, {"type": "action_error", "error": str(error), "total": 0.0}


class ScriptedGame:
    """Fake game returning a fixed sequence of states."""

    def __init__(self, states, error=None):
        self.states = list(states)
        self.error = error
        self.actions = []

    def step(self, action):
        self.actions.append(action)
        if self.error is not None:
            raise self.error
        state = self.states.pop(0) if self.states else {"state_type": "map"}
        done = state.get("state_type") == "game_over"
        return state, done, {"raw_state": state, "action_error": False}


def test_forced_end_turn_is_taken_without_the_agent():
    """With no playable card and no energy, end_turn is the only legal move."""
    decision = decide_action(Agent(), COMBAT_STATE, training=True)

    assert decision.forced_end_turn is True
    assert decision.agent_chose is False
    assert decision.action == {"type": "end_turn"}


def test_agent_chooses_when_it_has_a_real_choice():
    """A state with several candidates routes to the agent."""
    state = {
        **COMBAT_STATE,
        "player": {
            **COMBAT_STATE["player"],
            "energy": 3,
            "hand": [
                {"index": 0, "id": "STRIKE_IRONCLAD", "cost": 1, "type": "attack",
                 "target_type": "enemy"},
            ],
        },
    }

    decision = decide_action(Agent(), state, training=True)

    assert decision.agent_chose is True
    assert decision.forced_end_turn is False
    assert decision.skipped_agent is False


def test_apply_action_folds_auto_advanced_steps_into_one_outcome():
    """Forced end turns after the visible step are folded into its reward."""
    # The first state is another forced end_turn, so the loop auto-advances once.
    next_states = [dict(COMBAT_STATE), {"state_type": "map"}]
    game = ScriptedGame(next_states)

    outcome = apply_action(game, Agent(), ZeroReward(), COMBAT_STATE, {"type": "end_turn"})

    assert outcome.auto_steps, "the follow-up forced end_turn should be folded in"
    assert outcome.step_count == 1 + len(outcome.auto_steps)
    assert outcome.reward_details["auto_steps"] == len(outcome.auto_steps)
    assert outcome.action_error is False


def test_apply_action_raises_step_error_carrying_the_last_known_state():
    """Callers with reconnect logic need the state the failure happened at."""
    game = ScriptedGame([], error=RuntimeError("Connection refused"))

    with pytest.raises(StepError) as excinfo:
        apply_action(game, Agent(), ZeroReward(), COMBAT_STATE, {"type": "end_turn"})

    assert excinfo.value.state is COMBAT_STATE
    assert "Connection refused" in str(excinfo.value)
