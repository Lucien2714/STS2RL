"""Tests for environment reset menu navigation and step compatibility."""

from sts2rl.actions.game_action import GameAction
from sts2rl.env.game_env import GameEnv


class FakeClient:
    """Minimal STS2MCP client double for environment tests."""

    def __init__(self):
        self.actions = []
        self.state = {
            "state_type": "menu",
            "menu_screen": "main",
            "options": ["singleplayer"],
        }

    def get_state(self):
        return self.state

    def menu_select(self, option, seed=None):
        self.actions.append((option, seed))
        if option == "singleplayer":
            self.state = {"state_type": "menu", "menu_screen": "singleplayer"}
        elif option == "custom":
            self.state = {"state_type": "menu", "menu_screen": "custom_run"}
        elif option == "embark":
            self.state = {
                "state_type": "map",
                "run": {"act": 1, "floor": 0, "ascension": 0},
                "player": {"hp": 80, "gold": 0, "max_hp": 80},
            }
        else:
            raise AssertionError(f"unexpected menu option {option}")
        return self.state

    def end_turn(self):
        self.actions.append(("end_turn", None))
        self.state = {
            "state_type": "map",
            "run": {"act": 1, "floor": 1, "ascension": 0},
            "player": {"hp": 80, "gold": 0, "max_hp": 80},
        }
        return {"state": self.state}


def test_custom_seed_reset_embarks_before_returning_state():
    env = GameEnv(game_mode="custom", start_run_option="embark")
    env.client = FakeClient()

    state = env.reset(run_seed="ABC")

    assert state["state_type"] == "map"
    assert env.client.actions == [
        ("singleplayer", None),
        ("custom", "ABC"),
        ("embark", None),
    ]


def test_step_returns_raw_state_and_api_info_without_reward():
    env = GameEnv()
    env.client = FakeClient()
    env.action_dispatcher.client = env.client

    next_state, done, info = env.step(GameAction("end_turn"))

    assert next_state["run"]["floor"] == 1
    assert done is False
    assert info["raw_state"] == next_state
    assert info["action_error"] is False
    assert "reward_details" not in info


def test_step_converts_legacy_action_dictionary_at_environment_boundary():
    env = GameEnv()
    env.client = FakeClient()
    env.action_dispatcher.client = env.client

    next_state, done, info = env.step({"type": "end_turn"})

    assert next_state["state_type"] == "map"
    assert done is False
    assert info["action"] == {"type": "end_turn"}
    assert info["action_error"] is False
