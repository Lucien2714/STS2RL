"""Contract tests for raw environment reset and step behavior."""

import pytest

from sts2rl.actions.game_action import GameAction
from sts2rl.env.game_env import GameEnv
from sts2rl.env.mcp_client import STS2ClientError
from sts2rl.env.reset import ResetSpec
from sts2rl.env.types import EnvStep


class FakeClient:
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
            }
        else:
            raise AssertionError(f"unexpected menu option {option}")
        return self.state

    def end_turn(self):
        self.actions.append(("end_turn", None))
        self.state = {
            "state_type": "map",
            "run": {"act": 1, "floor": 1, "ascension": 0},
        }
        return {"state": self.state}


class FailingActionClient(FakeClient):
    def end_turn(self):
        raise STS2ClientError("backend rejected action")


def test_custom_seed_reset_embarks_before_returning_raw_state():
    client = FakeClient()
    env = GameEnv(client=client)
    spec = ResetSpec(
        character=0,
        game_mode="custom",
        run_seed="ABC",
        start_run_option="embark",
    )

    state = env.reset(spec)

    assert state["state_type"] == "map"
    assert client.actions == [
        ("singleplayer", None),
        ("custom", "ABC"),
        ("embark", None),
    ]


def test_step_returns_env_step_without_computing_reward():
    client = FakeClient()
    env = GameEnv(client=client)

    result = env.step(GameAction("end_turn"))

    assert isinstance(result, EnvStep)
    assert result.raw_state["run"]["floor"] == 1
    assert result.done is False
    assert result.info["action"] == {"type": "end_turn"}
    assert result.info["action_error"] is False
    assert "reward" not in result.info


def test_step_rejects_legacy_action_dictionary():
    env = GameEnv(client=FakeClient())

    with pytest.raises(TypeError, match="action must be GameAction"):
        env.step({"type": "end_turn"})


def test_step_returns_action_error_only_for_client_failures():
    env = GameEnv(client=FailingActionClient())

    result = env.step(GameAction("end_turn"))

    assert result.raw_state["state_type"] == "menu"
    assert result.done is False
    assert result.info == {
        "error": "backend rejected action",
        "action": {"type": "end_turn"},
        "action_error": True,
    }


def test_dispatcher_programming_errors_are_not_hidden_as_action_errors():
    env = GameEnv(client=FakeClient())

    with pytest.raises(ValueError, match="Unknown action type"):
        env.step(GameAction("not_a_real_action"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"game_mode": "invalid"},
        {"start_run_option": "invalid"},
    ],
)
def test_reset_spec_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        ResetSpec(**kwargs)
