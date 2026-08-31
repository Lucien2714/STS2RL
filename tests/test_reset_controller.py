"""ResetController state-machine tests."""

import pytest

from sts2rl.actions.dispatcher import ActionDispatcher
from sts2rl.env.mcp_client import STS2ClientError
from sts2rl.env.reset import ResetController, ResetSpec


class StandardRunClient:
    def __init__(self):
        self.calls = []
        self.state = {
            "state_type": "menu",
            "menu_screen": "main",
            "options": [
                {"name": "singleplayer", "enabled": True},
                {"name": "continue", "enabled": False},
            ],
        }

    def get_state(self):
        return self.state

    def menu_select(self, option, seed=None):
        self.calls.append((option, seed))
        if option == "singleplayer":
            self.state = {
                "state_type": "menu",
                "menu_screen": "singleplayer",
                "options": ["standard", "custom", "back"],
            }
        elif option == "standard":
            self.state = self._character_state(selected=False)
        elif option == "SILENT":
            self.state = self._character_state(selected=True)
        elif option == "confirm":
            self.state = {"state_type": "map", "run": {"floor": 0}}
        else:
            raise AssertionError(f"unexpected menu option {option}")
        return {"state": self.state}

    @staticmethod
    def _character_state(selected):
        return {
            "state_type": "menu",
            "menu_screen": "character_select",
            "characters": [
                {"id": "IRONCLAD", "selected": False},
                {"id": "SILENT", "selected": selected},
            ],
            "options": [
                {"name": "IRONCLAD", "enabled": True},
                {"name": "SILENT", "enabled": True},
                {"name": "confirm", "enabled": True},
            ],
        }


class TutorialClient:
    def __init__(self):
        self.calls = []
        self.state = {"state_type": "game_over"}

    def get_state(self):
        return self.state

    def menu_select(self, option, seed=None):
        self.calls.append((option, seed))
        states = {
            "main_menu": {
                "state_type": "menu",
                "menu_screen": "main",
                "options": ["singleplayer"],
            },
            "singleplayer": {
                "state_type": "menu",
                "menu_screen": "singleplayer",
                "options": ["standard"],
            },
            "standard": {
                "state_type": "menu",
                "menu_screen": "tutorial_prompt",
                "options": ["yes", "no"],
            },
            "no": {"state_type": "map"},
        }
        self.state = states[option]
        return self.state


class StaticMenuClient:
    def __init__(self, state):
        self.state = state
        self.calls = []

    def get_state(self):
        return self.state

    def menu_select(self, option, seed=None):
        self.calls.append((option, seed))
        return self.state


def controller(client, max_transitions=10):
    return ResetController(
        client,
        ActionDispatcher(client),
        max_transitions=max_transitions,
    )


def test_standard_reset_selects_requested_character_before_confirming():
    client = StandardRunClient()

    state = controller(client).reset(ResetSpec(character=1))

    assert state["state_type"] == "map"
    assert client.calls == [
        ("singleplayer", None),
        ("standard", None),
        ("SILENT", None),
        ("confirm", None),
    ]


def test_game_over_reset_returns_to_menu_and_declines_tutorial():
    client = TutorialClient()

    state = controller(client).reset(ResetSpec())

    assert state["state_type"] == "map"
    assert client.calls == [
        ("main_menu", None),
        ("singleplayer", None),
        ("standard", None),
        ("no", None),
    ]


def test_reset_rejects_an_active_run_by_default():
    client = StaticMenuClient({"state_type": "map", "run": {"floor": 3}})

    with pytest.raises(STS2ClientError, match="another run is active"):
        controller(client).reset(ResetSpec())

    assert client.calls == []


def test_reset_can_explicitly_reuse_an_active_run():
    active_state = {"state_type": "monster", "run": {"floor": 3}}
    client = StaticMenuClient(active_state)

    state = controller(client).reset(ResetSpec(allow_active_run=True))

    assert state is active_state
    assert client.calls == []


def test_reset_rejects_disabled_menu_option_without_dispatching_it():
    client = StaticMenuClient(
        {
            "state_type": "menu",
            "menu_screen": "main",
            "options": [{"name": "singleplayer", "enabled": False}],
        }
    )

    with pytest.raises(STS2ClientError, match="not enabled"):
        controller(client).reset(ResetSpec())

    assert client.calls == []


def test_reset_detects_a_menu_that_does_not_make_progress():
    client = StaticMenuClient(
        {
            "state_type": "menu",
            "menu_screen": "main",
            "options": ["singleplayer"],
        }
    )

    with pytest.raises(STS2ClientError, match="stopped making progress"):
        controller(client).reset(ResetSpec())

    assert client.calls == [("singleplayer", None)]


def test_reset_rejects_unknown_menu_screen():
    client = StaticMenuClient(
        {"state_type": "menu", "menu_screen": "credits", "options": ["back"]}
    )

    with pytest.raises(STS2ClientError, match="unsupported menu_screen='credits'"):
        controller(client).reset(ResetSpec())
