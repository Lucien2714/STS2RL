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


def controller(client, max_transitions=10, start_poll_attempts=5):
    return ResetController(
        client,
        ActionDispatcher(client),
        max_transitions=max_transitions,
        start_poll_attempts=start_poll_attempts,
        start_poll_seconds=0.0,
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


class TopLevelSelectionClient:
    """Reproduces the live menu: selection is top-level, not per character.

    Each character entry carries no ``selected`` flag, so reading only those
    flags makes the already-selected character look unselected and reset
    re-sends the same option until the loop detector fires.
    """

    def __init__(self, selected="IRONCLAD"):
        self.calls = []
        self.selected = selected
        self.state = {
            "state_type": "menu",
            "menu_screen": "main",
            "options": ["singleplayer", "multiplayer", "settings", "quit"],
        }

    def get_state(self):
        return self.state

    def menu_select(self, option, seed=None):
        self.calls.append((option, seed))
        if option == "singleplayer":
            self.state = {
                "state_type": "menu",
                "menu_screen": "singleplayer",
                "options": ["standard", "daily", "custom", "back"],
            }
        elif option == "standard":
            self.state = self._character_select()
        elif option in {"IRONCLAD", "SILENT"}:
            self.selected = option
            self.state = self._character_select()
        elif option == "confirm":
            self.state = {"state_type": "map", "run": {"floor": 0}}
        else:
            raise AssertionError(f"unexpected menu option {option}")
        return {"state": self.state}

    def _character_select(self):
        return {
            "state_type": "menu",
            "menu_screen": "character_select",
            "characters": [{"id": "IRONCLAD"}, {"id": "SILENT"}],
            "selected": {"character": self.selected},
            "options": [
                {"name": "IRONCLAD", "enabled": True},
                {"name": "SILENT", "enabled": True},
                {"name": "confirm", "enabled": True},
                {"name": "embark", "enabled": True},
            ],
        }


class StartingRunClient(TopLevelSelectionClient):
    """Reproduces the live transition: confirm disables every control.

    Pressing confirm leaves the screen on ``character_select`` with only the
    character buttons enabled while the run loads. Pressing it again fails, so
    reset has to recognize the screen as already started and wait.
    """

    def __init__(self, polls_before_run=2):
        super().__init__()
        self.polls_before_run = polls_before_run
        self.polls = 0

    def get_state(self):
        if self.state.get("menu_screen") == "starting":
            self.polls += 1
            if self.polls >= self.polls_before_run:
                self.state = {"state_type": "map", "run": {"floor": 0}}
        return self.state

    def menu_select(self, option, seed=None):
        if option == "confirm":
            self.calls.append((option, seed))
            self.state = {
                "state_type": "menu",
                "menu_screen": "starting",
                "characters": [{"id": "IRONCLAD"}, {"id": "SILENT"}],
                "selected": {"character": self.selected},
                "options": [
                    {"name": "IRONCLAD", "enabled": True},
                    {"name": "SILENT", "enabled": True},
                ],
            }
            return {"state": self.state}
        return super().menu_select(option, seed)


def test_reset_reads_a_top_level_character_selection():
    client = TopLevelSelectionClient(selected="IRONCLAD")

    state = controller(client).reset(ResetSpec(character=0))

    assert state["state_type"] == "map"
    assert client.calls == [
        ("singleplayer", None),
        ("standard", None),
        ("confirm", None),
    ]


def test_reset_switches_character_when_the_top_level_selection_differs():
    client = TopLevelSelectionClient(selected="SILENT")

    state = controller(client).reset(ResetSpec(character=0))

    assert state["state_type"] == "map"
    assert client.calls == [
        ("singleplayer", None),
        ("standard", None),
        ("IRONCLAD", None),
        ("confirm", None),
    ]


def test_reset_waits_out_the_transition_after_pressing_start():
    client = StartingRunClient(polls_before_run=2)

    state = controller(client).reset(ResetSpec(character=0))

    assert state["state_type"] == "map"
    assert client.calls.count(("confirm", None)) == 1


class AlreadyStartingClient:
    """Reset arrives on a screen that already withdrew its start button."""

    def __init__(self, polls_before_run=2):
        self.calls = []
        self.polls = 0
        self.polls_before_run = polls_before_run
        self.state = {
            "state_type": "menu",
            "menu_screen": "character_select",
            "characters": [{"id": "IRONCLAD"}],
            "selected": {"character": "IRONCLAD"},
            "options": [{"name": "IRONCLAD", "enabled": True}],
        }

    def get_state(self):
        self.polls += 1
        if self.polls > self.polls_before_run:
            self.state = {"state_type": "map", "run": {"floor": 0}}
        return self.state

    def menu_select(self, option, seed=None):
        self.calls.append((option, seed))
        raise AssertionError(f"reset should not have pressed {option!r}")


def test_reset_does_not_press_start_twice_on_a_screen_that_withdrew_it():
    client = AlreadyStartingClient()

    state = controller(client).reset(ResetSpec(character=0))

    assert state["state_type"] == "map"
    assert client.calls == []


class TransitionalStateClient(TopLevelSelectionClient):
    """The first screen of a run reports "unknown" before it settles."""

    def __init__(self, unknown_polls=2):
        super().__init__()
        self.unknown_polls = unknown_polls
        self.seen_unknown = 0

    def get_state(self):
        if self.state.get("state_type") == "unknown":
            self.seen_unknown += 1
            if self.seen_unknown >= self.unknown_polls:
                self.state = {"state_type": "event", "run": {"floor": 0}}
        return self.state

    def menu_select(self, option, seed=None):
        if option == "confirm":
            self.calls.append((option, seed))
            self.state = {"state_type": "unknown"}
            return {"state": self.state}
        return super().menu_select(option, seed)


def test_reset_waits_for_a_transitional_state_to_settle():
    client = TransitionalStateClient()

    state = controller(client).reset(ResetSpec(character=0))

    assert state["state_type"] == "event"


def test_reset_never_returns_a_transitional_state_to_the_caller():
    """An agent cannot act in "unknown", so reuse must settle it first."""
    client = TransitionalStateClient()
    client.state = {"state_type": "unknown"}

    state = controller(client).reset(
        ResetSpec(character=0, allow_active_run=True)
    )

    assert state["state_type"] == "event"
    assert client.calls == []


class GameOverClient(TopLevelSelectionClient):
    """Game over nests its options and dismisses itself before we answer."""

    def __init__(self, dismisses_itself=False):
        super().__init__()
        self.dismisses_itself = dismisses_itself
        self.state = {
            "state_type": "game_over",
            "game_over": {"message": "Run ended.", "options": ["main_menu"]},
            "run": {"floor": 3},
        }
        self.main_menu = {
            "state_type": "menu",
            "menu_screen": "main",
            "options": ["singleplayer", "multiplayer", "settings", "quit"],
        }

    def get_state(self):
        if self.dismisses_itself and self.state.get("state_type") == "game_over":
            self.state = self.main_menu
        return self.state

    def menu_select(self, option, seed=None):
        if option == "main_menu":
            self.calls.append((option, seed))
            if self.dismisses_itself:
                raise STS2ClientError("Unknown menu option: main_menu")
            self.state = self.main_menu
            return {"state": self.state}
        return super().menu_select(option, seed)


def test_game_over_options_are_read_from_the_nested_block():
    client = GameOverClient()

    state = controller(client).reset(ResetSpec(character=0))

    assert state["state_type"] == "map"
    assert client.calls[0] == ("main_menu", None)


def test_reset_recovers_when_game_over_dismisses_itself_first():
    client = GameOverClient(dismisses_itself=True)

    state = controller(client).reset(ResetSpec(character=0))

    assert state["state_type"] == "map"


class StaleResponseClient(TopLevelSelectionClient):
    """The POST response lags a step behind the transition it triggered."""

    def __init__(self):
        super().__init__()
        self.pending = None

    def get_state(self):
        if self.pending is not None:
            self.state = self.pending
            self.pending = None
        return self.state

    def menu_select(self, option, seed=None):
        stale = self.state
        super().menu_select(option, seed)
        self.pending = self.state
        self.state = stale
        return {"state": stale}


def test_reset_waits_out_a_stale_transition_response():
    client = StaleResponseClient()

    state = controller(client).reset(ResetSpec(character=0))

    assert state["state_type"] == "map"
