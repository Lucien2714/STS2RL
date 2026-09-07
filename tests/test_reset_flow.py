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
    # The seed rides on embark, not on the submenu's custom option: the API
    # only accepts it in contexts that expose a real seeded flow.
    assert client.actions == [
        ("singleplayer", None),
        ("custom", None),
        ("embark", "ABC"),
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


class CardRewardClient:
    """A rewards screen that relists a card after it is declined."""

    def __init__(self, *, lands_on: str = "rewards", proceed_fails: bool = False):
        self.calls = []
        self.lands_on = lands_on
        self.proceed_fails = proceed_fails
        self.state = {"state_type": "card_reward",
                      "card_reward": {"cards": [{"index": 0}], "can_skip": True}}

    def get_state(self):
        return self.state

    def _rewards(self):
        return {
            "state_type": "rewards",
            "rewards": {"items": [{"index": 0, "type": "card"}], "can_proceed": True},
            "run": {"floor": 3},
        }

    def skip_card_reward(self):
        self.calls.append("skip_card_reward")
        self.state = (
            self._rewards() if self.lands_on == "rewards"
            else {"state_type": self.lands_on, "run": {"floor": 3}}
        )
        return {"state": self.state}

    def select_card_reward(self, card_index):
        self.calls.append(f"select_card_reward:{card_index}")
        self.state = self._rewards()
        return {"state": self.state}

    def proceed(self):
        self.calls.append("proceed")
        if self.proceed_fails:
            raise STS2ClientError("not now")
        self.state = {"state_type": "map", "run": {"floor": 3}}
        return {"state": self.state}


def test_declining_a_card_also_leaves_the_rewards_screen():
    """The screen relists the declined card, so resolving it includes leaving."""
    client = CardRewardClient()

    with GameEnv(client=client) as env:
        step = env.step(GameAction("skip_card_reward"))

    assert client.calls == ["skip_card_reward", "proceed"]
    assert step.raw_state["state_type"] == "map"
    assert step.info["auto_proceeded"] is True


def test_taking_a_card_also_leaves_the_rewards_screen():
    client = CardRewardClient()

    with GameEnv(client=client) as env:
        step = env.step(GameAction("select_card_reward", card_index=0))

    assert client.calls == ["select_card_reward:0", "proceed"]
    assert step.raw_state["state_type"] == "map"


def test_a_card_reward_from_an_event_is_left_alone():
    """Only a decision that lands back on the rewards screen needs the exit."""
    client = CardRewardClient(lands_on="map")

    with GameEnv(client=client) as env:
        step = env.step(GameAction("skip_card_reward"))

    assert client.calls == ["skip_card_reward"]
    assert "auto_proceeded" not in step.info


def test_a_refused_automatic_exit_leaves_a_state_we_can_still_act_on():
    client = CardRewardClient(proceed_fails=True)

    with GameEnv(client=client) as env:
        step = env.step(GameAction("skip_card_reward"))

    assert step.raw_state["state_type"] == "rewards"
    assert step.info["action_error"] is False
    assert "auto_proceeded" not in step.info


def test_other_actions_never_trigger_the_automatic_exit():
    client = FakeClient()

    with GameEnv(client=client) as env:
        step = env.step(GameAction("end_turn"))

    assert client.actions == [("end_turn", None)]
    assert "auto_proceeded" not in step.info


class BundleClient:
    """A bundle overlay that opens a preview when a bundle is picked."""

    def __init__(self, *, can_confirm: bool = True, confirm_fails: bool = False):
        self.calls = []
        self.can_confirm = can_confirm
        self.confirm_fails = confirm_fails
        self.state = self._bundles(preview=False)

    def _bundles(self, *, preview: bool):
        return {
            "state_type": "bundle_select",
            "bundle_select": {
                "bundles": [{"index": 0}, {"index": 1}],
                "preview_showing": preview,
                "can_confirm": preview and self.can_confirm,
                "can_cancel": preview,
            },
        }

    def get_state(self):
        return self.state

    def select_bundle(self, index):
        self.calls.append(f"select_bundle:{index}")
        self.state = self._bundles(preview=True)
        return {"state": self.state}

    def confirm_bundle_selection(self):
        self.calls.append("confirm_bundle_selection")
        if self.confirm_fails:
            raise STS2ClientError("not now")
        self.state = {"state_type": "map", "run": {"floor": 4}}
        return {"state": self.state}


def test_picking_a_bundle_also_confirms_it():
    """Cancelling a preview returns to the same bundles, which is a loop."""
    client = BundleClient()

    with GameEnv(client=client) as env:
        step = env.step(GameAction("select_bundle", index=1))

    assert client.calls == ["select_bundle:1", "confirm_bundle_selection"]
    assert step.raw_state["state_type"] == "map"
    assert step.info["auto_proceeded"] is True


def test_a_bundle_pick_that_opens_nothing_confirmable_is_left_alone():
    client = BundleClient(can_confirm=False)

    with GameEnv(client=client) as env:
        step = env.step(GameAction("select_bundle", index=0))

    assert client.calls == ["select_bundle:0"]
    assert "auto_proceeded" not in step.info


def test_a_refused_bundle_confirmation_leaves_an_actionable_screen():
    client = BundleClient(confirm_fails=True)

    with GameEnv(client=client) as env:
        step = env.step(GameAction("select_bundle", index=0))

    assert step.raw_state["state_type"] == "bundle_select"
    assert step.info["action_error"] is False
    assert "auto_proceeded" not in step.info


class UpgradeClient:
    """A "choose a card to upgrade" overlay: pick one, then confirm or cancel."""

    def __init__(self, *, max_select: int = 1, screen: str = "card_select"):
        self.calls = []
        self.max_select = max_select
        self.screen = screen
        self.selected = 0
        self.state = self._prompt()

    def _prompt(self):
        block = {
            "screen_type": "upgrade",
            "selected_count": self.selected,
            "min_select": self.max_select,
            "max_select": self.max_select,
            "cards": [{"index": 0}, {"index": 1}],
            "can_confirm": self.selected >= self.max_select,
            "can_cancel": True,
        }
        return {"state_type": self.screen, self.screen: block}

    def get_state(self):
        return self.state

    def select_card(self, index):
        self.calls.append(f"select_card:{index}")
        self.selected += 1
        self.state = self._prompt()
        return {"state": self.state}

    def combat_select_card(self, card_index):
        self.calls.append(f"combat_select_card:{card_index}")
        self.selected += 1
        self.state = self._prompt()
        return {"state": self.state}

    def confirm_selection(self):
        self.calls.append("confirm_selection")
        self.state = {"state_type": "rest_site", "run": {"floor": 5}}
        return {"state": self.state}

    def combat_confirm_selection(self):
        self.calls.append("combat_confirm_selection")
        self.state = {"state_type": "monster", "run": {"floor": 5}}
        return {"state": self.state}


def test_a_full_card_selection_confirms_itself():
    """Cancelling puts every card back for argmax to pick again, forever."""
    client = UpgradeClient()

    with GameEnv(client=client) as env:
        step = env.step(GameAction("select_card", index=1))

    assert client.calls == ["select_card:1", "confirm_selection"]
    assert step.raw_state["state_type"] == "rest_site"
    assert step.info["auto_proceeded"] is True


def test_a_half_filled_selection_is_left_to_the_agent():
    """Choosing 2 of N is a real sequence of decisions, not one."""
    client = UpgradeClient(max_select=2)

    with GameEnv(client=client) as env:
        step = env.step(GameAction("select_card", index=0))

    assert client.calls == ["select_card:0"]
    assert "auto_proceeded" not in step.info
    assert step.raw_state["card_select"]["selected_count"] == 1


def test_hand_selection_confirms_itself_when_full():
    client = UpgradeClient(screen="hand_select")

    with GameEnv(client=client) as env:
        step = env.step(GameAction("combat_select_card", card_index=0))

    assert client.calls == ["combat_select_card:0", "combat_confirm_selection"]
    assert step.raw_state["state_type"] == "monster"


def test_a_choose_screen_that_never_reports_a_selection_is_left_alone():
    """"choose" picks immediately and holds selected_count at zero."""
    client = UpgradeClient()
    client.selected = -1  # so selected_count stays below max after the pick

    with GameEnv(client=client) as env:
        step = env.step(GameAction("select_card", index=0))

    assert client.calls == ["select_card:0"]
    assert "auto_proceeded" not in step.info
