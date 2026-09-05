import pytest

from sts2rl.actions.dispatcher import ActionDispatcher
from sts2rl.actions.game_action import GameAction, MenuSelectAction


class FakeClient:
    def __init__(self):
        self.calls = []

    def play_card(self, card_index, target=None):
        self.calls.append(("play_card", card_index, target))
        return {"state_type": "monster"}

    def select_card(self, index):
        self.calls.append(("select_card", index))
        return {"state_type": "card_select"}

    def menu_select(self, option, seed=None):
        self.calls.append(("menu_select", option, seed))
        return {"state_type": "menu"}


def test_game_action_round_trips_legacy_dictionary():
    action = GameAction.from_dict(
        {"type": "play_card", "card_index": 2, "target": "ENEMY_0"}
    )

    assert action.action_type == "play_card"
    assert action.params == {"card_index": 2, "target": "ENEMY_0"}
    assert action.to_dict() == {
        "type": "play_card",
        "card_index": 2,
        "target": "ENEMY_0",
    }


def test_dispatcher_routes_typed_game_action():
    client = FakeClient()
    dispatcher = ActionDispatcher(client)

    result = dispatcher.dispatch(
        GameAction("play_card", card_index=2, target="ENEMY_0")
    )

    assert result == {"state_type": "monster"}
    assert client.calls == [("play_card", 2, "ENEMY_0")]


def test_select_card_action_uses_api_index_parameter():
    client = FakeClient()
    dispatcher = ActionDispatcher(client)

    dispatcher.dispatch(GameAction("select_card", index=3))

    assert client.calls == [("select_card", 3)]


def test_menu_select_action_routes_option_and_seed():
    client = FakeClient()
    dispatcher = ActionDispatcher(client)

    dispatcher.dispatch(MenuSelectAction("custom", seed="ABC"))

    assert client.calls == [("menu_select", "custom", "ABC")]


def test_dispatcher_rejects_unknown_action_type():
    with pytest.raises(ValueError, match="Unknown action type"):
        ActionDispatcher(FakeClient()).dispatch(GameAction("not_an_action"))


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (GameAction("discard_potion", slot=1), ("discard_potion", (1,))),
        (GameAction("select_bundle", index=2), ("select_bundle", (2,))),
        (
            GameAction("confirm_bundle_selection"),
            ("confirm_bundle_selection", ()),
        ),
        (GameAction("cancel_bundle_selection"), ("cancel_bundle_selection", ())),
        (GameAction("select_relic", index=3), ("select_relic", (3,))),
        (GameAction("skip_relic_selection"), ("skip_relic_selection", ())),
        (
            GameAction("crystal_sphere_set_tool", tool="small"),
            ("crystal_sphere_set_tool", ("small",)),
        ),
        (
            GameAction("crystal_sphere_click_cell", x=4, y=7),
            ("crystal_sphere_click_cell", (4, 7)),
        ),
        (GameAction("crystal_sphere_proceed"), ("crystal_sphere_proceed", ())),
    ],
)
def test_dispatcher_routes_remaining_full_run_actions(action, expected):
    class RecordingClient:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            def record(*args):
                self.calls.append((name, args))
                return {"state_type": "map"}

            return record

    client = RecordingClient()

    ActionDispatcher(client).dispatch(action)

    assert client.calls == [expected]
