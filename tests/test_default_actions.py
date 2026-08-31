"""`proceed` must never be sent to a screen that does not accept it."""

import pytest

from sts2rl.action_spaces.base import LEGAL_ACTION_TYPES, is_legal_action_type
from sts2rl.action_spaces.defaults import REFRESH_STATE_ACTION, default_action
from sts2rl.agents.default.rule_based import DefaultPolicy
from sts2rl.agents.map.rule_based import MapPolicy
from sts2rl.agents.orchestrator import Agent
from sts2rl.flow.step_loop import decide_action

# Screens the API table says reject `proceed`.
NO_PROCEED_STATE_TYPES = [
    "monster",
    "elite",
    "boss",
    "hand_select",
    "card_select",
    "map",
    "card_reward",
    "event",
]


@pytest.mark.parametrize("state_type", NO_PROCEED_STATE_TYPES)
def test_proceed_is_not_legal_on_screens_that_reject_it(state_type):
    """The legality table must agree with the STS2MCP action reference."""
    assert not is_legal_action_type(state_type, "proceed")


@pytest.mark.parametrize("state_type", ["rewards", "treasure", "shop", "fake_merchant", "rest_site"])
def test_proceed_is_legal_where_the_api_accepts_it(state_type):
    """`proceed` stays available on the screens that do take it."""
    assert is_legal_action_type(state_type, "proceed")


@pytest.mark.parametrize("state_type", NO_PROCEED_STATE_TYPES)
def test_default_action_never_returns_illegal_proceed(state_type):
    """An empty screen must not fall back to a `proceed` the API would reject."""
    raw_state = {"state_type": state_type, "in_battle": state_type != "map"}

    action = default_action(raw_state)

    assert action["type"] != "proceed"
    assert is_legal_action_type(state_type, action["type"])


def test_default_action_prefers_a_real_candidate():
    """When the screen has a legal move, the default is that move."""
    raw_state = {
        "state_type": "map",
        "map": {"next_options": [{"index": 2, "type": "monster", "col": 1, "row": 3}]},
    }

    assert default_action(raw_state) == {
        "type": "choose_map_node",
        "index": 2,
        "action_key": "choose_map_node:2",
    }


def test_default_action_proceeds_where_the_screen_allows_it():
    """A shop with nothing affordable still proceeds, because shops accept it."""
    raw_state = {"state_type": "shop", "shop": {"items": [], "can_proceed": True}}

    assert default_action(raw_state)["type"] == "proceed"


def test_enemy_turn_resolves_to_a_state_refresh():
    """Combat outside the play phase has no legal move, so we re-read state."""
    raw_state = {
        "state_type": "monster",
        "in_battle": True,
        "battle": {"turn": "enemy", "is_play_phase": False, "enemies": []},
        "player": {"hand": [], "potions": []},
    }

    assert default_action(raw_state) == REFRESH_STATE_ACTION


def test_step_loop_skips_agent_without_sending_proceed():
    """The skipped-agent branch used to hardcode a `proceed` combat rejects."""
    raw_state = {
        "state_type": "monster",
        "in_battle": True,
        "battle": {"turn": "enemy", "is_play_phase": False, "enemies": []},
        "player": {"hand": [], "potions": [], "status": [], "relics": []},
    }

    decision = decide_action(Agent(), raw_state, training=True)

    assert decision.skipped_agent is True
    assert decision.agent_chose is False
    assert decision.action["type"] == REFRESH_STATE_ACTION["type"]


def test_map_policy_without_options_does_not_proceed():
    """The map screen only accepts choose_map_node."""
    raw_state = {"state_type": "map", "map": {"next_options": []}}

    action = MapPolicy().choose_action(raw_state)

    assert action["type"] != "proceed"
    assert is_legal_action_type("map", action["type"])


@pytest.mark.parametrize("state_type", sorted(LEGAL_ACTION_TYPES))
def test_default_policy_is_legal_on_every_known_screen(state_type):
    """The catch-all policy must be legal wherever it is used."""
    action = DefaultPolicy().choose_action({"state_type": state_type})

    assert is_legal_action_type(state_type, action["type"])
