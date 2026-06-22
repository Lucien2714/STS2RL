"""Tests for environment reset menu navigation."""

from sts2rl.env.game_env import GameEnv
from sts2rl.env.player import Player
from sts2rl.flow.player_detail import refresh_player_detail_for_map


class FakeClient:
    """Minimal STS2MCP client double for reset-flow tests."""

    def __init__(self):
        self.actions = []
        self.state = {
            "state_type": "menu",
            "menu_screen": "main",
            "options": ["singleplayer"],
        }

    def get_state(self):
        """Return the fake client's current state."""
        return self.state

    def menu_select(self, option, seed=None):
        """Record menu selections and advance through a tiny fake menu."""
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
        """Advance to a fake next state for step-boundary tests."""
        self.actions.append(("end_turn", None))
        self.state = {
            "state_type": "map",
            "run": {"act": 1, "floor": 1, "ascension": 0},
            "player": {"hp": 80, "gold": 0, "max_hp": 80},
        }
        return {"state": self.state}

    def get_player_detail(self):
        """Return a fake player-detail response."""
        return {
            "state_type": "player_detail",
            "status": "ok",
            "game_mode": "singleplayer",
            "run": {"act": 1, "floor": 1, "ascension": 0},
            "player": {
                "character": "The Ironclad",
                "hp": 70,
                "max_hp": 80,
                "block": 0,
                "gold": 42,
                "status": [],
                "relics": [
                    {"id": "BURNING_BLOOD", "name": "Burning Blood"},
                ],
                "potions": [
                    {"id": "FIRE_POTION", "name": "Fire Potion", "slot": 0},
                ],
                "max_potion_slots": 3,
                "deck_count": 1,
                "deck": [
                    {
                        "index": 0,
                        "id": "BASH",
                        "name": "Bash",
                        "cost": "2",
                        "star_cost": None,
                        "is_upgraded": True,
                    },
                ],
            },
        }


def test_custom_seed_reset_embarks_before_returning_state():
    """Custom seeded resets should select embark before returning map state."""
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
    """Steps should only communicate with STS2MCP and not compute reward."""
    env = GameEnv()
    env.client = FakeClient()
    env.action_dispatcher.client = env.client

    next_state, done, info = env.step({"type": "end_turn"})

    assert next_state["run"]["floor"] == 1
    assert done is False
    assert info["raw_state"] == next_state
    assert info["action_error"] is False
    assert "reward_details" not in info


def test_game_env_tracks_in_battle_across_card_select_overlay():
    """Card-select overlays should inherit battle context from previous states."""
    env = GameEnv()
    battle_state = env._state_from_action_result(
        {
            "state": {
                "state_type": "monster",
                "battle": {"turn": "player", "is_play_phase": True, "enemies": []},
                "player": {},
            }
        }
    )
    card_select_state = env._state_from_action_result(
        {
            "state": {
                "state_type": "card_select",
                "card_select": {
                    "screen_type": "simple_select",
                    "cards": [{"index": 0, "id": "STRIKE_IRONCLAD"}],
                },
                "player": {},
            }
        }
    )
    map_state = env._state_from_action_result(
        {
            "state": {
                "state_type": "map",
                "run": {"act": 1, "floor": 5},
                "player": {},
            }
        }
    )

    assert battle_state["in_battle"] is True
    assert card_select_state["in_battle"] is True
    assert map_state["in_battle"] is False


def test_refresh_player_detail_for_map_updates_player_model():
    """Map states should be enriched with player-detail before routing."""
    env = GameEnv()
    env.client = FakeClient()
    player = Player(character=0)
    raw_state = {
        "state_type": "map",
        "run": {"act": 1, "floor": 1, "ascension": 0},
        "map": {"next_options": []},
    }

    detail = refresh_player_detail_for_map(env, player, raw_state)

    assert detail is raw_state["player_detail"]
    assert player.current_hp == 70
    assert player.gold == 42
    assert player.deck_count == 1
    assert player.current_deck[0].card_id == "BASH"
    assert player.current_deck[0].upgraded is True
    assert len(player.relic_details) == 1
    assert len(player.potion_details) == 1


def test_player_detail_update_ignores_non_scalar_int_values():
    """Player detail parsing should tolerate non-scalar numeric fields."""
    player = Player(character=0)
    player.update_from_detail(
        {
            "state_type": "player_detail",
            "status": "ok",
            "player": {
                "character": "The Ironclad",
                "hp": {"unexpected": "object"},
                "max_hp": object(),
                "block": [],
                "gold": "44",
                "max_potion_slots": None,
                "status": [],
                "relics": [],
                "potions": [],
                "deck_count": {"bad": "count"},
                "deck": [
                    {
                        "id": "WHIRLWIND",
                        "name": "Whirlwind",
                        "cost": "X",
                        "star_cost": None,
                        "is_upgraded": False,
                    },
                ],
            },
        }
    )

    assert player.current_hp == 80
    assert player.max_hp == 80
    assert player.block == 0
    assert player.gold == 44
    assert player.deck_count == 1
    assert player.current_deck[0].regular_cost == -1
    assert player.current_deck[0].star_cost == 0
