"""Run-progress reward behaviour."""

from __future__ import annotations

import pytest

from sts2rl.env.constants import (
    BOSS_VICTORY_REWARD,
    NODE_PROGRESS_REWARD,
    STEP_COST,
)
from sts2rl.env.rewards import RunProgressReward


def _screen(state_type: str, *, floor: int = 5, hp: int = 70) -> dict:
    return {
        "state_type": state_type,
        "run": {"act": 1, "floor": floor},
        "player": {"hp": hp, "max_hp": 80, "gold": 100},
    }


def _battle(state_type: str = "monster", *, floor: int = 5, hp: int = 70,
            enemy_hp: int | None = 40) -> dict:
    state = _screen(state_type, floor=floor, hp=hp)
    enemies = [] if enemy_hp is None else [{"entity_id": "JAW_WORM_0", "hp": enemy_hp}]
    state["battle"] = {"turn": "player", "is_play_phase": True, "enemies": enemies}
    return state


def test_entering_a_node_is_worth_one_point():
    model = RunProgressReward()
    model.reset(_screen("map"))

    reward, details = model.compute(_screen("map", floor=5), _screen("map", floor=6))

    assert details["nodes"] == 1
    assert reward == pytest.approx(NODE_PROGRESS_REWARD - STEP_COST)


def test_several_nodes_at_once_are_each_counted():
    model = RunProgressReward()
    model.reset(_screen("map"))

    reward, details = model.compute(_screen("map", floor=5), _screen("map", floor=8))

    assert details["nodes"] == 3
    assert reward == pytest.approx(3 * NODE_PROGRESS_REWARD - STEP_COST)


def test_a_floor_that_never_advances_only_costs_the_step():
    model = RunProgressReward()
    model.reset(_screen("map"))

    reward, details = model.compute(_screen("map"), _screen("map"))

    assert "nodes" not in details
    assert reward == pytest.approx(-STEP_COST)


def test_beating_a_boss_pays_the_bonus_exactly_once():
    model = RunProgressReward()
    model.reset(_battle("boss"))

    fighting, _ = model.compute(_battle("boss"), _battle("boss", enemy_hp=12))
    winning, details = model.compute(_battle("boss", enemy_hp=3), _screen("rewards"))
    after, again = model.compute(_screen("rewards"), _screen("rewards"))

    assert "boss_defeated" not in _
    assert details["boss_defeated"] is True
    assert winning == pytest.approx(BOSS_VICTORY_REWARD - STEP_COST)
    assert "boss_defeated" not in again
    assert after == pytest.approx(-STEP_COST)
    assert fighting == pytest.approx(-STEP_COST)


def test_clearing_an_ordinary_fight_pays_no_boss_bonus():
    model = RunProgressReward()
    model.reset(_battle())

    reward, details = model.compute(_battle(enemy_hp=5), _screen("rewards"))

    assert "boss_defeated" not in details
    assert reward == pytest.approx(-STEP_COST)


def test_dying_to_a_boss_pays_nothing_and_clears_the_pending_fight():
    model = RunProgressReward()
    model.reset(_battle("boss"))

    reward, details = model.compute(
        _battle("boss", enemy_hp=30, hp=4), _screen("game_over", hp=0)
    )

    assert "boss_defeated" not in details
    assert reward == pytest.approx(-STEP_COST)
    assert model._boss_pending is False


def test_healing_pays_nothing():
    """Scoring HP taught resting at every rest site instead of upgrading.

    Healing paid immediately and upgrading paid nothing, so the agent took the
    heal every time.  HP still matters -- running out ends the run -- but it
    matters terminally, and pricing it per step is what made it farmable.
    """
    model = RunProgressReward()
    model.reset(_screen("rest_site", hp=40))

    reward, details = model.compute(
        _screen("rest_site", hp=40), _screen("rest_site", hp=70)
    )

    assert "hp_change" not in details
    assert reward == pytest.approx(-STEP_COST)


def test_taking_damage_costs_nothing_directly():
    model = RunProgressReward()
    model.reset(_battle(hp=70))

    reward, details = model.compute(_battle(hp=70), _battle(hp=45))

    assert "hp_change" not in details
    assert reward == pytest.approx(-STEP_COST)


def test_a_node_cleared_at_one_hp_scores_the_same_as_one_cleared_untouched():
    """Deliberate: the difference shows up as a shorter run, not a smaller step."""
    model = RunProgressReward()
    model.reset(_screen("map", hp=80))
    bloodied, _ = model.compute(
        _screen("map", floor=5, hp=80), _screen("map", floor=6, hp=1)
    )

    model.reset(_screen("map", hp=80))
    untouched, _ = model.compute(
        _screen("map", floor=5, hp=80), _screen("map", floor=6, hp=80)
    )

    assert bloodied == pytest.approx(untouched)


def test_loitering_is_never_free():
    model = RunProgressReward()
    model.reset(_screen("card_select"))

    total = sum(
        model.compute(_screen("card_select"), _screen("card_select"))[0]
        for _ in range(50)
    )

    assert total == pytest.approx(-50 * STEP_COST)


def test_rejected_actions_are_reported_without_reward():
    reward, details = RunProgressReward().action_error_reward("bad move")

    assert reward == 0.0
    assert details["action_error"] is True
    assert details["error"] == "bad move"
