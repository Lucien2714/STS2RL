"""Run-progress reward behaviour."""

from __future__ import annotations

import pytest

from sts2rl.env.constants import (
    BOSS_VICTORY_REWARD,
    NODE_PROGRESS_REWARD,
    STEP_COST,
)
from sts2rl.env.rewards import RunProgressReward


def _screen(state_type: str, *, floor: int = 5, hp: int = 70, act: int = 1) -> dict:
    return {
        "state_type": state_type,
        "run": {"act": act, "floor": floor},
        "player": {"hp": hp, "max_hp": 80, "gold": 100},
    }


def _game_over(*, floor: int = 51, victory: bool = False,
               outcome: str = "victory", hp: int = 30) -> dict:
    state = _screen("game_over", floor=floor, hp=hp)
    state["game_over"] = {
        "message": "Run ended in victory." if victory else "Run ended.",
        "victory": victory,
        "outcome": outcome if victory else "combat_death",
        "options": ["main_menu"],
    }
    return state


def _battle(state_type: str = "monster", *, floor: int = 5, hp: int = 70,
            enemy_hp: int | None = 40, act: int = 1) -> dict:
    state = _screen(state_type, floor=floor, hp=hp, act=act)
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


def test_leaving_the_boss_floor_pays_the_bonus_once():
    """Beating a boss is the only way onto a later floor."""
    model = RunProgressReward()

    fighting, first = model.compute(
        _battle("boss", floor=17), _battle("boss", floor=17, enemy_hp=12)
    )
    claiming, second = model.compute(
        _battle("boss", floor=17, enemy_hp=3), _screen("rewards", floor=17)
    )
    winning, details = model.compute(
        _screen("map", floor=17), _battle("monster", floor=18)
    )
    after, again = model.compute(
        _battle("monster", floor=18), _screen("rewards", floor=18)
    )

    assert "bosses_defeated" not in first
    assert "bosses_defeated" not in second
    assert fighting == pytest.approx(-STEP_COST)
    assert claiming == pytest.approx(-STEP_COST)
    assert details["bosses_defeated"] == 1
    assert winning == pytest.approx(
        BOSS_VICTORY_REWARD + NODE_PROGRESS_REWARD - STEP_COST
    )
    assert "bosses_defeated" not in again
    assert after == pytest.approx(-STEP_COST)


def test_two_bosses_in_one_act_are_each_paid():
    """The third act carries two at the highest difficulty.

    This is why the bonus counts floors rather than the act counter, which
    would advance once and pay for one of them.
    """
    model = RunProgressReward()

    first, first_details = model.compute(
        _battle("boss", floor=50), _battle("boss", floor=51)
    )
    second, second_details = model.compute(
        _battle("boss", floor=51), _screen("rewards", floor=52)
    )

    assert first_details["bosses_defeated"] == 1
    assert second_details["bosses_defeated"] == 1
    assert first == pytest.approx(
        BOSS_VICTORY_REWARD + NODE_PROGRESS_REWARD - STEP_COST
    )
    assert second == pytest.approx(
        BOSS_VICTORY_REWARD + NODE_PROGRESS_REWARD - STEP_COST
    )


def test_a_selection_prompt_during_a_boss_fight_pays_nothing():
    """``card_select`` reports no enemies, which used to read as a victory.

    Reopening it re-armed the old flag, so the bonus was farmable at ten
    points every other step against a node worth one.
    """
    model = RunProgressReward()

    total = 0.0
    for _ in range(3):
        reward, details = model.compute(
            _battle("boss", floor=17), _screen("card_select", floor=17)
        )
        total += reward
        assert "bosses_defeated" not in details
        reward, details = model.compute(
            _screen("card_select", floor=17), _battle("boss", floor=17)
        )
        total += reward
        assert "bosses_defeated" not in details

    assert total == pytest.approx(-6 * STEP_COST)


def test_winning_the_run_pays_the_last_boss():
    """The run ends on the final boss's own floor, so no later floor exists.

    ``game_over.victory`` is the only thing that reports it.  The bonus is one
    boss, not a new magnitude: winning is beating the last one.
    """
    model = RunProgressReward()

    reward, details = model.compute(
        _battle("boss", floor=51, enemy_hp=3),
        _game_over(floor=51, victory=True),
    )

    assert details["bosses_defeated"] == 1
    assert reward == pytest.approx(BOSS_VICTORY_REWARD - STEP_COST)


def test_winning_pays_once_and_forgets_the_boss_room():
    model = RunProgressReward()

    model.compute(_battle("boss", floor=51), _game_over(floor=51, victory=True))
    reward, details = model.compute(
        _game_over(floor=51, victory=True), _game_over(floor=51, victory=True)
    )

    assert "bosses_defeated" not in details
    assert reward == pytest.approx(-STEP_COST)


def test_a_run_that_ended_without_victory_pays_no_boss():
    model = RunProgressReward()

    reward, details = model.compute(
        _battle("boss", floor=51, enemy_hp=30, hp=4),
        _game_over(floor=51, victory=False, outcome="combat_death"),
    )

    assert "bosses_defeated" not in details
    assert reward == pytest.approx(-STEP_COST)


def test_a_game_over_that_reports_no_victory_field_is_not_a_win():
    """Game JSON, so a missing flag reads as "not a win" rather than truthy."""
    model = RunProgressReward()

    reward, details = model.compute(
        _battle("boss", floor=51), _screen("game_over", floor=51, hp=0)
    )

    assert "bosses_defeated" not in details
    assert reward == pytest.approx(-STEP_COST)


def test_dying_on_the_boss_floor_pays_nothing():
    """Three episodes in one measured run were paid and then died at floor 17,
    which beating the boss makes impossible."""
    model = RunProgressReward()

    reward, details = model.compute(
        _battle("boss", floor=17, enemy_hp=30, hp=4),
        _screen("game_over", floor=17, hp=0),
    )

    assert "bosses_defeated" not in details
    assert reward == pytest.approx(-STEP_COST)


def test_clearing_an_ordinary_fight_pays_no_boss_bonus():
    model = RunProgressReward()

    reward, details = model.compute(_battle(enemy_hp=5), _screen("rewards"))

    assert "bosses_defeated" not in details
    assert reward == pytest.approx(-STEP_COST)


def test_a_payload_with_no_run_block_neither_pays_nor_forgets_the_boss():
    """Floor reads as 0 there, which is not a later floor -- and the room has
    to survive the gap, or the bonus would be dropped instead of deferred."""
    model = RunProgressReward()

    lost, first = model.compute(_battle("boss", floor=17), {"state_type": "unknown"})
    back, second = model.compute({"state_type": "unknown"}, _screen("map", floor=17))
    winning, third = model.compute(
        _screen("map", floor=17), _battle("monster", floor=18)
    )

    assert "bosses_defeated" not in first
    assert "bosses_defeated" not in second
    assert lost == pytest.approx(-STEP_COST)
    assert back == pytest.approx(-STEP_COST)
    assert third["bosses_defeated"] == 1
    assert winning == pytest.approx(
        BOSS_VICTORY_REWARD + NODE_PROGRESS_REWARD - STEP_COST
    )


def test_a_new_run_forgets_the_boss_room_the_last_one_died_in():
    model = RunProgressReward()
    model.compute(_battle("boss", floor=17), _screen("game_over", floor=17, hp=0))

    model.reset(_screen("event", floor=1))
    reward, details = model.compute(_screen("map", floor=1), _screen("map", floor=2))

    assert "bosses_defeated" not in details
    assert reward == pytest.approx(NODE_PROGRESS_REWARD - STEP_COST)


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
