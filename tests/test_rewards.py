"""Battle and run reward-model behaviour."""

from __future__ import annotations

import pytest

from sts2rl.env.constants import (
    BATTLE_HP_LOSS_PENALTY,
    BATTLE_LOSS_PENALTY,
    BATTLE_WIN_REWARD,
    ENEMY_DAMAGE_REWARD,
    ENEMY_KILL_REWARD,
    FLOOR_PROGRESS_REWARD,
    GAME_OVER_PENALTY,
    POTION_USE_PENALTY,
    RUN_HP_CHANGE_REWARD,
    UNSPENT_ENERGY_PENALTY,
)
from sts2rl.env.rewards import BattleProgressReward


def _battle(
    *,
    hp: int = 70,
    enemy_hp: int | None = 40,
    energy: int = 3,
    gold: int = 100,
    max_hp: int = 80,
) -> dict:
    enemies = [] if enemy_hp is None else [{"entity_id": "JAW_WORM_0", "hp": enemy_hp}]
    return {
        "state_type": "monster",
        "run": {"floor": 5},
        "player": {"hp": hp, "max_hp": max_hp, "gold": gold, "energy": energy},
        "battle": {"turn": "player", "is_play_phase": True, "enemies": enemies},
    }


def _pair(first: int = 40, second: int = 30) -> dict:
    state = _battle()
    state["battle"]["enemies"] = [
        {"entity_id": "JAW_WORM_0", "hp": first},
        {"entity_id": "JAW_WORM_1", "hp": second},
    ]
    return state


def _screen(state_type: str, *, hp: int = 70, floor: int = 5) -> dict:
    return {
        "state_type": state_type,
        "run": {"floor": floor},
        "player": {"hp": hp, "max_hp": 80, "gold": 100, "energy": 0},
    }


def test_damaging_an_enemy_pays_proportionally_to_hp_removed():
    model = BattleProgressReward()
    model.reset(_battle())

    reward, details = model.compute(_battle(), _battle(enemy_hp=25))

    assert details["enemy_hp_lost"] == 15
    assert details["enemies_killed"] == 0
    assert reward == pytest.approx(15 * ENEMY_DAMAGE_REWARD)


def test_killing_one_of_several_enemies_adds_a_kill_bonus_but_not_a_win():
    model = BattleProgressReward()
    model.reset(_pair())

    reward, details = model.compute(_pair(6, 30), _pair(0, 30))

    assert details["enemies_killed"] == 1
    assert details["result"] is None
    assert reward == pytest.approx(6 * ENEMY_DAMAGE_REWARD + ENEMY_KILL_REWARD)


def test_killing_the_last_enemy_wins_the_battle_outright():
    model = BattleProgressReward()
    model.reset(_battle())

    reward, details = model.compute(_battle(enemy_hp=6), _battle(enemy_hp=0))

    assert details["result"] == "won"
    assert reward == pytest.approx(
        BATTLE_WIN_REWARD + 6 * ENEMY_DAMAGE_REWARD + ENEMY_KILL_REWARD
    )


def test_losing_player_hp_is_penalized_once_per_step():
    model = BattleProgressReward()
    model.reset(_battle())

    reward, details = model.compute(_battle(hp=70), _battle(hp=58))

    assert details["step_hp_lost"] == 12
    assert reward == pytest.approx(-12 * BATTLE_HP_LOSS_PENALTY)


def test_clearing_the_last_enemy_into_a_reward_screen_counts_as_a_win():
    model = BattleProgressReward()
    model.reset(_battle())

    reward, details = model.compute(_battle(), _screen("rewards"))

    assert details["result"] == "won"
    assert reward > BATTLE_WIN_REWARD / 2


def test_a_resolved_battle_is_not_scored_twice():
    model = BattleProgressReward()
    model.reset(_battle())
    model.compute(_battle(), _screen("rewards"))

    reward, details = model.compute(
        _battle(enemy_hp=None),
        _screen("rewards"),
    )

    assert details["already_resolved"] is True
    assert reward == 0.0


def test_dying_in_battle_is_penalized_as_a_loss():
    model = BattleProgressReward()
    model.reset(_battle())

    reward, details = model.compute(_battle(hp=4), _screen("game_over", hp=0))

    assert details["result"] == "lost"
    assert reward == pytest.approx(
        -BATTLE_LOSS_PENALTY - 4 * BATTLE_HP_LOSS_PENALTY
    )


def test_ending_a_turn_with_unspent_energy_is_penalized():
    model = BattleProgressReward()
    model.reset(_battle())

    reward, _ = model.compute(
        _battle(energy=2),
        _battle(),
        {"type": "end_turn"},
    )

    assert reward == pytest.approx(-2 * UNSPENT_ENERGY_PENALTY)


def test_using_a_potion_carries_a_fixed_cost():
    model = BattleProgressReward()
    model.reset(_battle())

    reward, details = model.compute(
        _battle(),
        _battle(),
        {"type": "use_potion", "slot": 0},
    )

    assert details["potion_used"] is True
    assert reward == pytest.approx(-POTION_USE_PENALTY)


def test_climbing_a_floor_outside_battle_pays_progress():
    model = BattleProgressReward()
    model.reset(_screen("map"))

    reward, details = model.compute(_screen("map", floor=5), _screen("map", floor=6))

    assert details["type"] == "default"
    assert reward == pytest.approx(FLOOR_PROGRESS_REWARD)


def test_healing_and_dying_outside_battle_are_both_scored():
    model = BattleProgressReward()
    model.reset(_screen("rest_site"))

    healed, _ = model.compute(
        _screen("rest_site", hp=50),
        _screen("rest_site", hp=62),
    )
    died, _ = model.compute(
        _screen("event", hp=3),
        _screen("game_over", hp=0),
    )

    assert healed == pytest.approx(12 * RUN_HP_CHANGE_REWARD)
    assert died == pytest.approx(-GAME_OVER_PENALTY - 3 * RUN_HP_CHANGE_REWARD)


def test_every_reward_magnitude_stays_within_one_order_of_the_others():
    """Guard the shared policy/value trunk against an outsized outcome term."""
    per_step = [
        ENEMY_KILL_REWARD,
        POTION_USE_PENALTY,
        UNSPENT_ENERGY_PENALTY,
        FLOOR_PROGRESS_REWARD,
        GAME_OVER_PENALTY,
    ]
    outcomes = [BATTLE_WIN_REWARD, BATTLE_LOSS_PENALTY]

    assert max(outcomes) <= 20 * max(per_step)


def test_action_errors_are_reported_without_reward():
    reward, details = BattleProgressReward().action_error_reward("bad move")

    assert reward == 0.0
    assert details["action_error"] is True
    assert details["error"] == "bad move"
