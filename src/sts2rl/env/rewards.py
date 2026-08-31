"""Reward-model interfaces and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from sts2rl.env.constants import (
    BATTLE_ENEMY_DAMAGE_REWARD,
    BATTLE_ENEMY_KILL_REWARD,
    BATTLE_GOLD_LOSS_PENALTY,
    BATTLE_HP_LOSS_PENALTY,
    BATTLE_LOSS_PENALTY,
    BATTLE_MAX_HP_LOSS_PENALTY,
    BATTLE_POTION_USE_PENALTY,
    BATTLE_REWARD_STATE_TYPES,
    BATTLE_STATE_TYPES,
    BATTLE_UNUSED_ENERGY_PENALTY,
    BATTLE_WIN_REWARD,
    RUN_ACT_REWARD,
    RUN_FLOOR_REWARD,
    RUN_GAME_OVER_PENALTY,
)
from sts2rl.env.state import (
    battle_has_alive_enemy,
    enemy_hp_map,
    parse_int,
    player_gold,
    player_hp,
    player_max_hp,
)


class RewardModel(ABC):
    """Interface for environment reward functions."""

    @abstractmethod
    def compute(
        self,
        prev_state: dict,
        next_state: dict,
        action: dict | None = None,
    ) -> tuple[float, dict]:
        """Return a scalar reward and structured reward details."""

    def reset(self, raw_state: dict | None = None) -> None:
        """Reset stateful reward context at episode boundaries."""

    def action_error_reward(self, error: object) -> tuple[float, dict]:
        """Return reward details for a failed action dispatch."""
        reward = 0.0
        return reward, {
            "type": "action_error",
            "error": str(error),
            "action_error": True,
            "battle_reward": 0.0,
            "run_reward": 0.0,
            "total": reward,
            "battle_details": {"type": "battle", "active": False, "total": 0.0},
            "run_details": {"type": "run", "total": 0.0},
        }


class BattleOutcomeReward(RewardModel):
    """Battle-scoped reward for winning fights while preserving resources."""

    def __init__(self) -> None:
        self.reset()

    def reset(self, raw_state: dict | None = None) -> None:
        """Clear battle bookkeeping and seed HP tracking from an optional state."""
        self._last_player_hp = player_hp(raw_state, None) if raw_state is not None else None
        self._battle_start_hp = None
        self._battle_start_gold = None
        self._battle_start_max_hp = None
        self._battle_reward_closed = False

    def compute(
        self,
        prev_state: dict,
        next_state: dict,
        action: dict | None = None,
    ) -> tuple[float, dict]:
        """Compute a battle-only reward for one raw-state transition."""
        if not self._is_battle_reward_state(prev_state):
            reward = 0.0
            details = {
                "type": "battle",
                "active": False,
                "prev_state_type": prev_state.get("state_type"),
                "next_state_type": next_state.get("state_type"),
                "result": None,
                "total": reward,
            }
            self._last_player_hp = player_hp(next_state, self._last_player_hp)
            return reward, details

        reward, details = self._compute_battle_reward(prev_state, next_state, action)
        self._last_player_hp = player_hp(next_state, self._last_player_hp)
        return reward, details

    def _is_battle_reward_state(self, state: dict) -> bool:
        """Return whether a transition should use battle reward shaping."""
        return state.get("state_type") in BATTLE_STATE_TYPES or state.get("in_battle") is True

    def _compute_battle_reward(
        self,
        prev_state: dict,
        next_state: dict,
        action: dict | None = None,
    ) -> tuple[float, dict]:
        """Reward battle outcomes and small resource deltas."""
        prev_hp = player_hp(prev_state, self._last_player_hp)
        next_hp = player_hp(next_state, prev_hp)
        prev_gold = player_gold(prev_state)
        next_gold = player_gold(next_state, prev_gold)
        prev_max_hp = player_max_hp(prev_state)
        next_max_hp = player_max_hp(next_state, prev_max_hp)
        prev_has_alive_enemy = battle_has_alive_enemy(prev_state)

        if prev_has_alive_enemy:
            self._battle_reward_closed = False

        if self._battle_start_hp is None and prev_has_alive_enemy:
            self._battle_start_hp = prev_hp
        if self._battle_start_gold is None and prev_has_alive_enemy:
            self._battle_start_gold = prev_gold
        if self._battle_start_max_hp is None and prev_has_alive_enemy:
            self._battle_start_max_hp = prev_max_hp

        if self._battle_reward_closed and not prev_has_alive_enemy:
            return 0.0, {
                "type": "battle",
                "active": True,
                "prev_state_type": prev_state.get("state_type"),
                "next_state_type": next_state.get("state_type"),
                "result": None,
                "already_resolved": True,
                "prev_hp": prev_hp,
                "next_hp": next_hp,
                "prev_gold": prev_gold,
                "next_gold": next_gold,
                "prev_max_hp": prev_max_hp,
                "next_max_hp": next_max_hp,
                "total": 0.0,
            }

        step_hp_lost = max(0, prev_hp - next_hp)
        battle_start_hp = self._battle_start_hp if self._battle_start_hp is not None else prev_hp
        battle_start_gold = (
            self._battle_start_gold if self._battle_start_gold is not None else prev_gold
        )
        battle_start_max_hp = (
            self._battle_start_max_hp if self._battle_start_max_hp is not None else prev_max_hp
        )
        total_hp_lost = max(0, battle_start_hp - next_hp)
        total_gold_lost = max(0, battle_start_gold - next_gold)
        total_max_hp_lost = max(0, battle_start_max_hp - next_max_hp)

        potion_used = bool(action and action.get("type") == "use_potion")
        potion_discarded = bool(action and action.get("type") == "discard_potion")
        potion_penalty = -BATTLE_POTION_USE_PENALTY if (potion_used or potion_discarded) else 0.0
        prev_state_type = prev_state.get("state_type")
        next_state_type = next_state.get("state_type")
        battle_result = self._battle_result(prev_state, next_state)
        enemy_hp_lost, enemies_killed = self._enemy_hp_progress(
            prev_state,
            next_state,
            count_missing_as_dead=battle_result != "lost",
        )

        enemy_damage_reward = float(enemy_hp_lost) * BATTLE_ENEMY_DAMAGE_REWARD
        enemy_kill_reward = float(enemies_killed) * BATTLE_ENEMY_KILL_REWARD
        end_turn_energy_penalty = self._end_turn_energy_penalty(prev_state, action)
        win_reward = BATTLE_WIN_REWARD if battle_result == "won" else 0.0
        loss_penalty = -BATTLE_LOSS_PENALTY if battle_result == "lost" else 0.0
        hp_penalty = -float(step_hp_lost) * BATTLE_HP_LOSS_PENALTY
        gold_penalty = (
            -float(total_gold_lost) * BATTLE_GOLD_LOSS_PENALTY if battle_result is not None else 0.0
        )
        max_hp_penalty = (
            -float(total_max_hp_lost) * BATTLE_MAX_HP_LOSS_PENALTY
            if battle_result is not None
            else 0.0
        )
        reward = (
            hp_penalty
            + gold_penalty
            + max_hp_penalty
            + potion_penalty
            + win_reward
            + loss_penalty
            + enemy_damage_reward
            + enemy_kill_reward
            + end_turn_energy_penalty
        )
        if battle_result is not None:
            self._battle_start_hp = None
            self._battle_start_gold = None
            self._battle_start_max_hp = None
            self._battle_reward_closed = True

        return reward, {
            "type": "battle",
            "active": True,
            "prev_state_type": prev_state_type,
            "next_state_type": next_state_type,
            "result": battle_result,
            "prev_hp": prev_hp,
            "next_hp": next_hp,
            "prev_gold": prev_gold,
            "next_gold": next_gold,
            "prev_max_hp": prev_max_hp,
            "next_max_hp": next_max_hp,
            "battle_start_hp": battle_start_hp,
            "battle_start_gold": battle_start_gold,
            "battle_start_max_hp": battle_start_max_hp,
            "step_hp_lost": step_hp_lost,
            "hp_lost": total_hp_lost,
            "gold_lost": total_gold_lost,
            "max_hp_lost": total_max_hp_lost,
            "hp_penalty": hp_penalty,
            "gold_penalty": gold_penalty,
            "max_hp_penalty": max_hp_penalty,
            "enemy_hp_lost": enemy_hp_lost,
            "enemies_killed": enemies_killed,
            "enemy_damage_reward": enemy_damage_reward,
            "enemy_kill_reward": enemy_kill_reward,
            "end_turn_energy_penalty": end_turn_energy_penalty,
            "potion_used": potion_used,
            "potion_discarded": potion_discarded,
            "potion_penalty": potion_penalty,
            "win_reward": win_reward,
            "loss_penalty": loss_penalty,
            "total": reward,
        }

    def _battle_result(self, prev_state: dict, next_state: dict) -> str | None:
        """Infer battle termination status from adjacent raw states."""
        next_state_type = next_state.get("state_type")
        if not self._is_battle_reward_state(prev_state):
            return None
        if next_state_type == "game_over":
            return "lost"
        if next_state_type in BATTLE_REWARD_STATE_TYPES:
            if self._battle_start_hp is not None or battle_has_alive_enemy(prev_state):
                return "won"
            return None
        if (
            next_state_type in BATTLE_STATE_TYPES
            and battle_has_alive_enemy(prev_state)
            and not battle_has_alive_enemy(next_state)
            and enemy_hp_map(prev_state)
        ):
            return "won"
        return None

    def _enemy_hp_progress(
        self,
        prev_state: dict,
        next_state: dict,
        count_missing_as_dead: bool,
    ) -> tuple[int, int]:
        """Measure diagnostic enemy HP damage and kill count between two battle states."""
        prev_enemies = enemy_hp_map(prev_state)
        next_enemies = enemy_hp_map(next_state)
        hp_lost = 0
        killed = 0

        for enemy_key, prev_hp in prev_enemies.items():
            next_hp = next_enemies.get(enemy_key)
            if next_hp is None:
                next_hp = 0 if count_missing_as_dead else prev_hp

            hp_lost += max(0, prev_hp - next_hp)
            if prev_hp > 0 and next_hp <= 0:
                killed += 1

        return hp_lost, killed

    def _end_turn_energy_penalty(self, prev_state: dict, action: dict | None) -> float:
        """Keep the old detail field while the configured penalty is zero."""
        if not action or action.get("type") != "end_turn":
            return 0.0

        energy = parse_int(prev_state.get("player", {}).get("energy", 0))
        return -BATTLE_UNUSED_ENERGY_PENALTY * float(max(0, energy))


class RunProgressReward(RewardModel):
    """Run-scoped reward for floors, acts, and death."""

    def compute(
        self,
        prev_state: dict,
        next_state: dict,
        action: dict | None = None,
    ) -> tuple[float, dict]:
        """Compute run-progress reward for one transition."""
        prev_run = prev_state.get("run", {})
        next_run = next_state.get("run", {})
        prev_floor = parse_int(prev_run.get("floor", 0))
        next_floor = parse_int(next_run.get("floor", prev_floor))
        prev_act = parse_int(prev_run.get("act", 0))
        next_act = parse_int(next_run.get("act", prev_act))
        floor_delta = next_floor - prev_floor
        act_delta = next_act - prev_act

        floor_reward = float(floor_delta) * RUN_FLOOR_REWARD
        act_reward = float(act_delta) * RUN_ACT_REWARD
        game_over_penalty = (
            -RUN_GAME_OVER_PENALTY if next_state.get("state_type") == "game_over" else 0.0
        )
        reward = floor_reward + act_reward + game_over_penalty

        return reward, {
            "type": "run",
            "prev_state_type": prev_state.get("state_type"),
            "next_state_type": next_state.get("state_type"),
            "prev_floor": prev_floor,
            "next_floor": next_floor,
            "floor_delta": floor_delta,
            "floor_reward": floor_reward,
            "prev_act": prev_act,
            "next_act": next_act,
            "act_delta": act_delta,
            "act_reward": act_reward,
            "game_over_penalty": game_over_penalty,
            "total": reward,
        }


class ScopedRewardModel(RewardModel):
    """Coordinate battle-training and run-progress reward scopes."""

    def __init__(
        self,
        battle_reward_model: BattleOutcomeReward | None = None,
        run_reward_model: RunProgressReward | None = None,
    ) -> None:
        self.battle_reward_model = battle_reward_model or BattleOutcomeReward()
        self.run_reward_model = run_reward_model or RunProgressReward()

    def reset(self, raw_state: dict | None = None) -> None:
        """Reset all stateful reward scopes."""
        self.battle_reward_model.reset(raw_state)
        self.run_reward_model.reset(raw_state)

    def compute(
        self,
        prev_state: dict,
        next_state: dict,
        action: dict | None = None,
    ) -> tuple[float, dict]:
        """Return total metric reward plus scoped rewards for training."""
        battle_reward, battle_details = self.battle_reward_model.compute(
            prev_state,
            next_state,
            action,
        )
        run_reward, run_details = self.run_reward_model.compute(prev_state, next_state, action)
        total = battle_reward + run_reward
        battle_active = bool(battle_details.get("active"))

        details = {
            "type": "battle" if battle_active else "run",
            "prev_state_type": prev_state.get("state_type"),
            "next_state_type": next_state.get("state_type"),
            "battle_reward": battle_reward,
            "run_reward": run_reward,
            "total": total,
            "battle_details": battle_details,
            "run_details": run_details,
        }
        if battle_active:
            details.update(battle_details)
            details["battle_reward"] = battle_reward
            details["run_reward"] = run_reward
            details["total"] = total
            details["battle_details"] = battle_details
            details["run_details"] = run_details
        return total, details
