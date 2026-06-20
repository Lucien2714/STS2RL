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
)
from sts2rl.env.state import (
    battle_has_alive_enemy,
    enemy_hp_map,
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
            "total": reward,
        }


class BattleProgressReward(RewardModel):
    """Hand-tuned reward model for battle progress and run advancement."""

    def __init__(self) -> None:
        self.reset()

    def reset(self, raw_state: dict | None = None) -> None:
        """Clear battle bookkeeping and seed HP tracking from an optional state."""
        self._last_player_hp = (
            player_hp(raw_state, None) if raw_state is not None else None
        )
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
        """Compute reward for one raw-state transition."""
        if self._is_battle_reward_state(prev_state):
            reward, details = self._compute_battle_reward(prev_state, next_state, action)
        else:
            reward, details = self._compute_default_reward(prev_state, next_state)

        self._last_player_hp = player_hp(next_state, self._last_player_hp)
        return reward, details

    def _is_battle_reward_state(self, state: dict) -> bool:
        """Return whether a transition should use battle reward shaping."""
        return state.get("state_type") in BATTLE_STATE_TYPES or state.get("in_battle") is True

    def _compute_default_reward(
        self,
        prev_state: dict,
        next_state: dict,
    ) -> tuple[float, dict]:
        """Reward non-battle floor progress and penalize HP loss."""
        prev_hp = player_hp(prev_state)
        next_hp = player_hp(next_state)
        prev_floor = prev_state.get("run", {}).get("floor", 0)
        next_floor = next_state.get("run", {}).get("floor", 0)

        reward = 0.0
        reward += float(next_floor - prev_floor) * 10.0
        reward += float(next_hp - prev_hp) * 0.2

        if next_state.get("state_type") == "game_over":
            reward -= 10.0

        return reward, {
            "type": "default",
            "total": reward,
        }

    def _compute_battle_reward(
        self,
        prev_state: dict,
        next_state: dict,
        action: dict | None = None,
    ) -> tuple[float, dict]:
        """Reward enemy progress and battle outcomes while tracking battle starts."""
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
            self._battle_start_gold
            if self._battle_start_gold is not None
            else prev_gold
        )
        battle_start_max_hp = (
            self._battle_start_max_hp
            if self._battle_start_max_hp is not None
            else prev_max_hp
        )
        total_hp_lost = max(0, battle_start_hp - next_hp)
        total_gold_lost = max(0, battle_start_gold - next_gold)
        total_max_hp_lost = max(0, battle_start_max_hp - next_max_hp)

        potion_used = bool(action and action.get("type") == "use_potion")
        potion_penalty = -BATTLE_POTION_USE_PENALTY if potion_used else 0.0
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
            -float(total_gold_lost) * BATTLE_GOLD_LOSS_PENALTY
            if battle_result is not None
            else 0.0
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
            "potion_penalty": potion_penalty,
            "win_reward": win_reward,
            "loss_penalty": loss_penalty,
            "total": reward,
        }

    def _battle_result(self, prev_state: dict, next_state: dict) -> str | None:
        """Infer battle termination status from adjacent raw states."""
        prev_state_type = prev_state.get("state_type")
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
        """Measure enemy HP damage and kill count between two battle states."""
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
        """Penalize ending turn with unused player energy."""
        if not action or action.get("type") != "end_turn":
            return 0.0

        energy = self._parse_int(prev_state.get("player", {}).get("energy", 0))
        return -BATTLE_UNUSED_ENERGY_PENALTY * float(max(0, energy))

    def _parse_int(self, value: object, default: int = 0) -> int:
        """Parse an integer-like value with a safe default."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
