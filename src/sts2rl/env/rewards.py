"""Reward-model interfaces and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from sts2rl.env.constants import (
    BOSS_VICTORY_REWARD,
    NODE_PROGRESS_REWARD,
    STEP_COST,
)
from sts2rl.env.state import battle_has_alive_enemy


class RewardModel(ABC):
    """Score one raw environment transition."""

    @abstractmethod
    def compute(
        self,
        prev_state: dict,
        next_state: dict,
        action: dict | None = None,
    ) -> tuple[float, dict]:
        """Return the reward for one transition and a breakdown of it."""

    def reset(self, raw_state: dict | None = None) -> None:
        """Clear per-run bookkeeping before a new episode."""

    def action_error_reward(self, error: object) -> tuple[float, dict]:
        """Score a rejected action, which changed nothing in the game."""
        return 0.0, {
            "type": "action_error",
            "action_error": True,
            "error": str(error),
            "total": 0.0,
        }


class RunProgressReward(RewardModel):
    """Score climbing the spire, and nothing else.

    One point per node entered, ten for a boss, and a small charge per step so
    standing still is never free.  Deliberately not scored: HP, enemy damage,
    kills, gold, potions, and unspent energy.  Those are means, not ends, and
    hand weighting them is how a reward model stops matching the objective.

    HP was scored here once, and it taught exactly the wrong lesson: healing
    paid immediately, so the agent rested at every rest site instead of
    upgrading a card, because upgrading pays nothing this step.  HP still
    matters -- running out ends the run and with it the progress -- but it
    matters *terminally*, and pricing a terminal consideration per step is how
    reward hacking starts.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self, raw_state: dict | None = None) -> None:
        """Forget any boss the previous run was fighting."""
        del raw_state
        self._boss_pending = False

    def compute(
        self,
        prev_state: dict,
        next_state: dict,
        action: dict | None = None,
    ) -> tuple[float, dict]:
        """Return the reward for one transition and a breakdown of it."""
        del action

        nodes = max(0, _floor(next_state) - _floor(prev_state))
        boss = self._resolve_boss(prev_state, next_state)

        details: dict[str, object] = {"type": "run_progress"}
        reward = -STEP_COST
        if nodes:
            reward += nodes * NODE_PROGRESS_REWARD
            details["nodes"] = nodes
        if boss:
            reward += BOSS_VICTORY_REWARD
            details["boss_defeated"] = True

        details["total"] = reward
        return reward, details

    def _resolve_boss(self, prev_state: dict, next_state: dict) -> bool:
        """Return whether this transition ended a boss fight in a victory.

        A boss fight spans many transitions, so the pending flag is raised
        while the boss is alive and consumed exactly once when it is not.
        """
        if prev_state.get("state_type") == "boss" and battle_has_alive_enemy(prev_state):
            self._boss_pending = True

        if not self._boss_pending:
            return False
        if next_state.get("state_type") == "game_over":
            self._boss_pending = False
            return False
        if battle_has_alive_enemy(next_state):
            return False

        self._boss_pending = False
        return True


def _floor(state: dict) -> int:
    """Return how many nodes the run has entered, or 0 when unreported."""
    run = state.get("run")
    if not isinstance(run, dict):
        return 0
    value = run.get("floor")
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value
