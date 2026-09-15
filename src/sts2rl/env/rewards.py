"""Reward-model interfaces and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from sts2rl.env.constants import (
    BOSS_VICTORY_REWARD,
    NODE_PROGRESS_REWARD,
    STEP_COST,
)


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

    One point per node entered, ten per boss beaten, and a small charge per
    step so standing still is never free.  Deliberately not scored: HP, enemy
    damage, kills, gold, potions, and unspent energy.  Those are means, not
    ends, and hand weighting them is how a reward model stops matching the
    objective.

    A boss is scored as *leaving its floor*: the run enters a boss room at some
    floor, and the only way past that floor is to win.  Both halves of that are
    read off ``run.floor``, the monotone counter the node reward already uses.

    It used to be inferred from the battle instead -- a flag armed while the
    boss screen showed a live enemy and consumed when none was left.  "No enemy
    is alive" is also what every screen carrying no battle reports, so opening a
    card-selection prompt mid-fight read as a victory, and reopening it re-armed
    the flag: ten points every other step against a node worth one.  Measured
    over 621 episodes, four of seven boss bonuses went to runs that then died on
    the boss floor, which winning makes impossible.

    The act counter would be the obvious stateless alternative and is wrong:
    the third act carries two bosses at the highest difficulty, so an act-based
    bonus would pay once for two of them.  A floor is one room, so this counts
    them one at a time.

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
        """Forget a boss room the previous run had entered."""
        del raw_state
        self._boss_floor: int | None = None

    def compute(
        self,
        prev_state: dict,
        next_state: dict,
        action: dict | None = None,
    ) -> tuple[float, dict]:
        """Return the reward for one transition and a breakdown of it."""
        del action

        # A missing run block means "this payload does not say", which is not
        # the same as floor zero.  Reading it as zero made the drop free and
        # the rebound pay the whole floor over again: one measured episode was
        # paid 18 nodes it had not climbed.
        prev_floor = _floor(prev_state)
        next_floor = _floor(next_state)
        nodes = 0
        if prev_floor is not None and next_floor is not None:
            nodes = max(0, next_floor - prev_floor)
        bosses = self._resolve_boss(prev_state, next_state)

        details: dict[str, object] = {"type": "run_progress"}
        reward = -STEP_COST
        if nodes:
            reward += nodes * NODE_PROGRESS_REWARD
            details["nodes"] = nodes
        if bosses:
            reward += bosses * BOSS_VICTORY_REWARD
            details["bosses_defeated"] = bosses

        details["total"] = reward
        return reward, details

    def _resolve_boss(self, prev_state: dict, next_state: dict) -> int:
        """Return how many boss floors this transition left behind.

        A boss room is remembered by its floor while the run is in it, and the
        bonus is paid once the run stands on a later floor.  Dying there never
        pays, because the floor never advances, and a payload that reports no
        floor settles nothing -- it neither pays nor forgets the room.

        The room the transition *arrives* in is armed only after that check.
        Arming it first let one boss room overwrite the one being left, which
        is exactly the back-to-back pair the third act ends with.
        """
        if prev_state.get("state_type") == "boss" and _floor(prev_state):
            self._boss_floor = _floor(prev_state)

        # Winning ends the run standing on the last boss's own floor, so there
        # is never a later floor to step onto and the rule above cannot see it.
        # The game_over block says so outright.  Paid on the transition *into*
        # the win so that re-scoring a finished run cannot pay again; the
        # episode loop stops at ``done`` anyway, but that is its invariant to
        # keep, not one this model should lean on.
        if _won(next_state):
            self._boss_floor = None
            return 0 if _won(prev_state) else 1

        next_floor = _floor(next_state)
        beaten = (
            self._boss_floor is not None
            and next_floor is not None
            and next_floor > self._boss_floor
        )
        if beaten:
            self._boss_floor = None

        if next_state.get("state_type") == "boss" and next_floor:
            self._boss_floor = next_floor

        return 1 if beaten else 0


def _won(state: dict) -> bool:
    """Return whether this state reports a run that ended in victory.

    ``victory`` is the mod's own boolean, alongside an ``outcome`` enum that
    repeats it.  Strict ``is True`` because this is game JSON: a build that
    omits the field must read as "not a win", not as a truthy object.  The
    ``message`` string also differs on a win and is deliberately not used --
    matching a server's wording is the brittle way to ask this.
    """
    block = state.get("game_over")
    return isinstance(block, dict) and block.get("victory") is True


def _floor(state: dict) -> int | None:
    """Return how many nodes the run has entered, or None when unreported.

    None rather than zero: menus and unresolved frames carry no run block, and
    treating that as the ground floor turns every return to a real screen into
    a fresh climb worth the whole floor number.
    """
    run = state.get("run")
    if not isinstance(run, dict):
        return None
    value = run.get("floor")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
