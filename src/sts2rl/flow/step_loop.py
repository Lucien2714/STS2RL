"""The one implementation of "take one step in the game".

Training and evaluation used to carry near-identical copies of this loop body:
decide what to do, dispatch it, score the transition, auto-advance the forced
states that follow, and fold their rewards into the visible step. The copies had
already drifted. Both now call :func:`decide_action` and :func:`apply_action`
here; what stays on each side is genuinely theirs — training adds the model
update, checkpointing, and reconnect recovery, evaluation adds its accounting.

Errors are raised as :class:`StepError`, which carries the best-known game state
at the point of failure so a caller with reconnect logic can recover from the
right place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sts2rl.action_spaces.defaults import default_action
from sts2rl.env.rewards import RewardModel
from sts2rl.flow.battle_flow import (
    advance_forced_end_turn_states,
    fold_reward_details,
    is_forced_end_turn_state,
    should_skip_agent,
)


class StepError(Exception):
    """A step failed mid-flight; ``state`` is the last state we knew about."""

    def __init__(self, message: str, state: dict | None) -> None:
        super().__init__(message)
        self.state = state


@dataclass(frozen=True)
class StepDecision:
    """What the loop decided to do this step, and how it decided it."""

    action: dict
    # end_turn was the only legal action: taken without consulting the agent, and
    # not trained on.
    forced_end_turn: bool = False
    # Combat outside the player's play phase: the agent is bypassed entirely.
    skipped_agent: bool = False

    @property
    def agent_chose(self) -> bool:
        """Return whether the trainable agent actually made this choice."""
        return not (self.forced_end_turn or self.skipped_agent)


@dataclass
class StepOutcome:
    """The result of dispatching one action, with forced follow-ups folded in."""

    next_raw_state: dict
    reward: float
    done: bool
    reward_details: dict
    info: dict = field(default_factory=dict)
    auto_steps: list = field(default_factory=list)

    @property
    def action_error(self) -> bool:
        """Return whether the dispatch itself failed."""
        return bool(self.info.get("action_error"))

    @property
    def error(self) -> object:
        """Return the dispatch error, if any."""
        return self.info.get("error")

    @property
    def step_count(self) -> int:
        """Return how many game steps this visible step represents."""
        return 1 + len(self.auto_steps)


def decide_action(agent: Any, raw_state: dict, *, training: bool = True) -> StepDecision:
    """Choose this step's action, bypassing the agent where it has no say.

    Three cases, in order: ``end_turn`` is the only legal move (auto-advanced),
    combat is outside the player's play phase (the agent is skipped and we send
    whatever the screen actually accepts), or the agent decides.
    """
    if is_forced_end_turn_state(agent, raw_state):
        return StepDecision({"type": "end_turn"}, forced_end_turn=True)

    if should_skip_agent(raw_state):
        # Not `proceed`: combat rejects it. default_action resolves to a state
        # refresh here, which is the correct "wait for the server" no-op.
        return StepDecision(default_action(raw_state), skipped_agent=True)

    policy_state = {
        "screen_type": raw_state.get("state_type"),
        "raw_state": raw_state,
    }
    return StepDecision(agent.choose_action(policy_state, training=training))


def apply_action(
    game: Any,
    agent: Any,
    reward_model: RewardModel,
    prev_raw_state: dict,
    action: dict,
) -> StepOutcome:
    """Dispatch one action, score it, and fold any forced follow-ups into it.

    Raises :class:`StepError` if the client call fails outright (as opposed to
    the API rejecting the action, which comes back as ``action_error`` in the
    returned outcome).
    """
    try:
        next_raw_state, done, info = game.step(action)
    except Exception as exc:
        raise StepError(str(exc), prev_raw_state) from exc

    if info.get("action_error"):
        reward, reward_details = reward_model.action_error_reward(
            info.get("error", "action dispatch failed")
        )
    else:
        reward, reward_details = reward_model.compute(prev_raw_state, next_raw_state, action)

    auto_steps: list = []
    if next_raw_state is not None and not done:
        try:
            next_raw_state, auto_reward, auto_done, auto_steps = advance_forced_end_turn_states(
                game,
                agent,
                reward_model,
                next_raw_state,
            )
        except Exception as exc:
            raise StepError(str(exc), next_raw_state) from exc
        reward += auto_reward
        done = done or auto_done

    return StepOutcome(
        next_raw_state=next_raw_state,
        reward=reward,
        done=done,
        reward_details=fold_reward_details(reward_details, reward, auto_steps),
        info=info,
        auto_steps=auto_steps,
    )
