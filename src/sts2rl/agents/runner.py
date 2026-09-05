"""Episode loop connecting the raw environment, reward model, and an agent."""

from __future__ import annotations

from dataclasses import dataclass
import time

from sts2rl.actions import GameAction
from sts2rl.agents.action_space import NoLegalActionsError
from sts2rl.agents.base import Agent, Transition
from sts2rl.env.game_env import GameEnv
from sts2rl.env.reset import ResetSpec
from sts2rl.env.rewards import BattleProgressReward, RewardModel
from sts2rl.env.types import GameObservation, RawState


@dataclass(frozen=True)
class EpisodeResult:
    """Summary and transitions from one runner invocation."""

    initial_state: RawState
    final_state: RawState
    transitions: tuple[Transition, ...]
    total_reward: float
    terminated: bool
    truncated: bool

    @property
    def steps(self) -> int:
        return len(self.transitions)


REFRESH_BACKOFF_SECONDS = 0.25


class EpisodeRunner:
    """Execute one complete run while keeping learning outside GameEnv."""

    def __init__(
        self,
        env: GameEnv,
        agent: Agent,
        reward_model: RewardModel | None = None,
        max_steps: int = 10_000,
        max_state_refreshes: int = 3,
        refresh_backoff_seconds: float = REFRESH_BACKOFF_SECONDS,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if max_state_refreshes < 0:
            raise ValueError("max_state_refreshes must not be negative")
        self.env = env
        self.agent = agent
        self.reward_model = reward_model or BattleProgressReward()
        self.max_steps = max_steps
        self.max_state_refreshes = max_state_refreshes
        self.refresh_backoff_seconds = refresh_backoff_seconds

    def run(self, reset_spec: ResetSpec | None = None) -> EpisodeResult:
        """Reset the environment and run until game over or the step limit."""
        initial_state = self.env.reset(reset_spec)
        observation = GameObservation(initial_state)
        transitions: list[Transition] = []
        total_reward = 0.0
        self.reward_model.reset(initial_state)
        self.agent.reset(observation)

        for _ in range(self.max_steps):
            decision = self._choose_with_refresh(observation)
            if decision is None:
                break
            action, observation = decision
            env_step = self.env.step(action)
            if env_step.info.get("action_error"):
                reward, reward_info = self.reward_model.action_error_reward(
                    env_step.info.get("error")
                )
            else:
                reward, reward_info = self.reward_model.compute(
                    observation.raw_state,
                    env_step.raw_state,
                    action.to_dict(),
                )
            next_observation = GameObservation(env_step.raw_state)
            transition = Transition(
                state=observation,
                action=action,
                reward=float(reward),
                next_state=next_observation,
                done=env_step.done,
                info={**env_step.info, "reward": reward_info},
            )
            transitions.append(transition)
            total_reward += float(reward)
            self.agent.observe(transition)
            observation = next_observation
            if env_step.done:
                self.agent.finish_episode(observation, truncated=False)
                return EpisodeResult(
                    initial_state=initial_state,
                    final_state=observation.raw_state,
                    transitions=tuple(transitions),
                    total_reward=total_reward,
                    terminated=True,
                    truncated=False,
                )

        self.agent.finish_episode(observation, truncated=True)
        return EpisodeResult(
            initial_state=initial_state,
            final_state=observation.raw_state,
            transitions=tuple(transitions),
            total_reward=total_reward,
            terminated=False,
            truncated=True,
        )

    def _choose_with_refresh(
        self, observation: GameObservation
    ) -> tuple[GameAction, GameObservation] | None:
        """Choose an action, re-reading state while none is legal yet.

        Combat states outside the player's play phase legally expose no action,
        so the server is given time to settle before each retry.  Returning
        None truncates the episode instead of failing the whole training run.
        """
        for attempt in range(self.max_state_refreshes + 1):
            try:
                return self.agent.choose_action(observation), observation
            except NoLegalActionsError:
                if attempt == self.max_state_refreshes:
                    return None
                if self.refresh_backoff_seconds:
                    time.sleep(self.refresh_backoff_seconds * (2**attempt))
                observation = GameObservation(self.env.get_state())

