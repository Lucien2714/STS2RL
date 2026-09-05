"""Episode loop connecting the raw environment, reward model, and an agent."""

from __future__ import annotations

from dataclasses import dataclass

from sts2rl.actions import GameAction
from sts2rl.agents.action_space import NoLegalActionsError
from sts2rl.agents.base import Agent, Transition
from sts2rl.env.game_env import GameEnv
from sts2rl.env.mcp_client import STS2ClientError
from sts2rl.env.reset import ResetSpec
from sts2rl.env.rewards import BattleProgressReward, RewardModel
from sts2rl.env.types import GameObservation, RawState


class ObservationError(RuntimeError):
    """Raised when a required full player-detail snapshot cannot be obtained."""


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


class EpisodeRunner:
    """Execute one complete run while keeping learning outside GameEnv."""

    def __init__(
        self,
        env: GameEnv,
        agent: Agent,
        reward_model: RewardModel | None = None,
        max_steps: int = 10_000,
        max_state_refreshes: int = 3,
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

    def run(self, reset_spec: ResetSpec | None = None) -> EpisodeResult:
        """Reset the environment and run until game over or the step limit."""
        initial_state = self.env.reset(reset_spec)
        observation = self._observation(initial_state)
        transitions: list[Transition] = []
        total_reward = 0.0
        self.reward_model.reset(initial_state)
        self.agent.reset(observation)

        for _ in range(self.max_steps):
            action, observation = self._choose_with_refresh(observation)
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
            next_observation = self._observation(
                env_step.raw_state,
                terminal=env_step.done,
            )
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
    ) -> tuple[GameAction, GameObservation]:
        for _ in range(self.max_state_refreshes):
            try:
                return self.agent.choose_action(observation), observation
            except NoLegalActionsError:
                observation = self._observation(self.env.get_state())
        return self.agent.choose_action(observation), observation

    def _observation(
        self,
        raw_state: RawState,
        *,
        terminal: bool = False,
    ) -> GameObservation:
        if terminal or raw_state.get("state_type") == "game_over":
            return GameObservation(raw_state=raw_state, player_detail=None)
        try:
            player_detail = self.env.get_player_detail()
        except STS2ClientError as exc:
            raise ObservationError(
                "Failed to obtain required player detail for "
                f"state_type={raw_state.get('state_type')!r}: {exc}"
            ) from exc
        return GameObservation(raw_state=raw_state, player_detail=player_detail)
