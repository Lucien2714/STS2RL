"""PPO over a variable set of complete structured action candidates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F

from sts2rl.actions import GameAction
from sts2rl.agents.action_space import LegalActionProvider
from sts2rl.agents.base import Agent, Transition
from sts2rl.encoder import GameEncoder, GameTokenizer, TokenizedDecision
from sts2rl.env.types import GameObservation


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters for candidate-action PPO."""

    learning_rate: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 0.98
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    rollout_size: int = 256
    update_epochs: int = 4
    minibatch_size: int = 32

    def __post_init__(self) -> None:
        if self.rollout_size < 1 or self.update_epochs < 1:
            raise ValueError("rollout_size and update_epochs must be positive")
        if self.minibatch_size < 1:
            raise ValueError("minibatch_size must be positive")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be between 0 and 1")
        if self.clip_ratio < 0:
            raise ValueError("clip_ratio must not be negative")


@dataclass
class _PendingDecision:
    decision: TokenizedDecision
    action_index: int
    log_probability: Tensor
    value: Tensor


@dataclass
class _RolloutStep:
    decision: TokenizedDecision
    next_observation: GameObservation
    action_index: int
    old_log_probability: Tensor
    old_value: Tensor
    reward: float
    done: bool


class CandidatePPOAgent(Agent):
    """PPO agent trained end-to-end over structured dynamic candidates."""

    def __init__(
        self,
        tokenizer: GameTokenizer,
        game_encoder: GameEncoder,
        action_provider: LegalActionProvider | None = None,
        config: PPOConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.game_encoder = game_encoder
        self.action_provider = action_provider or LegalActionProvider()
        self.config = config or PPOConfig()
        self.device = torch.device(device or "cpu")
        self.game_encoder.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.game_encoder.parameters(), lr=self.config.learning_rate
        )
        self.training_enabled = True
        self._pending: _PendingDecision | None = None
        self._forced_action = False
        self._carried_reward = 0.0
        self._rollout: list[_RolloutStep] = []
        self.last_update: dict[str, float] = {}
        self.environment_steps = 0
        self.optimizer_updates = 0
        self._completed_update_metrics: list[dict[str, float]] = []

    def reset(self, initial_state: GameObservation) -> None:
        del initial_state
        self._pending = None
        self._forced_action = False
        self._carried_reward = 0.0

    def choose_action(self, state: GameObservation) -> GameAction:
        if self._pending is not None:
            raise RuntimeError(
                "observe() must be called before choosing another action"
            )

        candidates = self.action_provider.require_candidates(state.raw_state)
        # A single candidate carries no policy gradient: its log probability is
        # always zero and its ratio always one, so training on it only dilutes
        # the advantage statistics of genuine decisions.
        if len(candidates) == 1:
            self._forced_action = True
            return candidates[0]

        decision = self.tokenizer.tokenize_decision(state, candidates)
        with torch.no_grad():
            output = self.game_encoder.policy_value(decision.to(self.device))
            distribution = Categorical(logits=output.logits)
            if self.training_enabled:
                action_index_tensor = distribution.sample()
            else:
                action_index_tensor = torch.argmax(output.logits)
            log_probability = distribution.log_prob(action_index_tensor)

        action_index = int(action_index_tensor.item())
        if self.training_enabled:
            self._pending = _PendingDecision(
                decision=decision,
                action_index=action_index,
                log_probability=log_probability.detach().cpu(),
                value=output.value.detach().cpu(),
            )
        return candidates[action_index]

    def observe(self, transition: Transition) -> None:
        forced = self._forced_action
        self._forced_action = False
        if not self.training_enabled:
            return

        self.environment_steps += 1
        if forced:
            self._absorb_forced_transition(transition)
        else:
            if self._pending is None:
                raise RuntimeError("choose_action() must be called before observe()")
            self._rollout.append(
                _RolloutStep(
                    decision=self._pending.decision,
                    next_observation=transition.next_state,
                    action_index=self._pending.action_index,
                    old_log_probability=self._pending.log_probability,
                    old_value=self._pending.value,
                    reward=float(transition.reward) + self._carried_reward,
                    done=transition.done,
                )
            )
            self._carried_reward = 0.0
            self._pending = None
        if self._rollout and (
            len(self._rollout) >= self.config.rollout_size or self._rollout[-1].done
        ):
            self.update()

    def _absorb_forced_transition(self, transition: Transition) -> None:
        """Merge a forced step into the decision it followed.

        Forced steps are never scored, so their reward would otherwise be lost.
        Extending the preceding recorded transition keeps the return of every
        trained decision equal to the return the environment actually paid.
        """
        if not self._rollout or self._pending is not None:
            self._carried_reward += float(transition.reward)
            return
        last = self._rollout[-1]
        last.reward += float(transition.reward)
        last.next_observation = transition.next_state
        last.done = transition.done

    def finish_episode(self, final_state: GameObservation, truncated: bool) -> None:
        del final_state, truncated
        if self._pending is not None:
            raise RuntimeError("cannot finish an episode with an unobserved action")
        if self.training_enabled and self._rollout:
            self.update()

    def train(self, enabled: bool = True) -> None:
        """Switch between stochastic learning and deterministic evaluation."""
        if not enabled and self._pending is not None:
            raise RuntimeError("cannot change mode with an unobserved action")
        self.training_enabled = enabled
        self.game_encoder.train(enabled)

    def eval(self) -> None:
        """Use deterministic candidate selection without collecting rollouts."""
        self.train(False)

    def checkpoint_state(self) -> dict[str, object]:
        """Return model and optimizer tensors at a clean boundary.

        Lifetime counters belong to ``TrainingState``, which the checkpoint
        stores once; the caller restores them onto the agent.
        """
        self._require_clean_checkpoint_boundary("save")
        return {
            "encoder": self.game_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_checkpoint_state(self, state: Mapping[str, object]) -> None:
        """Restore model and optimizer tensors into an unused agent."""
        self._require_clean_checkpoint_boundary("load")
        encoder_state = state.get("encoder")
        optimizer_state = state.get("optimizer")
        if not isinstance(encoder_state, Mapping):
            raise ValueError("checkpoint agent encoder must be a mapping")
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("checkpoint agent optimizer must be a mapping")

        self.game_encoder.load_state_dict(dict(encoder_state))
        self.optimizer.load_state_dict(dict(optimizer_state))
        self._move_optimizer_state_to_device()
        self.last_update = {}
        self._completed_update_metrics.clear()

    def drain_update_metrics(self) -> tuple[dict[str, float], ...]:
        """Return completed PPO update metrics once, in completion order."""
        metrics = tuple(dict(item) for item in self._completed_update_metrics)
        self._completed_update_metrics.clear()
        return metrics

    def abort_episode(self) -> None:
        """Discard an incomplete action and rollout without undoing prior updates."""
        self._pending = None
        self._forced_action = False
        self._carried_reward = 0.0
        self._rollout.clear()

    def update(self) -> dict[str, float]:
        """Run PPO updates over the current variable-length rollout."""
        if not self._rollout:
            return {}

        advantages, returns = self._advantages_and_returns()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        metrics: dict[str, float] = {}
        for _ in range(self.config.update_epochs):
            order = torch.randperm(len(self._rollout)).tolist()
            for start in range(0, len(order), self.config.minibatch_size):
                metrics = self._optimize_minibatch(
                    order[start : start + self.config.minibatch_size],
                    advantages,
                    returns,
                )

        self.optimizer_updates += 1
        metrics["environment_steps"] = float(self.environment_steps)
        metrics["optimizer_update"] = float(self.optimizer_updates)
        self._rollout.clear()
        self.last_update = metrics
        self._completed_update_metrics.append(dict(metrics))
        return metrics

    def _optimize_minibatch(
        self,
        indices: list[int],
        advantages: Tensor,
        returns: Tensor,
    ) -> dict[str, float]:
        """Run one clipped-surrogate optimizer step over a subset of the rollout."""
        policy_losses: list[Tensor] = []
        value_losses: list[Tensor] = []
        entropies: list[Tensor] = []
        for index in indices:
            step = self._rollout[index]
            output = self.game_encoder.policy_value(step.decision.to(self.device))
            distribution = Categorical(logits=output.logits)
            action_index = torch.tensor(step.action_index, device=self.device)
            new_log_probability = distribution.log_prob(action_index)
            ratio = torch.exp(
                new_log_probability - step.old_log_probability.to(self.device)
            )
            advantage = advantages[index]
            unclipped = ratio * advantage
            clipped = (
                torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                )
                * advantage
            )
            policy_losses.append(-torch.minimum(unclipped, clipped))
            value_losses.append(F.mse_loss(output.value, returns[index]))
            entropies.append(distribution.entropy())

        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy = torch.stack(entropies).mean()
        loss = (
            policy_loss
            + self.config.value_coefficient * value_loss
            - self.config.entropy_coefficient * entropy
        )
        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.game_encoder.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "rollout_steps": float(len(self._rollout)),
        }

    def _advantages_and_returns(self) -> tuple[Tensor, Tensor]:
        values = torch.stack([step.old_value for step in self._rollout]).to(self.device)
        advantages = torch.zeros(len(self._rollout), device=self.device)
        with torch.no_grad():
            last_step = self._rollout[-1]
            next_value = (
                torch.zeros((), device=self.device)
                if last_step.done
                else self.game_encoder.value(
                    self.tokenizer.tokenize_state(last_step.next_observation).to(
                        self.device
                    )
                )
            )
            gae = torch.zeros((), device=self.device)
            for index in range(len(self._rollout) - 1, -1, -1):
                step = self._rollout[index]
                nonterminal = 0.0 if step.done else 1.0
                delta = (
                    step.reward
                    + self.config.gamma * next_value * nonterminal
                    - values[index]
                )
                gae = (
                    delta
                    + self.config.gamma * self.config.gae_lambda * nonterminal * gae
                )
                advantages[index] = gae
                next_value = values[index]
        return advantages, advantages + values

    def _require_clean_checkpoint_boundary(self, operation: str) -> None:
        if self._pending is not None:
            raise RuntimeError(
                f"cannot {operation} a checkpoint with an unobserved action"
            )
        if self._rollout:
            raise RuntimeError(
                f"cannot {operation} a checkpoint with a non-empty rollout"
            )

    def _move_optimizer_state_to_device(self) -> None:
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if isinstance(value, Tensor):
                    optimizer_state[key] = value.to(self.device)
