"""PPO over a variable set of complete structured action candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F

from sts2rl.actions import GameAction
from sts2rl.agents.action_space import LegalActionProvider
from sts2rl.agents.base import Agent, Transition
from sts2rl.agents.features import FeatureEncoder, HashingFeatureEncoder
from sts2rl.env.types import RawState


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters for candidate-action PPO."""

    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    rollout_size: int = 256
    update_epochs: int = 4

    def __post_init__(self) -> None:
        if self.hidden_dim < 1 or self.rollout_size < 1 or self.update_epochs < 1:
            raise ValueError(
                "hidden_dim, rollout_size, and update_epochs must be positive"
            )
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be between 0 and 1")
        if self.clip_ratio < 0:
            raise ValueError("clip_ratio must not be negative")


class CandidateActorCritic(nn.Module):
    """Score each candidate conditioned on state and estimate state value."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.state_tower = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.action_tower = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        self.policy_bias = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state_features: Tensor,
        candidate_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        state_embedding = self.state_tower(state_features)
        action_embeddings = self.action_tower(candidate_features)
        logits = action_embeddings @ state_embedding / math.sqrt(
            state_embedding.shape[-1]
        ) + self.policy_bias(action_embeddings).squeeze(-1)
        value = self.value_head(state_embedding).squeeze(-1)
        return logits, value

    def value(self, state_features: Tensor) -> Tensor:
        """Estimate state value without requiring action candidates."""
        return self.value_head(self.state_tower(state_features)).squeeze(-1)


@dataclass
class _PendingDecision:
    state_features: Tensor
    candidate_features: Tensor
    candidates: tuple[GameAction, ...]
    action_index: int
    log_probability: Tensor
    value: Tensor


@dataclass
class _RolloutStep:
    state_features: Tensor
    candidate_features: Tensor
    next_state_features: Tensor
    action_index: int
    old_log_probability: Tensor
    old_value: Tensor
    reward: float
    done: bool


class CandidatePPOAgent(Agent):
    """PPO agent whose categorical support is rebuilt from each raw state."""

    def __init__(
        self,
        feature_encoder: FeatureEncoder | None = None,
        action_provider: LegalActionProvider | None = None,
        config: PPOConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.feature_encoder = feature_encoder or HashingFeatureEncoder()
        self.action_provider = action_provider or LegalActionProvider()
        self.config = config or PPOConfig()
        self.device = torch.device(device or "cpu")
        self.model = CandidateActorCritic(
            self.feature_encoder.state_dim,
            self.feature_encoder.action_dim,
            self.config.hidden_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.training_enabled = True
        self._pending: _PendingDecision | None = None
        self._rollout: list[_RolloutStep] = []
        self.last_update: dict[str, float] = {}

    def reset(self, initial_state: RawState) -> None:
        del initial_state
        self._pending = None

    def choose_action(self, state: RawState) -> GameAction:
        if self._pending is not None:
            raise RuntimeError(
                "observe() must be called before choosing another action"
            )

        candidates = self.action_provider.require_candidates(state)
        state_features = self._state_features(state)
        candidate_features = self._candidate_features(state, candidates)
        with torch.no_grad():
            logits, value = self.model(state_features, candidate_features)
            distribution = Categorical(logits=logits)
            if self.training_enabled:
                action_index_tensor = distribution.sample()
            else:
                action_index_tensor = torch.argmax(logits)
            log_probability = distribution.log_prob(action_index_tensor)

        action_index = int(action_index_tensor.item())
        if self.training_enabled:
            self._pending = _PendingDecision(
                state_features=state_features.detach(),
                candidate_features=candidate_features.detach(),
                candidates=candidates,
                action_index=action_index,
                log_probability=log_probability.detach(),
                value=value.detach(),
            )
        return candidates[action_index]

    def observe(self, transition: Transition) -> None:
        if not self.training_enabled:
            return
        if self._pending is None:
            raise RuntimeError("choose_action() must be called before observe()")

        chosen_action = self._pending.candidates[self._pending.action_index]
        if chosen_action.to_dict() != transition.action.to_dict():
            raise ValueError("observed action does not match the sampled PPO action")

        self._rollout.append(
            _RolloutStep(
                state_features=self._pending.state_features,
                candidate_features=self._pending.candidate_features,
                next_state_features=self._state_features(
                    transition.next_state
                ).detach(),
                action_index=self._pending.action_index,
                old_log_probability=self._pending.log_probability,
                old_value=self._pending.value,
                reward=float(transition.reward),
                done=transition.done,
            )
        )
        self._pending = None
        if len(self._rollout) >= self.config.rollout_size or transition.done:
            self.update()

    def finish_episode(self, final_state: RawState, truncated: bool) -> None:
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
        self.model.train(enabled)

    def eval(self) -> None:
        """Use deterministic candidate selection without collecting rollouts."""
        self.train(False)

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
            policy_losses: list[Tensor] = []
            value_losses: list[Tensor] = []
            entropies: list[Tensor] = []
            for index, step in enumerate(self._rollout):
                logits, value = self.model(
                    step.state_features.to(self.device),
                    step.candidate_features.to(self.device),
                )
                distribution = Categorical(logits=logits)
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
                value_losses.append(F.mse_loss(value, returns[index]))
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
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            metrics = {
                "loss": float(loss.detach().cpu()),
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "rollout_steps": float(len(self._rollout)),
            }

        self._rollout.clear()
        self.last_update = metrics
        return metrics

    def _advantages_and_returns(self) -> tuple[Tensor, Tensor]:
        values = torch.stack([step.old_value for step in self._rollout]).to(self.device)
        advantages = torch.zeros(len(self._rollout), device=self.device)
        with torch.no_grad():
            last_step = self._rollout[-1]
            next_value = (
                torch.zeros((), device=self.device)
                if last_step.done
                else self.model.value(last_step.next_state_features.to(self.device))
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

    def _state_features(self, state: RawState) -> Tensor:
        features = self.feature_encoder.encode_state(state).to(self.device)
        if features.ndim != 1 or features.shape[0] != self.feature_encoder.state_dim:
            raise ValueError(
                "encode_state() must return a flat tensor matching state_dim"
            )
        return features

    def _candidate_features(
        self,
        state: RawState,
        candidates: tuple[GameAction, ...],
    ) -> Tensor:
        features = torch.stack(
            [
                self.feature_encoder.encode_action(state, action).to(self.device)
                for action in candidates
            ]
        )
        if features.ndim != 2 or features.shape[1] != self.feature_encoder.action_dim:
            raise ValueError(
                "encode_action() must return a flat tensor matching action_dim"
            )
        return features
