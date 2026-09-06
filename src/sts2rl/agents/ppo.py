"""PPO over a variable set of complete structured action candidates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import threading

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


@dataclass
class _Lane:
    """One environment's in-flight decision and its slice of the rollout.

    Every client steps its own game, so the decision awaiting an observation
    and the reward carried across forced steps are per-environment.  The
    rollout is kept per lane as well because GAE walks a trajectory backwards:
    interleaving two games into one flat list would make step i-1 the temporal
    predecessor of step i only by accident, and the advantage would propagate
    across environments without anything failing.
    """

    pending: _PendingDecision | None = None
    forced_action: bool = False
    carried_reward: float = 0.0
    steps: list[_RolloutStep] = field(default_factory=list)


class LaneView(Agent):
    """One environment's handle on the shared agent.

    ``EpisodeRunner`` takes an ``Agent``; this binds every call to a lane so
    the runner needs to know nothing about there being several.
    """

    def __init__(self, agent: "CandidatePPOAgent", lane: int) -> None:
        self.agent = agent
        self.lane = lane

    def reset(self, initial_state: GameObservation) -> None:
        self.agent.reset(initial_state, lane=self.lane)

    def choose_action(self, state: GameObservation) -> GameAction:
        return self.agent.choose_action(state, lane=self.lane)

    def observe(self, transition: Transition) -> None:
        self.agent.observe(transition, lane=self.lane)

    def finish_episode(self, final_state: GameObservation, truncated: bool) -> None:
        self.agent.finish_episode(final_state, truncated, lane=self.lane)


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
        self._lanes: dict[int, _Lane] = {}
        # One agent serves every client, so the forward pass, the rollout, and
        # the optimizer are shared mutable state.  The game is the bottleneck --
        # each worker spends its time in HTTP, outside this lock -- so
        # serializing the small tensor work costs almost nothing.
        self._lock = threading.RLock()
        self.last_update: dict[str, float] = {}
        self.environment_steps = 0
        self.optimizer_updates = 0
        self._completed_update_metrics: list[dict[str, float]] = []

    def lane_view(self, lane: int) -> LaneView:
        """Return an Agent bound to one environment's lane."""
        if lane < 0:
            raise ValueError("lane must not be negative")
        return LaneView(self, lane)

    def _lane(self, lane: int) -> _Lane:
        with self._lock:
            return self._lanes.setdefault(lane, _Lane())

    def _rollout_length(self) -> int:
        return sum(len(lane.steps) for lane in self._lanes.values())

    def reset(self, initial_state: GameObservation, lane: int = 0) -> None:
        del initial_state
        entry = self._lane(lane)
        entry.pending = None
        entry.forced_action = False
        entry.carried_reward = 0.0

    def choose_action(self, state: GameObservation, lane: int = 0) -> GameAction:
        entry = self._lane(lane)
        if entry.pending is not None:
            raise RuntimeError(
                "observe() must be called before choosing another action"
            )

        candidates = self.action_provider.require_candidates(state.raw_state)
        # A single candidate carries no policy gradient: its log probability is
        # always zero and its ratio always one, so training on it only dilutes
        # the advantage statistics of genuine decisions.
        if len(candidates) == 1:
            entry.forced_action = True
            return candidates[0]

        decision = self.tokenizer.tokenize_decision(state, candidates)
        with self._lock, torch.no_grad():
            output = self.game_encoder.policy_value(decision.to(self.device))
            distribution = Categorical(logits=output.logits)
            if self.training_enabled:
                action_index_tensor = distribution.sample()
            else:
                action_index_tensor = torch.argmax(output.logits)
            log_probability = distribution.log_prob(action_index_tensor)

        action_index = int(action_index_tensor.item())
        if self.training_enabled:
            entry.pending = _PendingDecision(
                decision=decision,
                action_index=action_index,
                log_probability=log_probability.detach().cpu(),
                value=output.value.detach().cpu(),
            )
        return candidates[action_index]

    def observe(self, transition: Transition, lane: int = 0) -> None:
        entry = self._lane(lane)
        forced = entry.forced_action
        entry.forced_action = False
        if not self.training_enabled:
            return

        with self._lock:
            self.environment_steps += 1
            if forced:
                self._absorb_forced_transition(transition, entry)
            else:
                if entry.pending is None:
                    raise RuntimeError(
                        "choose_action() must be called before observe()"
                    )
                entry.steps.append(
                    _RolloutStep(
                        decision=entry.pending.decision,
                        next_observation=transition.next_state,
                        action_index=entry.pending.action_index,
                        old_log_probability=entry.pending.log_probability,
                        old_value=entry.pending.value,
                        reward=float(transition.reward) + entry.carried_reward,
                        done=transition.done,
                    )
                )
                entry.carried_reward = 0.0
                entry.pending = None
            ready = self._rollout_length() >= self.config.rollout_size
        if ready:
            self.update()

    def _absorb_forced_transition(self, transition: Transition, entry: _Lane) -> None:
        """Merge a forced step into the decision it followed.

        Forced steps are never scored, so their reward would otherwise be lost.
        Extending the preceding recorded transition keeps the return of every
        trained decision equal to the return the environment actually paid.
        """
        if not entry.steps or entry.pending is not None:
            entry.carried_reward += float(transition.reward)
            return
        last = entry.steps[-1]
        last.reward += float(transition.reward)
        last.next_observation = transition.next_state
        last.done = transition.done

    def finish_episode(
        self,
        final_state: GameObservation,
        truncated: bool,
        lane: int = 0,
    ) -> None:
        """End an episode without flushing the rollout.

        The rollout deliberately spans episode boundaries: episodes here are
        far shorter than ``rollout_size``, and updating on the dozen
        transitions one episode happens to produce makes both the advantage
        normalization and the gradient almost pure noise.  GAE already zeroes
        the bootstrap and the trace at every terminal, so accumulating across
        episodes changes no return.  The trainer flushes before checkpointing.
        """
        del final_state, truncated
        if self._lane(lane).pending is not None:
            raise RuntimeError("cannot finish an episode with an unobserved action")

    def train(self, enabled: bool = True) -> None:
        """Switch between stochastic learning and deterministic evaluation."""
        if not enabled and any(lane.pending for lane in self._lanes.values()):
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
        """Discard incomplete actions and rollouts without undoing prior updates."""
        with self._lock:
            self._lanes.clear()

    def update(self) -> dict[str, float]:
        """Run PPO updates over the current variable-length rollout.

        Each lane is one uninterrupted trajectory, so GAE is walked backwards
        per lane and only the results are concatenated.  Doing it over the
        concatenation instead would let one environment's advantage flow into
        another's, which nothing would report.
        """
        with self._lock:
            lanes = [self._lanes[key] for key in sorted(self._lanes)]
            lanes = [lane for lane in lanes if lane.steps]
            if not lanes:
                return {}

            steps: list[_RolloutStep] = []
            advantage_chunks: list[Tensor] = []
            return_chunks: list[Tensor] = []
            for lane in lanes:
                lane_advantages, lane_returns = self._advantages_and_returns(lane.steps)
                advantage_chunks.append(lane_advantages)
                return_chunks.append(lane_returns)
                steps.extend(lane.steps)

            advantages = torch.cat(advantage_chunks)
            returns = torch.cat(return_chunks)
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(unbiased=False) + 1e-8
                )

            metrics: dict[str, float] = {}
            for _ in range(self.config.update_epochs):
                order = torch.randperm(len(steps)).tolist()
                for start in range(0, len(order), self.config.minibatch_size):
                    metrics = self._optimize_minibatch(
                        order[start : start + self.config.minibatch_size],
                        steps,
                        advantages,
                        returns,
                    )

            self.optimizer_updates += 1
            metrics["environment_steps"] = float(self.environment_steps)
            metrics["optimizer_update"] = float(self.optimizer_updates)
            metrics["lanes"] = float(len(lanes))
            for lane in lanes:
                lane.steps.clear()
            self.last_update = metrics
            self._completed_update_metrics.append(dict(metrics))
            return metrics

    def _optimize_minibatch(
        self,
        indices: list[int],
        steps: list[_RolloutStep],
        advantages: Tensor,
        returns: Tensor,
    ) -> dict[str, float]:
        """Run one clipped-surrogate optimizer step over a subset of the rollout."""
        policy_losses: list[Tensor] = []
        value_losses: list[Tensor] = []
        entropies: list[Tensor] = []
        for index in indices:
            step = steps[index]
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
            "rollout_steps": float(len(steps)),
        }

    def _advantages_and_returns(
        self, steps: list[_RolloutStep]
    ) -> tuple[Tensor, Tensor]:
        """Return GAE advantages and returns for one lane's trajectory."""
        values = torch.stack([step.old_value for step in steps]).to(self.device)
        advantages = torch.zeros(len(steps), device=self.device)
        with torch.no_grad():
            last_step = steps[-1]
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
            for index in range(len(steps) - 1, -1, -1):
                step = steps[index]
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
        if any(lane.pending for lane in self._lanes.values()):
            raise RuntimeError(
                f"cannot {operation} a checkpoint with an unobserved action"
            )
        if self._rollout_length():
            raise RuntimeError(
                f"cannot {operation} a checkpoint with a non-empty rollout"
            )

    def _move_optimizer_state_to_device(self) -> None:
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if isinstance(value, Tensor):
                    optimizer_state[key] = value.to(self.device)
