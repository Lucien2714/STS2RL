"""PPO over a variable set of complete structured action candidates."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
import math
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
    # Stop an update's epochs once the policy has moved this far (approximate KL
    # against the policy that collected the rollout), as Baselines' target_kl does.
    # The clip limits the ratio only for the actions in the batch, and four epochs
    # of minibatches can still carry the policy far; on a state absent from the
    # batch -- a card reward screen, a few per episode -- the clip limits nothing,
    # and one update took P(skip card reward) from 0.21 to 1.00. None: no limit.
    target_kl: float | None = None

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
        if self.target_kl is not None and not self.target_kl > 0:
            raise ValueError("target_kl must be positive or None")


@dataclass
class _PendingDecision:
    decision: TokenizedDecision
    action_index: int
    log_probability: Tensor
    value: Tensor  # Raw reward units, fixed at sampling time.
    # False for an action another policy chose (see ``choose_external``).
    policy_trainable: bool = True


@dataclass
class _RolloutStep:
    decision: TokenizedDecision
    next_observation: GameObservation
    action_index: int
    old_log_probability: Tensor
    old_value: Tensor  # Raw reward units, independent of later return scales.
    reward: float
    done: bool
    episode_end: bool = False
    bootstrap_value: Tensor | None = None  # Raw value of a truncated final state.
    policy_trainable: bool = True


@dataclass
class _ReturnScale:
    """A running scale for returns, so the critic's target stays O(1).

    Episode returns here grow with the policy: about +/-2 while it dies on
    floor 3, and +57 once it clears a boss.  The critic's loss is squared, so a
    fivefold growth in the target is a twenty-fivefold growth in the loss, and
    the encoder is *shared* with the actor -- a critic gradient spike damages
    the policy's representation without the policy loss ever misbehaving.
    That is what happened: value_loss 0.29 to 5.51 and gradient norm 4.3 to
    19.1 over eight updates, entropy healthy throughout, and evaluated floor
    down from 15.0 to 8.4.

    Only the scale is tracked, not the mean: zero return means "made no
    progress", which is worth keeping at zero.
    """

    var: float = 1.0
    count: float = 1e-4

    def update(self, values: Tensor) -> None:
        """Fold a batch of raw returns into the running second moment."""
        batch = values.detach().to("cpu", torch.float64)
        batch_count = float(batch.numel())
        if batch_count == 0:
            return
        batch_var = float((batch * batch).mean())
        total = self.count + batch_count
        self.var = (self.var * self.count + batch_var * batch_count) / total
        self.count = total

    @property
    def scale(self) -> float:
        """Return the divisor, floored so an all-zero history cannot explode."""
        return max(math.sqrt(self.var), 1e-6)

    def state(self) -> dict[str, float]:
        return {"var": self.var, "count": self.count}

    def load(self, state: Mapping[str, object]) -> None:
        self.var = float(state.get("var", 1.0))
        self.count = float(state.get("count", 1e-4))


@dataclass
class _Lane:
    """One environment's in-flight decision and its slice of the rollout.

    Every client steps its own game, so the decision awaiting an observation
    is per-environment. The rollout is kept per lane as well because GAE walks
    a trajectory backwards:
    interleaving two games into one flat list would make step i-1 the temporal
    predecessor of step i only by accident, and the advantage would propagate
    across environments without anything failing.
    """

    pending: _PendingDecision | None = None
    steps: list[_RolloutStep] = field(default_factory=list)
    # An external action taken without a rollout entry is awaiting its observation;
    # its reward is folded into the decision before it (see ``choose_external``).
    folding: bool = False
    # Reward folded while the lane had no decision of this episode to fold into:
    # paid to the next recorded decision, the one that earned it.
    carried_reward: float = 0.0


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

    def choose_external(
        self, state: GameObservation, action: GameAction, *, record: bool = True
    ) -> GameAction:
        return self.agent.choose_external(state, action, lane=self.lane, record=record)

    def observe(self, transition: Transition) -> None:
        self.agent.observe(transition, lane=self.lane)

    def discard_decision(self) -> None:
        self.agent.discard_decision(lane=self.lane)

    def finish_episode(self, final_state: GameObservation, truncated: bool) -> None:
        self.agent.finish_episode(final_state, truncated, lane=self.lane)


def _policy_step(step: _RolloutStep) -> bool:
    """Whether a step trains the policy: a real choice, made by this policy."""
    return step.policy_trainable and len(step.decision.actions) > 1


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
        self._return_scale = _ReturnScale()
        # One agent serves every client, so the forward pass, the rollout, and
        # the optimizer are shared mutable state.  The game is the bottleneck --
        # each worker spends its time in HTTP, outside this lock -- so
        # serializing the small tensor work costs almost nothing.
        self._lock = threading.RLock()
        # Called right after an update completes, which is the one moment the
        # rollout is empty by construction.  The trainer checkpoints there
        # instead of manufacturing an empty rollout later by force.
        self.on_update: Callable[[], None] | None = None
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
        with self._lock:
            entry = self._lane(lane)
            # Also protect callers that reset without finish_episode(): use
            # the old trajectory's final observation, never the new start.
            if entry.steps:
                self._close_episode(entry, entry.steps[-1].next_observation)
            entry.pending = None
            entry.folding = False
            entry.carried_reward = 0.0

    def choose_action(self, state: GameObservation, lane: int = 0) -> GameAction:
        entry = self._lane(lane)
        if entry.pending is not None:
            raise RuntimeError(
                "observe() must be called before choosing another action"
            )

        candidates = self.action_provider.require_candidates(state.raw_state)
        if len(candidates) == 1 and not self.training_enabled:
            return candidates[0]

        decision = self.tokenizer.tokenize_decision(state, candidates)
        with self._lock, torch.no_grad():
            output = self.game_encoder.policy_value(decision.to(self.device))
            distribution = Categorical(logits=output.logits)
            if self.training_enabled and len(candidates) > 1:
                action_index_tensor = distribution.sample()
            else:
                action_index_tensor = torch.argmax(output.logits)
            log_probability = distribution.log_prob(action_index_tensor)
            action_index = int(action_index_tensor.item())
            if self.training_enabled:
                # Another lane may update while this action is in flight.
                # Capture both the value and its units under the model lock.
                entry.pending = _PendingDecision(
                    decision=decision,
                    action_index=action_index,
                    log_probability=log_probability.detach().cpu(),
                    value=(output.value * self._return_scale.scale).detach().cpu(),
                )
        return candidates[action_index]

    def choose_external(
        self,
        state: GameObservation,
        action: GameAction,
        lane: int = 0,
        *,
        record: bool = True,
    ) -> GameAction:
        """Take an action another policy chose, and learn its value but not its choice.

        The combat search plays fights; PPO still owns everything around them.
        Its steps belong in the trajectory -- GAE has to walk through the fight
        for the reward after it to reach the map choice before it, and the
        critic is what should learn what a deck is worth once the fight is
        played well -- but the action is not a sample from this policy, so it
        has no log probability to form a ratio with and gets no policy
        gradient. It is handled exactly like a forced step: in GAE and the
        critic loss, out of the policy loss and the advantage normalisation.

        ``record=False`` leaves it out of the rollout altogether: the whole fight
        becomes part of the environment's transition from the decision before it
        to the decision after. Its reward is folded into the last recorded
        decision of this lane, and a death in it ends that decision's episode.
        That puts only this policy's own choices in a rollout -- three times as
        many per update -- and shortens the path from a card pick to the boss it
        meets from a hundred steps to a dozen decisions. The critic then values
        only the states between fights, which is what it is for here.
        """
        entry = self._lane(lane)
        if entry.pending is not None:
            raise RuntimeError(
                "observe() must be called before choosing another action"
            )
        candidates = self.action_provider.require_candidates(state.raw_state)
        wanted = action.to_dict()
        matches = [i for i, candidate in enumerate(candidates) if candidate.to_dict() == wanted]
        if not matches:
            raise ValueError(f"{action} is not one of this state's candidates")
        action_index = matches[0]
        if not self.training_enabled:
            return candidates[action_index]
        if not record:
            entry.folding = True
            return candidates[action_index]
        decision = self.tokenizer.tokenize_decision(state, candidates)
        with self._lock:
            entry.pending = _PendingDecision(
                decision=decision,
                action_index=action_index,
                log_probability=torch.zeros(()),
                value=self._raw_value(state),
                policy_trainable=False,
            )
        return candidates[action_index]

    def observe(self, transition: Transition, lane: int = 0) -> None:
        entry = self._lane(lane)
        if not self.training_enabled:
            return

        with self._lock:
            self.environment_steps += 1
            if entry.pending is None and entry.folding:
                self._fold(entry, transition)
                return
            if entry.pending is None:
                raise RuntimeError(
                    "choose_action() must be called before observe()"
                )
            reward = float(transition.reward) + entry.carried_reward
            entry.carried_reward = 0.0
            entry.steps.append(
                _RolloutStep(
                    decision=entry.pending.decision,
                    next_observation=transition.next_state,
                    action_index=entry.pending.action_index,
                    old_log_probability=entry.pending.log_probability,
                    old_value=entry.pending.value,
                    reward=reward,
                    done=transition.done,
                    episode_end=transition.done,
                    policy_trainable=entry.pending.policy_trainable,
                )
            )
            entry.pending = None
            ready = self._rollout_length() >= self.config.rollout_size
        if ready:
            self.update()
            if self.on_update is not None:
                self.on_update()

    def _fold(self, entry: _Lane, transition: Transition) -> None:
        """Merge an unrecorded step into the decision before it.

        Never across an episode boundary: a terminal step is never extended --
        that once erased every terminal in the rollout (CLAUDE.md, "Forced steps
        are folded") -- and its reward goes forward to the next decision instead.
        """
        entry.folding = False
        last = entry.steps[-1] if entry.steps else None
        if last is None or last.episode_end:
            if transition.done:
                # Died before this episode recorded a decision: nothing to credit.
                entry.carried_reward = 0.0
            else:
                entry.carried_reward += float(transition.reward)
            return
        last.reward += float(transition.reward)
        last.next_observation = transition.next_state
        if transition.done:
            last.done = True
            last.episode_end = True

    def _raw_value(self, state: GameObservation) -> Tensor:
        """Read one bootstrap in reward units while the caller holds the lock."""
        with torch.no_grad():
            value = self.game_encoder.value(
                self.tokenizer.tokenize_state(state).to(self.device)
            )
            return (value * self._return_scale.scale).detach().cpu()

    def _close_episode(self, entry: _Lane, final_state: GameObservation) -> None:
        """Cut the trace at a nonterminal boundary, retaining its bootstrap."""
        if not entry.steps or entry.steps[-1].episode_end or entry.steps[-1].done:
            return
        last = entry.steps[-1]
        last.next_observation = final_state
        last.bootstrap_value = self._raw_value(final_state)
        last.episode_end = True

    def finish_episode(
        self,
        final_state: GameObservation,
        truncated: bool,
        lane: int = 0,
    ) -> None:
        """End an episode without flushing the rollout.

        Terminal transitions already carry a zero bootstrap. A truncation
        keeps V(final_state), but cuts the trace so the next episode cannot
        contribute rewards. Neither boundary forces a small PPO update.
        """
        with self._lock:
            entry = self._lane(lane)
            if entry.pending is not None:
                raise RuntimeError("cannot finish an episode with an unobserved action")
            if truncated:
                self._close_episode(entry, final_state)

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
        """Return a detached copy of the model and optimizer tensors.

        Saving is a read-only snapshot, so it does not require an idle agent:
        a rollout in progress is not written and simply carries on, and another
        client's undelivered decision changes neither the weights nor the
        counters.  Demanding an empty rollout is what used to force a PPO
        update on whatever happened to be collected -- 13 transitions in one
        measured case -- and then save the weights that update had just moved.

        The copy is taken under the lock because another client can be running
        an optimizer step, and serialising live tensors would write a mixture
        of before and after.  Lifetime counters belong to ``TrainingState``,
        which stores them once.
        """
        with self._lock:
            return {
                "encoder": {
                    name: tensor.detach().clone()
                    for name, tensor in self.game_encoder.state_dict().items()
                },
                "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                # Without this a resumed critic reads its own predictions at
                # the wrong scale, which is worse than not saving it at all.
                "return_scale": self._return_scale.state(),
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
        return_scale = state.get("return_scale")
        if isinstance(return_scale, Mapping):
            self._return_scale.load(return_scale)
        self._move_optimizer_state_to_device()
        self.last_update = {}
        self._completed_update_metrics.clear()

    def initialize_from(
        self,
        encoder_state: Mapping[str, Tensor],
        return_scale: Mapping[str, object],
        optimizer_state: Mapping[str, object] | None = None,
    ) -> None:
        """Start from another run's model: its weights and return scale, and optionally
        its optimizer's moments.

        The return scale comes with the critic it calibrates -- a critic reading its
        own predictions at the wrong scale is worse than an untrained one.

        The optimizer is fresh unless ``optimizer_state`` is given. A fresh Adam moves
        every weight by about the full learning rate in its first steps, and measured
        after two restarts the card choices moved three times as far in 18 updates as
        in a settled run, and each run lost about two floors for its first 100+
        episodes. The moments are carried only onto the same parameters in the same
        shapes; this run's learning rate replaces the old one.
        """
        self._require_clean_checkpoint_boundary("initialize")
        self.game_encoder.load_state_dict(dict(encoder_state))
        self._return_scale.load(return_scale)
        if optimizer_state is not None:
            self._load_optimizer_moments(optimizer_state)

    def _load_optimizer_moments(self, state: Mapping[str, object]) -> None:
        parameters = list(self.game_encoder.parameters())
        groups = state.get("param_groups")
        moments = state.get("state")
        if (
            not isinstance(groups, list)
            or len(groups) != 1
            or len(groups[0].get("params", ())) != len(parameters)
            or not isinstance(moments, Mapping)
        ):
            raise ValueError("optimizer state does not belong to this model's parameters")
        for index, entry in moments.items():
            for name in ("exp_avg", "exp_avg_sq"):
                tensor = entry.get(name) if isinstance(entry, Mapping) else None
                if not isinstance(tensor, Tensor) or tensor.shape != parameters[int(index)].shape:
                    raise ValueError(
                        f"optimizer moment {name} of parameter {index} does not match its shape"
                    )
        self.optimizer.load_state_dict(dict(state))
        for group in self.optimizer.param_groups:
            group["lr"] = self.config.learning_rate
        self._move_optimizer_state_to_device()

    def drain_update_metrics(self) -> tuple[dict[str, float], ...]:
        """Return completed PPO update metrics once, in completion order."""
        metrics = tuple(dict(item) for item in self._completed_update_metrics)
        self._completed_update_metrics.clear()
        return metrics

    def discard_decision(self, lane: int = 0) -> None:
        """Forget one lane's chosen action, keeping everything it has learned.

        A refusal that leaves the screen unchanged is not a transition: the
        action was legal and the screen was simply not ready.  The decision is
        dropped so the lane can choose again, while the rollout it has already
        collected is untouched.
        """
        with self._lock:
            self._lane(lane).pending = None
            self._lane(lane).folding = False

    def abort_lane(self, lane: int) -> None:
        """Discard one environment's whole trajectory after its episode failed.

        The steps leading up to a crash are the least trustworthy in the
        rollout, not the most: a client on its way out serves stale or partial
        state, and rewards are differences between states -- a floor that
        failed to update invents progress that never happened.  Keeping them to
        save the handful of samples a rare crash costs trades data quality for
        almost no data.

        Only this lane is cleared.  ``abort_episode`` clears every lane and
        would throw away the other clients' work.
        """
        with self._lock:
            entry = self._lane(lane)
            entry.pending = None
            entry.steps.clear()
            entry.folding = False
            entry.carried_reward = 0.0

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
            # Learn the scale from the returns this batch actually saw, then
            # give the critic a target that does not grow with the policy.
            self._return_scale.update(returns)
            returns = returns / self._return_scale.scale
            policy_mask = torch.tensor(
                [_policy_step(step) for step in steps],
                dtype=torch.bool,
                device=self.device,
            )
            policy_advantages = advantages[policy_mask]
            if len(policy_advantages) > 1:
                policy_advantages = (policy_advantages - policy_advantages.mean()) / (
                    policy_advantages.std(unbiased=False) + 1e-8
                )
            # Forced and external actions participate in per-environment-step
            # GAE and in critic training, but not in the policy's normalization
            # or loss.
            advantages = torch.zeros_like(advantages)
            advantages[policy_mask] = policy_advantages

            metrics: dict[str, float] = {}
            optimizer_steps = 0
            stopped = False
            for _ in range(self.config.update_epochs):
                order = torch.randperm(len(steps)).tolist()
                for start in range(0, len(order), self.config.minibatch_size):
                    result = self._optimize_minibatch(
                        order[start : start + self.config.minibatch_size],
                        steps,
                        advantages,
                        returns,
                    )
                    if result.get("kl_stopped"):
                        stopped = True
                        metrics["approx_kl"] = result["approx_kl"]
                        break
                    metrics = result
                    optimizer_steps += 1
                if stopped:
                    break
            metrics["optimizer_steps"] = float(optimizer_steps)
            metrics["kl_early_stop"] = float(stopped)

            self.optimizer_updates += 1
            metrics["environment_steps"] = float(self.environment_steps)
            metrics["optimizer_update"] = float(self.optimizer_updates)
            metrics["lanes"] = float(len(lanes))
            metrics["return_scale"] = self._return_scale.scale
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
        kl_terms: list[Tensor] = []
        for index in indices:
            step = steps[index]
            if not _policy_step(step):
                value = self.game_encoder.value(step.decision.state.to(self.device))
                value_losses.append(F.mse_loss(value, returns[index]))
                continue
            output = self.game_encoder.policy_value(step.decision.to(self.device))
            distribution = Categorical(logits=output.logits)
            action_index = torch.tensor(step.action_index, device=self.device)
            new_log_probability = distribution.log_prob(action_index)
            log_ratio = new_log_probability - step.old_log_probability.to(self.device)
            ratio = torch.exp(log_ratio)
            # The low-variance estimator of KL(old || new): (r - 1) - log r >= 0.
            kl_terms.append(((ratio - 1.0) - log_ratio).detach())
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

        zero = torch.zeros((), device=self.device)
        approx_kl = float(torch.stack(kl_terms).mean().cpu()) if kl_terms else 0.0
        if self.config.target_kl is not None and approx_kl > 1.5 * self.config.target_kl:
            # Measured before stepping, so the step that would cross the limit is
            # never taken (Stable-Baselines3 uses the same 1.5 margin).
            return {"kl_stopped": 1.0, "approx_kl": approx_kl}
        policy_loss = torch.stack(policy_losses).mean() if policy_losses else zero
        value_loss = torch.stack(value_losses).mean()
        entropy = torch.stack(entropies).mean() if entropies else zero
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
            "approx_kl": approx_kl,
        }

    def _advantages_and_returns(
        self, steps: list[_RolloutStep]
    ) -> tuple[Tensor, Tensor]:
        """Return GAE advantages and returns for one lane's trajectory."""
        # Sampling values are already in reward units. Pending decisions can
        # survive another lane's update; never reinterpret them at a new scale.
        values = torch.stack([step.old_value for step in steps]).to(self.device)
        advantages = torch.zeros(len(steps), device=self.device)
        with torch.no_grad():
            gae = torch.zeros((), device=self.device)
            for index in range(len(steps) - 1, -1, -1):
                step = steps[index]
                boundary = step.done or step.episode_end or index == len(steps) - 1
                if step.done:
                    next_value = torch.zeros((), device=self.device)
                elif boundary:
                    bootstrap = step.bootstrap_value
                    if bootstrap is None:
                        bootstrap = self._raw_value(step.next_observation)
                    next_value = bootstrap.to(self.device)
                else:
                    next_value = values[index + 1]
                delta = step.reward + self.config.gamma * next_value - values[index]
                trace_mask = 0.0 if boundary else 1.0
                gae = (
                    delta
                    + self.config.gamma * self.config.gae_lambda * trace_mask * gae
                )
                advantages[index] = gae
        return advantages, advantages + values

    def _require_clean_checkpoint_boundary(self, operation: str) -> None:
        """Refuse to overwrite the weights an in-flight decision was made under."""
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
