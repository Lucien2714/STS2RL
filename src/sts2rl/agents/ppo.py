"""PPO over a variable set of complete structured action candidates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
import math
from dataclasses import dataclass, field, replace
import threading

import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F

from sts2rl.actions import GameAction
from sts2rl.agents.action_space import LegalActionProvider
from sts2rl.agents.base import Agent, Transition, without_excluded
from sts2rl.encoder import GameEncoder, GameTokenizer, GameVocabulary, TokenizedDecision
from sts2rl.env.types import GameObservation


# Every screen the game reports, which is all a per-screen setting can name.
STATE_TYPES = GameVocabulary.FIXED_TOKENS["state_types"]


def _screen_pairs(
    value: object,
    name: str,
    quantity: str,
    *,
    within: Callable[[float], bool],
    bounds: str,
) -> tuple[tuple[str, float], ...]:
    """Normalise a per-screen mapping into sorted ``(state_type, number)`` pairs.

    A mapping, a sequence of pairs, and the JSON reading of either arrive here,
    so a plan round-trips through ``asdict`` and ``from_dict`` unchanged and
    two spellings of one mapping compare equal. ``within`` is the range a
    finite number must fall in, and ``bounds`` is how the error names it.
    """
    items = value.items() if isinstance(value, Mapping) else value
    numbers: dict[str, float] = {}
    for item in items:
        try:
            state_type, number = item
        except (TypeError, ValueError):
            raise ValueError(f"{name} must map state types to {quantity}s") from None
        if state_type not in STATE_TYPES:
            raise ValueError(f"{name} names an unknown state type: {state_type!r}")
        if state_type in numbers:
            raise ValueError(f"{name} names {state_type!r} twice")
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise TypeError(f"{name} {quantity} for {state_type!r} must be a number")
        if not math.isfinite(number) or not within(number):
            raise ValueError(f"{name} {quantity} for {state_type!r} must be {bounds}")
        numbers[state_type] = float(number)
    return tuple(sorted(numbers.items()))


def _exploration_pairs(value: object) -> tuple[tuple[str, float], ...]:
    """Sorted ``(state_type, epsilon)`` pairs: rates at least 0 and below 1."""
    return _screen_pairs(
        value,
        "exploration",
        "rate",
        within=lambda epsilon: 0 <= epsilon < 1,
        bounds="at least 0 and below 1",
    )


def _reference_pairs(value: object) -> tuple[tuple[str, float], ...]:
    """Sorted ``(state_type, beta)`` pairs: coefficients that are 0 or more."""
    return _screen_pairs(
        value,
        "reference_kl_screens",
        "coefficient",
        within=lambda beta: beta >= 0,
        bounds="a finite number, 0 or more",
    )


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
    # It bounds only the estimate from the sampled actions. The reference term
    # (reference_kl_coefficient) moves the policy on candidates that were never
    # sampled, which this estimator cannot see, so it is not what limits the
    # pull; the coefficient is.
    target_kl: float | None = None
    # Targeted exploration, as sorted (state_type, epsilon) pairs. On a screen
    # of that type a real choice is drawn from the mixture
    #     mu(a) = (1 - epsilon) * pi(a) + epsilon / N
    # over its N candidates, and the mixture is the policy PPO trains there:
    # the stored log probability and the ratio are both the mixture's, so an
    # update still starts at ratio 1 and the clip and target_kl keep their
    # meaning. The entropy bonus cannot do this job: it is one coefficient
    # for every screen, and the screens that collapse -- a rest site or a card
    # reward, a few decisions per episode -- are not the ones it is tuned on.
    # A mapping is accepted and normalised, so order does not affect equality
    # and a plan round-trips through JSON exactly. Empty: the plain policy,
    # drawing nothing extra.
    exploration: tuple[tuple[str, float], ...] = ()
    # Forward KL toward a frozen reference policy -- a human-cloned actor --
    # added to the loss as beta * KL(pi_ref || pi_theta), averaged over the
    # sampled decisions of each minibatch. Its gradient on the logits is
    # pi_theta - pi_ref: nonzero wherever the reference puts mass, which is
    # exactly what the surrogate never gives an option the policy has driven
    # to ~0 -- upgrading at a rest site sat at 0.01-0.04 against a human 0.65,
    # skipping a card reward at 0.00 against 0.52, and 54 updates of
    # --explore sampled them without moving them. The term is weighted by
    # neither the advantage nor the ratio, and enters neither the clip nor the
    # KL stop: it is a pull toward the reference, not a reward. 0: no reference
    # is built, and the path is exactly the one without it.
    reference_kl_coefficient: float = 0.0
    # The coefficient per screen, as sorted (state_type, beta) pairs, each
    # replacing reference_kl_coefficient on its screen: 0 switches the
    # reference off there whatever the global coefficient, and a positive one
    # switches it on there alone. The reference is a human clone, and it is
    # trustworthy on some screens and not others -- upgrading at high HP at a
    # rest site and taking elites on the map, against 65% accuracy on card
    # rewards and 40% in shops -- so one coefficient everywhere revived the
    # collapsed options within 12 updates (upgrade 0.01 to 0.16, skip 0.00 to
    # 0.66) and also pushed skipping a card reward past the reference's 0.62
    # and the humans' 0.52, while the floor fell from 26.5 to 18.4 on the same
    # seeds. Normalised like ``exploration``, so order and spelling do not
    # count. Empty: reference_kl_coefficient on every screen, which is exactly
    # the path from before this existed.
    reference_kl_screens: tuple[tuple[str, float], ...] = ()

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
        object.__setattr__(self, "exploration", _exploration_pairs(self.exploration))
        beta = self.reference_kl_coefficient
        if (
            isinstance(beta, bool)
            or not isinstance(beta, (int, float))
            or not math.isfinite(beta)
            or beta < 0
        ):
            raise ValueError("reference_kl_coefficient must be a finite number, 0 or more")
        object.__setattr__(self, "reference_kl_coefficient", float(beta))
        object.__setattr__(
            self, "reference_kl_screens", _reference_pairs(self.reference_kl_screens)
        )

    def exploration_rate(self, state_type: str) -> float:
        """Return the epsilon in force on a screen, 0 where none is configured."""
        for screen, epsilon in self.exploration:
            if screen == state_type:
                return epsilon
        return 0.0

    def reference_kl_for(self, state_type: str) -> float:
        """Return the reference coefficient in force on a screen: its own, else the global one."""
        for screen, beta in self.reference_kl_screens:
            if screen == state_type:
                return beta
        return self.reference_kl_coefficient

    def reference_screens(self) -> tuple[str, ...]:
        """The screens the reference term is in force on, in ``STATE_TYPES`` order."""
        return tuple(screen for screen in STATE_TYPES if self.reference_kl_for(screen) > 0)

    @property
    def uses_reference(self) -> bool:
        """Whether any screen pulls toward the reference, which the agent then needs."""
        return bool(self.reference_screens())

    def without_reference(self) -> PPOConfig:
        """This configuration for an agent that only plays a checkpoint's actor.

        A script that samples or argmaxes a saved policy and never updates --
        capturing snapshots, playing boss fights -- has no use for the artifact
        its run pulled toward and must not be made to find it: every
        coefficient at 0 asks for no reference. Everything else is kept, so
        the exploration rates and the rollout such an agent collects are
        still the run's.
        """
        return replace(self, reference_kl_coefficient=0.0, reference_kl_screens=())


@dataclass
class _PendingDecision:
    decision: TokenizedDecision
    action_index: int
    # log mu_old(a): the mixture's where the screen explores, else log pi_old(a).
    log_probability: Tensor
    # log pi_old(a), the policy's own, kept apart for the policy_kl metric.
    policy_log_probability: Tensor
    value: Tensor  # Raw reward units, fixed at sampling time.
    state_type: str
    # False for an action another policy chose (see ``choose_external``).
    policy_trainable: bool = True
    # The rate the mixture was sampled at -- 0 where nothing was mixed in: a
    # screen without a rate, a forced step, an external action -- and whether
    # it was the uniform branch that drew this action.
    epsilon: float = 0.0
    explored: bool = False
    # log pi_ref(.|s) over these same candidates, for the reference KL. None
    # where the term does not apply: a forced or external step, a screen
    # whose coefficient is 0, or no reference at all.
    reference_log_probabilities: Tensor | None = None


@dataclass
class _RolloutStep:
    decision: TokenizedDecision
    next_observation: GameObservation
    action_index: int
    old_log_probability: Tensor  # The mixture's where the screen explored.
    old_policy_log_probability: Tensor
    old_value: Tensor  # Raw reward units, independent of later return scales.
    reward: float
    done: bool
    state_type: str
    episode_end: bool = False
    bootstrap_value: Tensor | None = None  # Raw value of a truncated final state.
    policy_trainable: bool = True
    epsilon: float = 0.0
    explored: bool = False
    reference_log_probabilities: Tensor | None = None


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

    def choose_action(
        self, state: GameObservation, exclude: Sequence[GameAction] = ()
    ) -> GameAction:
        return self.agent.choose_action(state, lane=self.lane, exclude=exclude)

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


def _mixture_log_probability(log_probability: Tensor, epsilon: float, count: int) -> Tensor:
    """Return log((1 - epsilon) * pi(a) + epsilon / count) from log pi(a).

    Formed in log space: pi(a) can underflow for an action the policy has
    all but abandoned -- the very action exploration exists to revisit -- and
    the log of that underflow is -inf where the mixture's is finite.
    """
    return torch.logaddexp(
        log_probability + math.log1p(-epsilon),
        # As a difference: the quotient itself underflows for the smallest rates.
        torch.full_like(log_probability, math.log(epsilon) - math.log(count)),
    )


class CandidatePPOAgent(Agent):
    """PPO agent trained end-to-end over structured dynamic candidates.

    ``hold_open_steps``: unrecorded steps (``choose_external(record=False)``)
    can follow any lane's last recorded step, so an update trains only the
    transitions that are complete and keeps each lane's open last step for the
    next one (``_holds_open_step``). The first unrecorded step turns it on;
    passing it at construction also covers the first fight of each lane.

    ``reference_encoder``: the frozen policy the reference term pulls toward
    (``PPOConfig.reference_kl_coefficient``, per screen
    ``PPOConfig.reference_kl_screens``). Its parameters are its own -- never
    the trained encoder's, never the optimizer's -- it is read only under
    ``no_grad``, and it stays in eval mode whatever ``train`` does to the
    actor, so an update cannot move it. Needed only to *train* with a positive
    coefficient on some screen: evaluation builds the agent from a saved plan,
    never samples, and has no use for the artifact.
    """

    def __init__(
        self,
        tokenizer: GameTokenizer,
        game_encoder: GameEncoder,
        action_provider: LegalActionProvider | None = None,
        config: PPOConfig | None = None,
        device: str | torch.device | None = None,
        *,
        hold_open_steps: bool = False,
        reference_encoder: GameEncoder | None = None,
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
        if reference_encoder is not None:
            if not self.config.uses_reference:
                raise ValueError(
                    "a reference policy needs a positive reference_kl_coefficient, "
                    "globally or on a screen in reference_kl_screens"
                )
            # Memory, not identity: nn.Parameter(actor_parameter.detach()) is
            # a new object on the actor's storage, and every optimizer step
            # would move the "frozen" reference with it.
            trained = {
                parameter.untyped_storage().data_ptr()
                for parameter in self.game_encoder.parameters()
                if parameter.numel()
            }
            if any(
                parameter.numel() and parameter.untyped_storage().data_ptr() in trained
                for parameter in reference_encoder.parameters()
            ):
                raise ValueError(
                    "the reference policy must not share memory with the trained "
                    "encoder: a parameter on the actor's storage moves with every update"
                )
            reference_encoder.to(self.device)
            reference_encoder.eval()
            reference_encoder.requires_grad_(False)
        self.reference_encoder = reference_encoder
        self.training_enabled = True
        self._lanes: dict[int, _Lane] = {}
        self._return_scale = _ReturnScale()
        # One agent serves every client, so the forward pass, the rollout, and
        # the optimizer are shared mutable state.  The game is the bottleneck --
        # each worker spends its time in HTTP, outside this lock -- so
        # serializing the small tensor work costs almost nothing.
        self._lock = threading.RLock()
        # Called right after an update completes, which is the one moment the
        # rollout holds no completed transition by construction (at most one open
        # step per lane, see ``_holds_open_step``).  The trainer checkpoints there
        # instead of manufacturing an empty rollout later by force.
        self.on_update: Callable[[], None] | None = None
        self.hold_open_steps = hold_open_steps
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
        """Every recorded step, held ones included: all of it is work in flight."""
        return sum(len(lane.steps) for lane in self._lanes.values())

    def _holds_open_step(self, entry: _Lane) -> bool:
        """Whether this lane's last step can still receive folded reward.

        With fights out of the rollout, a fight's reward and its death are folded
        into the decision before it (``_fold``). An update that trains that
        decision while the fight is still running trains it without the fight's
        floors, boss reward and terminal; the clear then leaves ``_fold`` nothing
        to fold into, so the rest of the fight is paid to the lane's next
        decision, and a death is dropped. With seven lanes an update almost
        always lands while other lanes are mid-fight: in runs/step2n-train two
        thirds of the environment steps were unrecorded fight steps.

        Only two things complete the step: the end of its episode, or an
        observed successor step, which makes it no longer the last. A pending
        decision does not: a refused action is discarded (``discard_decision``)
        and the lane can then fold a fight into this step after all. Anything
        else is held: the ``folding`` flag alone is not enough, because it is
        false while the search chooses the fight's next action and before the
        fight's first action, which is most of the time a fight takes.
        """
        if not self.hold_open_steps or not entry.steps:
            return False
        last = entry.steps[-1]
        return not (last.done or last.episode_end)

    def _trainable_length(self) -> int:
        """Completed transitions: the rollout minus each lane's held open step."""
        return self._rollout_length() - sum(
            self._holds_open_step(lane) for lane in self._lanes.values()
        )

    def _update_if_ready(self) -> None:
        """Update once ``rollout_size`` transitions are complete, then call the hook.

        The threshold is checked under the same lock acquisition that selects
        the steps. Between a check in ``observe`` and the update, another lane
        can observe or discard: two lanes that both saw a full rollout would
        otherwise run a second, undersized update straight after the first.
        """
        with self._lock:
            if self._trainable_length() < self.config.rollout_size:
                return
            metrics = self.update()
        if metrics and self.on_update is not None:
            self.on_update()

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
        # Closing a held step can complete the rollout.
        self._update_if_ready()

    def choose_action(
        self,
        state: GameObservation,
        lane: int = 0,
        exclude: Sequence[GameAction] = (),
    ) -> GameAction:
        """Sample (or, in evaluation, take the best of) this state's candidates.

        ``exclude`` removes actions the game refused on this unchanged screen
        before anything is scored, and the decision is recorded over that
        reduced set: the stored log probability is then a probability over what
        was really offered, and the update re-encodes the same reduced set, so
        the ratio compares like with like.  Recording the full set instead
        would store a sample the behaviour policy could not draw, under a
        probability renormalised over actions it never saw.  A set reduced to
        one candidate is a forced step like any other (``_policy_step``).

        A screen with an exploration rate (``PPOConfig.exploration``) is drawn
        from the mixture of the policy and the uniform choice over that same
        set, and the stored log probability is the mixture's: the update
        trains the mixture too, so the ratio compares like with like there as
        well.

        With a reference policy, log pi_ref over that same set is stored
        beside the sample on every screen whose coefficient
        (``PPOConfig.reference_kl_for``) is positive: the update re-encodes
        the stored decision and the reference KL compares the two over
        identical candidates. The reference is read here rather than in the
        update because the reference encoder never changes, so its one forward
        pass per decision is final. A screen whose coefficient is 0 reads
        nothing and stores None, and the update asks nothing of such a step.
        """
        entry = self._lane(lane)
        if entry.pending is not None:
            raise RuntimeError(
                "observe() must be called before choosing another action"
            )

        candidates = self.action_provider.require_candidates(state.raw_state)
        if exclude:
            candidates = without_excluded(candidates, exclude)
        if len(candidates) == 1 and not self.training_enabled:
            return candidates[0]

        state_type = str(state.raw_state.get("state_type"))
        sampling = self.training_enabled and len(candidates) > 1
        # The rate in force: only a real choice, made while learning, is mixed.
        # Evaluation and a forced step read no random number for it.
        epsilon = self.config.exploration_rate(state_type) if sampling else 0.0
        reference = self._reference(state_type) if sampling else None
        decision = self.tokenizer.tokenize_decision(state, candidates)
        with self._lock, torch.no_grad():
            on_device = decision.to(self.device)
            output = self.game_encoder.policy_value(on_device)
            reference_log_probabilities = None
            if reference is not None:
                # float32 whatever the reference computes in: the KL multiplies
                # these by their own exponential, and a half-precision log of a
                # vanishing probability would not survive that.
                reference_log_probabilities = F.log_softmax(
                    reference.policy_value(on_device).logits.float(), dim=-1
                ).detach().cpu().clone()
            distribution = Categorical(logits=output.logits)
            explored = False
            if epsilon > 0.0 and float(torch.rand(())) < epsilon:
                # The mixture's uniform branch, drawn explicitly so the step
                # can say which branch chose it.
                action_index_tensor = torch.randint(
                    len(candidates), (), device=self.device
                )
                explored = True
            elif sampling:
                action_index_tensor = distribution.sample()
            else:
                action_index_tensor = torch.argmax(output.logits)
            policy_log_probability = distribution.log_prob(action_index_tensor)
            # What was sampled is the mixture's marginal, whichever branch drew
            # the action, so that is what the ratio starts from.
            log_probability = (
                _mixture_log_probability(policy_log_probability, epsilon, len(candidates))
                if epsilon > 0.0
                else policy_log_probability
            )
            action_index = int(action_index_tensor.item())
            if self.training_enabled:
                # Another lane may update while this action is in flight.
                # Capture both the value and its units under the model lock.
                entry.pending = _PendingDecision(
                    decision=decision,
                    action_index=action_index,
                    log_probability=log_probability.detach().cpu(),
                    policy_log_probability=policy_log_probability.detach().cpu(),
                    value=(output.value * self._return_scale.scale).detach().cpu(),
                    state_type=state_type,
                    epsilon=epsilon,
                    explored=explored,
                    reference_log_probabilities=reference_log_probabilities,
                )
        return candidates[action_index]

    def _reference(self, state_type: str) -> GameEncoder | None:
        """Return the reference encoder if this screen's coefficient calls for one.

        Checked here, on the first sampled decision of such a screen, rather
        than at construction: evaluation builds the agent from a saved plan
        whose coefficient is positive and never samples, so the artifact need
        not be there. Training without it would store no reference
        distribution and the update would have nothing to pull toward, which
        is worth failing before a rollout is collected.
        """
        if not self.config.reference_kl_for(state_type) > 0:
            return None
        if self.reference_encoder is None:
            raise RuntimeError(
                "training with a reference KL needs a reference policy: pass "
                "reference_encoder, or set every reference KL coefficient to 0"
            )
        return self.reference_encoder

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
            with self._lock:
                entry.folding = True
                self.hold_open_steps = True
            return candidates[action_index]
        decision = self.tokenizer.tokenize_decision(state, candidates)
        with self._lock:
            entry.pending = _PendingDecision(
                decision=decision,
                action_index=action_index,
                log_probability=torch.zeros(()),
                policy_log_probability=torch.zeros(()),
                value=self._raw_value(state),
                state_type=str(state.raw_state.get("state_type")),
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
                # A death folded here completes the held step, which can fill
                # the rollout.
                self._fold(entry, transition)
            elif entry.pending is None:
                raise RuntimeError(
                    "choose_action() must be called before observe()"
                )
            else:
                reward = float(transition.reward) + entry.carried_reward
                entry.carried_reward = 0.0
                entry.steps.append(
                    _RolloutStep(
                        decision=entry.pending.decision,
                        next_observation=transition.next_state,
                        action_index=entry.pending.action_index,
                        old_log_probability=entry.pending.log_probability,
                        old_policy_log_probability=entry.pending.policy_log_probability,
                        old_value=entry.pending.value,
                        reward=reward,
                        done=transition.done,
                        state_type=entry.pending.state_type,
                        episode_end=transition.done,
                        policy_trainable=entry.pending.policy_trainable,
                        epsilon=entry.pending.epsilon,
                        explored=entry.pending.explored,
                        reference_log_probabilities=entry.pending.reference_log_probabilities,
                    )
                )
                entry.pending = None
            # Count only completed transitions. A held step counted here would
            # call for an update with nothing more to train, on every step.
            ready = self._trainable_length() >= self.config.rollout_size
        if ready:
            self._update_if_ready()

    def _fold(self, entry: _Lane, transition: Transition) -> None:
        """Merge an unrecorded step into the decision before it.

        Never across an episode boundary: a terminal step is never extended --
        that once erased every terminal in the rollout (CLAUDE.md, "Forced steps
        are folded") -- and its reward goes forward to the next decision instead.

        An update never takes the step folded into here (``_holds_open_step``),
        so a missing step means the episode truly has no decision yet.
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
        # A truncation closes a held step, which can complete the rollout; if
        # training ends here, the full batch must not wait for an episode that
        # never comes.
        self._update_if_ready()

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

        A lane's open last step (``_holds_open_step``) is not trained here and
        stays in the lane for the next update, where it arrives with the fight's
        reward and terminal. The step before it bootstraps from the held step's
        own sampled value, which is exactly the ``values[index + 1]`` the walk
        would use if the held step were in the batch; only the trace is cut
        there, because the held step's reward is not known yet.
        """
        with self._lock:
            lanes: list[_Lane] = []
            trained_counts: list[int] = []
            steps: list[_RolloutStep] = []
            advantage_chunks: list[Tensor] = []
            return_chunks: list[Tensor] = []
            held_steps = 0
            for key in sorted(self._lanes):
                lane = self._lanes[key]
                held = self._holds_open_step(lane)
                trained = lane.steps[:-1] if held else lane.steps
                held_steps += held
                if not trained:
                    continue
                lane_advantages, lane_returns = self._advantages_and_returns(
                    trained, tail_value=lane.steps[-1].old_value if held else None
                )
                advantage_chunks.append(lane_advantages)
                return_chunks.append(lane_returns)
                steps.extend(trained)
                lanes.append(lane)
                trained_counts.append(len(trained))
            if not lanes:
                return {}

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
            # The reference KL of every minibatch that stepped, weighted by the
            # steps it was measured on; ``reference_kl`` alone is the last one's.
            reference_kl_total = 0.0
            reference_steps = 0.0
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
                        metrics["policy_kl"] = result["policy_kl"]
                        break
                    metrics = result
                    optimizer_steps += 1
                    if "reference_steps" in result:
                        reference_kl_total += result["reference_kl"] * result["reference_steps"]
                        reference_steps += result["reference_steps"]
                if stopped:
                    break
            # Here, not only in ``_optimize_minibatch``: a KL stop on the first
            # minibatch returns no minibatch metrics at all.
            metrics["rollout_steps"] = float(len(steps))
            metrics["optimizer_steps"] = float(optimizer_steps)
            metrics["kl_early_stop"] = float(stopped)
            if self.config.uses_reference:
                # Weighted over the minibatches that stepped, by the reference
                # steps each was measured on. 0.0 after a stop on the first
                # minibatch means nothing was measured, not that the policy is
                # near the reference. ``reference_kl`` and ``reference_steps``
                # are the last stepped minibatch's own: that count is not the
                # denominator of this mean.
                metrics["reference_kl_mean"] = (
                    reference_kl_total / reference_steps if reference_steps else 0.0
                )

            self.optimizer_updates += 1
            metrics["environment_steps"] = float(self.environment_steps)
            metrics["optimizer_update"] = float(self.optimizer_updates)
            metrics["lanes"] = float(len(lanes))
            metrics["return_scale"] = self._return_scale.scale
            metrics["held_steps"] = float(held_steps)
            metrics.update(self._exploration_metrics(steps, advantages))
            for lane, count in zip(lanes, trained_counts):
                del lane.steps[:count]
            self.last_update = metrics
            self._completed_update_metrics.append(dict(metrics))
            return metrics

    def _exploration_metrics(
        self, steps: list[_RolloutStep], advantages: Tensor
    ) -> dict[str, float]:
        """Count the trained steps the uniform branch drew, and how they scored.

        ``explored_positive_advantage`` is the share of those steps whose
        normalised advantage is positive -- the sign the policy loss sees for
        them, after the batch's normalisation -- and 0 when none was drawn.
        """
        explored = [step.explored for step in steps]
        count = sum(explored)
        metrics = {"explored_steps": float(count)}
        for screen, _ in self.config.exploration:
            metrics[f"explore/{screen}"] = float(
                sum(1 for step in steps if step.explored and step.state_type == screen)
            )
        positive = 0.0
        if count:
            mask = torch.tensor(explored, dtype=torch.bool, device=advantages.device)
            positive = float((advantages[mask] > 0).sum()) / count
        metrics["explored_positive_advantage"] = positive
        return metrics

    def _optimize_minibatch(
        self,
        indices: list[int],
        steps: list[_RolloutStep],
        advantages: Tensor,
        returns: Tensor,
    ) -> dict[str, float]:
        """Run one clipped-surrogate optimizer step over a subset of the rollout.

        The reference KL is formed on every policy step of a screen with a
        positive coefficient, from the same re-encoded logits the surrogate
        uses, on the policy itself -- an explored screen trains the mixture's
        ratio, but the pull is toward what the policy should put on each
        candidate, not what the mixture does. Each step's KL is added to the
        loss on its own screen's coefficient, averaged over the minibatch's
        policy steps -- a step whose screen has no coefficient is one of
        those steps and contributes nothing; the surrogate, the advantage
        normalisation and the KL stop never see any of it.
        """
        uses_reference = self.config.uses_reference
        policy_losses: list[Tensor] = []
        value_losses: list[Tensor] = []
        entropies: list[Tensor] = []
        kl_terms: list[Tensor] = []
        policy_kl_terms: list[Tensor] = []
        # The KL of every reference step: unweighted for the metrics, on its
        # coefficient for the loss, and by screen for the per-screen metrics.
        reference_kl_terms: list[Tensor] = []
        weighted_reference_kl_terms: list[Tensor] = []
        reference_kl_by_screen: dict[str, list[Tensor]] = {}
        for index in indices:
            step = steps[index]
            if not _policy_step(step):
                value = self.game_encoder.value(step.decision.state.to(self.device))
                value_losses.append(F.mse_loss(value, returns[index]))
                continue
            output = self.game_encoder.policy_value(step.decision.to(self.device))
            distribution = Categorical(logits=output.logits)
            beta = self.config.reference_kl_for(step.state_type) if uses_reference else 0.0
            if beta > 0:
                reference_kl = self._reference_kl(step, output.logits)
                reference_kl_terms.append(reference_kl)
                weighted_reference_kl_terms.append(beta * reference_kl)
                reference_kl_by_screen.setdefault(step.state_type, []).append(reference_kl.detach())
            action_index = torch.tensor(step.action_index, device=self.device)
            new_policy_log_probability = distribution.log_prob(action_index)
            old_log_probability = step.old_log_probability.to(self.device)
            old_policy_log_probability = step.old_policy_log_probability.to(self.device)
            if step.epsilon > 0.0:
                # The mixture was sampled, so the mixture is trained: its
                # gradient reaches the policy through (1 - epsilon) * pi, and
                # the uniform share keeps the ratio bounded on an action the
                # policy had all but abandoned.
                new_log_probability = _mixture_log_probability(
                    new_policy_log_probability, step.epsilon, len(step.decision.actions)
                )
            else:
                new_log_probability = new_policy_log_probability
            log_ratio = new_log_probability - old_log_probability
            ratio = torch.exp(log_ratio)
            # The low-variance estimator of KL(old || new): (r - 1) - log r >= 0.
            kl_terms.append(((ratio - 1.0) - log_ratio).detach())
            # The same estimator for the policy alone, KL(pi_old || pi_new).
            # The action came from the mixture, so each term is weighted by
            # w = pi_old(a) / mu_old(a), exactly 1 where nothing was mixed in:
            # w * ((q - 1) - log q) with q = pi_new(a) / pi_old(a). The product
            # w * q is taken as one exponent, because on an action the policy
            # had abandoned and has since recovered w underflows and q
            # overflows in float32, and 0 * inf is NaN where the term is finite.
            policy_log_ratio = new_policy_log_probability - old_policy_log_probability
            weight = torch.exp(old_policy_log_probability - old_log_probability)
            weighted_ratio = torch.exp(new_policy_log_probability - old_log_probability)
            policy_kl_terms.append(
                ((weighted_ratio - weight) - weight * policy_log_ratio).detach()
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

        zero = torch.zeros((), device=self.device)
        approx_kl = float(torch.stack(kl_terms).mean().cpu()) if kl_terms else 0.0
        policy_kl = (
            float(torch.stack(policy_kl_terms).mean().cpu()) if policy_kl_terms else 0.0
        )
        if self.config.target_kl is not None and approx_kl > 1.5 * self.config.target_kl:
            # Measured before stepping, so the step that would cross the limit is
            # never taken (Stable-Baselines3 uses the same 1.5 margin).
            return {"kl_stopped": 1.0, "approx_kl": approx_kl, "policy_kl": policy_kl}
        policy_loss = torch.stack(policy_losses).mean() if policy_losses else zero
        value_loss = torch.stack(value_losses).mean()
        entropy = torch.stack(entropies).mean() if entropies else zero
        loss = (
            policy_loss
            + self.config.value_coefficient * value_loss
            - self.config.entropy_coefficient * entropy
        )
        terms = [("policy_loss", policy_loss), ("value_loss", value_loss), ("entropy", entropy)]
        if uses_reference:
            # A minibatch of forced and external steps alone has no reference
            # step, and contributes nothing rather than failing.
            reference_kl = torch.stack(reference_kl_terms).mean() if reference_kl_terms else zero
            if not bool(torch.isfinite(reference_kl)):
                raise RuntimeError("the reference KL is not finite")
            if self.config.reference_kl_screens:
                # Each step on its screen's coefficient, over the minibatch's
                # policy steps, with the steps whose screen has none at 0.
                weighted_reference_kl = (
                    torch.stack(weighted_reference_kl_terms).sum() / len(policy_losses)
                    if weighted_reference_kl_terms
                    else zero
                )
            else:
                # One coefficient everywhere, so every policy step is a
                # reference step and the mean above is the one the loss takes.
                # Formed as beta times that mean rather than as the sum of the
                # weighted terms over the count: the two agree only to rounding,
                # and this is the path every run without overrides trained on
                # and resumes onto bit for bit.
                weighted_reference_kl = self.config.reference_kl_coefficient * reference_kl
            loss = loss + weighted_reference_kl
            terms.append(("beta * reference_kl", weighted_reference_kl))
        if not bool(torch.isfinite(loss)):
            # Before the backward pass, so Adam never steps on a NaN or inf
            # gradient. Every term can be finite while the sum is not: a
            # coefficient large enough to overflow its term in float32.
            raise RuntimeError(
                "the loss is not finite before the backward pass: "
                + ", ".join(f"{name}={float(value.detach().cpu()):g}" for name, value in terms)
            )
        self.optimizer.zero_grad()
        loss.backward()
        # A non-finite norm is refused rather than scaled by: the step would
        # otherwise write NaN into every weight it touches.
        gradient_norm = nn.utils.clip_grad_norm_(
            self.game_encoder.parameters(),
            self.config.max_grad_norm,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        metrics = {
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "rollout_steps": float(len(steps)),
            "approx_kl": approx_kl,
            "policy_kl": policy_kl,
        }
        if uses_reference:
            metrics["reference_kl"] = float(reference_kl.detach().cpu())
            metrics["reference_steps"] = float(len(reference_kl_terms))
            # One key per screen the term is in force on, 0.0 when this
            # minibatch held none of its steps -- like ``reference_kl`` itself.
            for screen in self.config.reference_screens():
                screen_terms = reference_kl_by_screen.get(screen)
                metrics[f"reference_kl/{screen}"] = (
                    float(torch.stack(screen_terms).mean().cpu()) if screen_terms else 0.0
                )
        return metrics

    def _reference_kl(self, step: _RolloutStep, logits: Tensor) -> Tensor:
        """KL(pi_ref || pi_theta) on one stored decision, from its re-encoded logits.

        sum_a pi_ref(a) (log pi_ref(a) - log pi_theta(a)), with pi_ref stored
        at sampling time over the same candidates. Its gradient on the logits
        is pi_theta - pi_ref, nonzero wherever the reference puts mass; the
        log-softmax keeps a logit gap of 60 finite where a log of the
        probability would not be.
        """
        reference = step.reference_log_probabilities
        if reference is None:
            raise RuntimeError(
                "a sampled decision carries no reference distribution; the "
                "reference KL needs one on every policy step of a screen with "
                "a positive coefficient"
            )
        # Stored on the CPU, so this is a check and not a device synchronisation.
        if not bool(torch.isfinite(reference).all()):
            raise RuntimeError("a stored reference distribution is not finite")
        log_pi = F.log_softmax(logits, dim=-1)
        reference = reference.to(self.device)
        if log_pi.shape != reference.shape:
            raise RuntimeError(
                f"reference distribution over {reference.numel()} candidates for "
                f"a decision with {log_pi.numel()}"
            )
        return (reference.exp() * (reference - log_pi)).sum()

    def _advantages_and_returns(
        self, steps: list[_RolloutStep], tail_value: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Return GAE advantages and returns for one lane's trajectory.

        ``tail_value`` is the sampled value of a held step that follows the last
        one (see ``update``). It replaces a fresh V(next_observation) only for a
        last step that did not end its episode.
        """
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
                    if bootstrap is None and tail_value is not None and not step.episode_end:
                        bootstrap = tail_value
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
