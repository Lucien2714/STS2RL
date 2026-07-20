"""PPO algorithm for candidate-action agents.

This module keeps only the PPO update rule, rollout collectors, optimizer, and
checkpoint IO. The three composable pieces come from elsewhere: legal-action
enumeration from :mod:`sts2rl.action_spaces`, feature encoding from
:mod:`sts2rl.encoders`, and the actor-critic network (policy module) from
:mod:`sts2rl.models.policies` — pass ``policy=`` to swap architectures.

A bare ``PPOCandidateAgent()`` defaults to the battle encoder/action space and
*is* the battle agent; pass ``encoder=`` / ``action_space=`` for a non-battle
screen (the orchestrator's ``create_screen_agent`` builds those bindings).
"""

from __future__ import annotations

import logging
import random

import torch
from torch import nn
from torch.distributions import Categorical

from sts2rl.action_spaces.battle import BattleActionSpace
from sts2rl.agents.candidate_agent import CandidateActionAgent
from sts2rl.encoders.battle_encoder import BattleStateEncoder
from sts2rl.models.policies import CandidatePPOPolicy

logger = logging.getLogger(__name__)


class PPORolloutCollector:
    """Per-trajectory rollout buffer for one client/episode stream.

    Each collector owns its own pending transition and rollout buffer, so several
    concurrent clients can share one ``PPOCandidateAgent`` (model + optimizer)
    without clobbering each other's on-policy data. The agent stays unaware of how
    many trajectories exist or how they are scheduled; the harness simply gives each
    client its own collector via ``PPOCandidateAgent.new_rollout()``.
    """

    def __init__(self, agent: "PPOCandidateAgent") -> None:
        self.agent = agent
        self.buffer: list[dict] = []
        self._pending: dict | None = None

    def choose_action(self, raw_state: dict, training: bool = True) -> dict:
        """Sample a legal action and stash its on-policy data on this collector."""
        agent = self.agent
        candidates = agent.valid_action_candidates(raw_state)
        if not candidates:
            action = agent._fallback_action(raw_state)
            agent.last_action_selection = {
                "method": "fallback",
                "reason": "no_valid_candidates",
                "training": training,
                "epsilon": agent.epsilon,
                "action_key": None,
                "q": None,
                "prob": None,
                "value": None,
                "valid_action_count": 0,
            }
            self._pending = None
            return action

        state_vector = agent.encode_state(raw_state)
        candidate_vectors = [
            agent.encode_action(raw_state, candidate["action"]) for candidate in candidates
        ]
        inputs = [state_vector + candidate_vector for candidate_vector in candidate_vectors]
        with torch.no_grad():
            input_tensor = torch.tensor(inputs, dtype=torch.float32, device=agent.device)
            state_tensor = torch.tensor([state_vector], dtype=torch.float32, device=agent.device)
            logits = agent.model.score(input_tensor)
            value = float(agent.model.value(state_tensor).item())
            distribution = Categorical(logits=logits)
            if training:
                action_index_tensor = distribution.sample()
                method = "ppo_sample"
            else:
                action_index_tensor = torch.argmax(logits)
                method = "ppo_argmax"
            action_index = int(action_index_tensor.item())
            logprob = float(distribution.log_prob(action_index_tensor).item())
            probs = distribution.probs.detach().cpu().tolist()
            logits_list = logits.detach().cpu().tolist()

        candidate = candidates[action_index]
        self._pending = {
            "state": list(state_vector),
            "candidate_vectors": [list(vector) for vector in candidate_vectors],
            "action_index": action_index,
            "action_vector": list(candidate_vectors[action_index]),
            "logprob": logprob,
            "value": value,
        }
        agent.last_action_selection = {
            "method": method,
            "reason": "candidate_policy",
            "training": training,
            "epsilon": agent.epsilon,
            "action_key": candidate["action_key"],
            "q": float(logits_list[action_index]),
            "prob": float(probs[action_index]),
            "value": value,
            "valid_action_count": len(candidates),
        }
        return agent._public_action(candidate)

    def remember(
        self,
        state,
        action_vector,
        reward: float,
        next_state,
        done: bool,
        next_action_vectors,
    ) -> None:
        """Append one transition to this collector's trajectory buffer."""
        agent = self.agent
        state = list(state)
        action_vector = list(action_vector)
        pending = self._pending
        if (
            pending is None
            or pending["state"] != state
            or pending["action_vector"] != action_vector
        ):
            pending = agent._transition_from_encoded_action(state, action_vector)

        self.buffer.append(
            {
                "state": state,
                "action_vector": action_vector,
                "reward": float(reward),
                "next_state": list(next_state),
                "done": bool(done),
                "next_action_vectors": [list(vector) for vector in next_action_vectors],
                "candidate_vectors": pending["candidate_vectors"],
                "action_index": int(pending["action_index"]),
                "logprob": float(pending["logprob"]),
                "value": float(pending["value"]),
            }
        )
        agent.trained_steps += 1
        self._pending = None


class PPOCandidateAgent(CandidateActionAgent):
    """Agent that samples legal candidate actions for one screen with PPO.

    The agent owns the shared model/optimizer and the update logic; per-trajectory
    rollout state lives in :class:`PPORolloutCollector` instances created via
    :meth:`new_rollout`. A built-in default collector keeps single-agent use
    (tests, evaluation, direct calls) working unchanged. It composes a
    :class:`~sts2rl.encoders` encoder (battle by default) via
    :class:`CandidateActionAgent` rather than subclassing it.
    """

    def __init__(
        self,
        encoder=None,
        action_space=None,
        gamma=0.99,
        learning_rate=0.00025,
        hidden_size=256,
        rollout_steps=128,
        minibatch_size=32,
        update_epochs=4,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy=None,
        device=None,
    ):
        super().__init__(
            encoder or BattleStateEncoder(),
            action_space or BattleActionSpace(),
            device=device,
        )
        self.ACTION_SCHEMA = self.encoder.PPO_SCHEMA
        self.gamma = gamma
        self.rollout_steps = rollout_steps
        self.minibatch_size = minibatch_size
        self.update_epochs = update_epochs
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.model = (
            policy
            or CandidatePPOPolicy(
                self.state_size,
                self.model_input_size,
                hidden_size,
            )
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)

        self.collectors: list[PPORolloutCollector] = []
        self._default_rollout = self.new_rollout()

    def new_rollout(self) -> PPORolloutCollector:
        """Create and register a rollout collector for one trajectory/client."""
        collector = PPORolloutCollector(self)
        self.collectors.append(collector)
        return collector

    def buffered_steps(self) -> int:
        """Return the total transitions buffered across all collectors."""
        return sum(len(collector.buffer) for collector in self.collectors)

    def choose_action(self, raw_state: dict, training: bool = True) -> dict:
        """Choose an action via the built-in default collector (single-agent use)."""
        return self._default_rollout.choose_action(raw_state, training=training)

    def remember(self, *args, **kwargs) -> None:
        """Record a transition on the built-in default collector (single-agent use)."""
        return self._default_rollout.remember(*args, **kwargs)

    def train_step(self) -> float | None:
        """Run a PPO update once enough rollout steps are buffered across collectors."""
        return self.update()

    def update(self) -> float | None:
        """Update the shared policy from all collectors' trajectories.

        Advantages/returns are computed per collector (each buffer is one
        temporally-ordered trajectory segment), then concatenated for a single
        minibatched PPO update. This keeps GAE within trajectory boundaries even
        when several clients fill their buffers independently.
        """
        segments = [collector.buffer for collector in self.collectors if collector.buffer]
        if sum(len(segment) for segment in segments) < self.rollout_steps:
            return None

        rollout: list[dict] = []
        advantage_segments = []
        return_segments = []
        for segment in segments:
            advantages, returns = self._segment_gae(segment)
            rollout.extend(segment)
            advantage_segments.append(advantages)
            return_segments.append(returns)

        advantages = torch.cat(advantage_segments)
        returns = torch.cat(return_segments)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.tensor(
            [transition["state"] for transition in rollout],
            dtype=torch.float32,
            device=self.device,
        )
        old_logprobs = torch.tensor(
            [transition["logprob"] for transition in rollout],
            dtype=torch.float32,
            device=self.device,
        )

        last_loss = None
        indices = list(range(len(rollout)))
        for _ in range(self.update_epochs):
            random.shuffle(indices)
            for start in range(0, len(indices), self.minibatch_size):
                minibatch_indices = indices[start : start + self.minibatch_size]
                new_logprobs, entropy = self._action_logprobs(rollout, minibatch_indices)
                new_values = self.model.value(states[minibatch_indices])

                logratio = new_logprobs - old_logprobs[minibatch_indices]
                ratio = logratio.exp()
                mb_advantages = advantages[minibatch_indices]
                policy_loss_unclipped = -mb_advantages * ratio
                policy_loss_clipped = -mb_advantages * torch.clamp(
                    ratio,
                    1.0 - self.clip_coef,
                    1.0 + self.clip_coef,
                )
                policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()
                value_loss = 0.5 * (new_values - returns[minibatch_indices]).pow(2).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                last_loss = float(loss.item())

        for collector in self.collectors:
            collector.buffer.clear()
        self.learn_steps += 1
        return last_loss

    def _segment_gae(self, segment: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (advantages, returns) for one trajectory segment via GAE."""
        rewards = [transition["reward"] for transition in segment]
        dones = [transition["done"] for transition in segment]
        values = torch.tensor(
            [transition["value"] for transition in segment],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            bootstrap_value = self._bootstrap_value(segment[-1])
            advantages = torch.zeros(len(segment), dtype=torch.float32, device=self.device)
            last_gae = 0.0
            for index in reversed(range(len(segment))):
                if index == len(segment) - 1:
                    next_value = bootstrap_value
                else:
                    next_value = values[index + 1]
                next_non_terminal = 0.0 if dones[index] else 1.0
                delta = rewards[index] + self.gamma * next_value * next_non_terminal - values[index]
                last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
                advantages[index] = last_gae
            returns = advantages + values
        return advantages, returns

    def current_q_values(self, raw_state: dict, selected_action: dict | None = None) -> dict:
        """Return PPO logits and probabilities for dashboard compatibility."""
        candidates = self.valid_action_candidates(raw_state)
        if not candidates:
            return {
                "available": False,
                "reason": "No legal PPO battle actions are available",
                "screen_type": raw_state.get("state_type"),
                "actions": [],
            }

        selected_key = None
        if selected_action is not None:
            try:
                selected_key = self.action_key(selected_action, raw_state)
            except (KeyError, ValueError):
                selected_key = None

        state_vector = self.encode_state(raw_state)
        inputs = [
            state_vector + self.encode_action(raw_state, candidate["action"])
            for candidate in candidates
        ]
        with torch.no_grad():
            input_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
            logits = self.model.score(input_tensor)
            probs = torch.softmax(logits, dim=0)
            logits_list = logits.detach().cpu().tolist()
            probs_list = probs.detach().cpu().tolist()

        actions = []
        best_valid = None
        for candidate, logit, prob in zip(candidates, logits_list, probs_list):
            action = {
                "id": candidate["action_key"],
                "key": candidate["action_key"],
                "q": self._safe_float(logit),
                "masked_q": self._safe_float(logit),
                "prob": self._safe_float(prob),
                "valid": True,
                "selected": candidate["action_key"] == selected_key,
            }
            actions.append(action)
            if action["q"] is not None and (best_valid is None or action["q"] > best_valid["q"]):
                best_valid = action

        return {
            "available": True,
            "screen_type": raw_state.get("state_type"),
            "selected_action_id": selected_key,
            "selected_q": next((action["q"] for action in actions if action["selected"]), None),
            "best_valid_action": best_valid,
            "actions": actions,
        }

    def bc_score(self, state_action: torch.Tensor) -> torch.Tensor:
        """Return differentiable actor logits per encoded state/action row."""
        return self.model.score(state_action)

    def _transition_from_encoded_action(
        self, state: list[float], action_vector: list[float]
    ) -> dict:
        with torch.no_grad():
            state_tensor = torch.tensor([state], dtype=torch.float32, device=self.device)
            input_tensor = torch.tensor(
                [state + action_vector],
                dtype=torch.float32,
                device=self.device,
            )
            logits = self.model.score(input_tensor)
            distribution = Categorical(logits=logits)
            action_index = torch.tensor(0, device=self.device)
            return {
                "state": list(state),
                "candidate_vectors": [list(action_vector)],
                "action_index": 0,
                "action_vector": list(action_vector),
                "logprob": float(distribution.log_prob(action_index).item()),
                "value": float(self.model.value(state_tensor).item()),
            }

    def _bootstrap_value(self, transition: dict):
        if transition["done"] or not transition["next_action_vectors"]:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            state_tensor = torch.tensor(
                [transition["next_state"]],
                dtype=torch.float32,
                device=self.device,
            )
            return self.model.value(state_tensor).squeeze(0)

    def _action_logprobs(self, rollout: list[dict], indices: list[int]):
        logprobs = []
        entropies = []
        for index in indices:
            transition = rollout[index]
            state = transition["state"]
            inputs = [
                state + candidate_vector for candidate_vector in transition["candidate_vectors"]
            ]
            input_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
            logits = self.model.score(input_tensor)
            distribution = Categorical(logits=logits)
            action_index = torch.tensor(
                transition["action_index"],
                dtype=torch.long,
                device=self.device,
            )
            logprobs.append(distribution.log_prob(action_index))
            entropies.append(distribution.entropy())
        return torch.stack(logprobs), torch.stack(entropies)

    def save(self, path: str) -> None:
        """Save PPO model, optimizer, and rollout counters."""
        torch.save(
            {
                "action_schema": self.ACTION_SCHEMA,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "trained_steps": self.trained_steps,
                "learn_steps": self.learn_steps,
                "state_size": self.state_size,
                "action_feature_size": self.action_feature_size,
                "model_input_size": self.model_input_size,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load PPO model and optimizer from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        action_schema = checkpoint.get("action_schema")
        if action_schema != self.ACTION_SCHEMA:
            raise ValueError(
                f"Incompatible PPO checkpoint action_schema={action_schema!r}; "
                f"expected {self.ACTION_SCHEMA!r}."
            )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.trained_steps = checkpoint.get("trained_steps", self.trained_steps)
        self.learn_steps = checkpoint.get("learn_steps", self.learn_steps)
        logger.info(
            "Loaded PPO checkpoint from %s: trained_steps=%d learn_steps=%d schema=%s",
            path,
            self.trained_steps,
            self.learn_steps,
            self.ACTION_SCHEMA,
        )
