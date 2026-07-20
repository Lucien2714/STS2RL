"""DQN algorithm for candidate-action agents.

This module keeps only the DQN update rule, replay buffer, optimizer, and
checkpoint IO. The three composable pieces come from elsewhere: legal-action
enumeration from :mod:`sts2rl.action_spaces`, feature encoding from
:mod:`sts2rl.encoders`, and the scoring network (policy module) from
:mod:`sts2rl.models.policies` — pass ``policy=`` to swap architectures.

A bare ``DQNCandidateAgent()`` defaults to the battle encoder/action space and
*is* the battle agent; pass ``encoder=`` / ``action_space=`` for a non-battle
screen (the orchestrator's ``create_screen_agent`` builds those bindings).
"""

from __future__ import annotations

import copy
import logging
import random
from collections import deque

import torch
from torch import nn

from sts2rl.action_spaces.battle import BattleActionSpace
from sts2rl.agents.candidate_agent import CandidateActionAgent
from sts2rl.encoders.battle_encoder import BattleStateEncoder
from sts2rl.models.policies import CandidateQNetwork

logger = logging.getLogger(__name__)


class DQNCandidateAgent(CandidateActionAgent):
    """Agent that scores legal action candidates for one screen with a DQN."""

    def __init__(
        self,
        encoder=None,
        action_space=None,
        gamma=0.8,
        epsilon=1.0,
        learning_rate=0.00025,
        epsilon_decay=0.9999,
        epsilon_min=0.10,
        batch_size=64,
        memory_size=100000,
        update_freq=4,
        update_freq_target=2000,
        hidden_size=256,
        policy=None,
        device=None,
    ):
        super().__init__(
            encoder or BattleStateEncoder(),
            action_space or BattleActionSpace(),
            device=device,
        )
        self.ACTION_SCHEMA = self.encoder.DQN_SCHEMA
        self.update_freq = update_freq
        self.update_freq_target = update_freq_target
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=memory_size)

        self.model = (policy or CandidateQNetwork(self.model_input_size, hidden_size)).to(
            self.device
        )
        # The target network is an algorithm concern: a frozen copy of whatever
        # policy module is in use (deepcopy generalizes to swapped architectures).
        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

    def choose_action(self, raw_state: dict, training: bool = True) -> dict:
        """Choose a legal action using exploration or candidate Q-values."""
        candidates = self.valid_action_candidates(raw_state)
        if not candidates:
            action = self._fallback_action(raw_state)
            self.last_action_selection = {
                "method": "fallback",
                "reason": "no_valid_candidates",
                "training": training,
                "epsilon": self.epsilon,
                "action_key": None,
                "q": None,
                "valid_action_count": 0,
            }
            return action

        if training and random.random() < self.epsilon:
            candidate = random.choice(candidates)
            self.last_action_selection = {
                "method": "epsilon_random",
                "reason": "epsilon_exploration",
                "training": training,
                "epsilon": self.epsilon,
                "action_key": candidate["action_key"],
                "q": None,
                "valid_action_count": len(candidates),
            }
            return self._public_action(candidate)

        q_values = self._score_candidates(raw_state, candidates, self.model)
        # Break ties randomly so equal/near-equal Q-values do not collapse onto the
        # first candidate (which is always end_turn) and bias the policy toward it.
        best_q = max(q_values)
        best_indices = [index for index, q in enumerate(q_values) if q >= best_q - 1e-6]
        best_index = random.choice(best_indices)
        candidate = candidates[best_index]
        self.last_action_selection = {
            "method": "greedy_q",
            "reason": "candidate_argmax",
            "training": training,
            "epsilon": self.epsilon,
            "action_key": candidate["action_key"],
            "q": q_values[best_index],
            "valid_action_count": len(candidates),
        }
        return self._public_action(candidate)

    def remember(
        self,
        state,
        action_vector,
        reward: float,
        next_state,
        done: bool,
        next_action_vectors,
    ) -> None:
        """Store one transition in replay memory."""
        self.replay_buffer.append(
            (
                list(state),
                list(action_vector),
                float(reward),
                list(next_state),
                bool(done),
                [list(vector) for vector in next_action_vectors],
            )
        )
        self.trained_steps += 1

    def train_step(self) -> float | None:
        """Run one replay update when enough samples are available."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones, next_actions = zip(*batch)

        state_action_inputs = [list(state) + list(action) for state, action in zip(states, actions)]
        current_inputs = torch.tensor(
            state_action_inputs,
            dtype=torch.float32,
            device=self.device,
        )
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.bool, device=self.device)

        current_q = self.model(current_inputs)

        max_next_q_values = []
        with torch.no_grad():
            for next_state, done, candidate_vectors in zip(next_states, dones, next_actions):
                if done or not candidate_vectors:
                    max_next_q_values.append(0.0)
                    continue
                next_inputs = [
                    list(next_state) + list(candidate_vector)
                    for candidate_vector in candidate_vectors
                ]
                next_tensor = torch.tensor(next_inputs, dtype=torch.float32, device=self.device)
                next_q = self.target_model(next_tensor)
                max_next_q_values.append(float(next_q.max().item()))

            max_next_q = torch.tensor(max_next_q_values, dtype=torch.float32, device=self.device)
            target_q = rewards_tensor + self.gamma * max_next_q * (~dones_tensor).float()

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.update_freq_target == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        if self.learn_steps % self.update_freq == 0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return float(loss.item())

    def current_q_values(self, raw_state: dict, selected_action: dict | None = None) -> dict:
        """Return candidate Q-values for dashboards and evaluation telemetry."""
        candidates = self.valid_action_candidates(raw_state)
        if not candidates:
            return {
                "available": False,
                "reason": "No legal DQN battle actions are available",
                "screen_type": raw_state.get("state_type"),
                "actions": [],
            }

        selected_key = None
        if selected_action is not None:
            try:
                selected_key = self.action_key(selected_action, raw_state)
            except (KeyError, ValueError):
                selected_key = None

        q_values = self._score_candidates(raw_state, candidates, self.model)
        actions = []
        best_valid = None
        for candidate, q_value in zip(candidates, q_values):
            action = {
                "id": candidate["action_key"],
                "key": candidate["action_key"],
                "q": self._safe_float(q_value),
                "masked_q": self._safe_float(q_value),
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

    def _score_candidates(self, raw_state: dict, candidates: list[dict], model) -> list[float]:
        if not candidates:
            return []
        state_vector = self.encode_state(raw_state)
        inputs = [
            state_vector + self.encode_action(raw_state, candidate["action"])
            for candidate in candidates
        ]
        with torch.no_grad():
            input_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
            q_tensor = model.score(input_tensor).detach().cpu()
        return [float(value) for value in q_tensor.tolist()]

    def bc_score(self, state_action: torch.Tensor) -> torch.Tensor:
        """Return a differentiable score per encoded state/action row.

        This is the single primitive behavioral cloning needs: the trainer encodes
        the candidate set once (via ``encode_state``/``encode_action``), then applies
        a softmax over these scores. Any candidate-scoring agent reuses the identical
        BC loss by overriding only this method. The DQN score is the Q-value.
        """
        return self.model.score(state_action)

    def save(self, path: str) -> None:
        """Save model, optimizer, and exploration state to a checkpoint."""
        torch.save(
            {
                "action_schema": self.ACTION_SCHEMA,
                "model_state_dict": self.model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "trained_steps": self.trained_steps,
                "learn_steps": self.learn_steps,
                "state_size": self.state_size,
                "action_feature_size": self.action_feature_size,
                "model_input_size": self.model_input_size,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model, optimizer, and exploration state from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        action_schema = checkpoint.get("action_schema")
        if action_schema != self.ACTION_SCHEMA:
            raise ValueError(
                f"Incompatible battle checkpoint action_schema={action_schema!r}; "
                f"expected {self.ACTION_SCHEMA!r}. Retrain or use a matching checkpoint."
            )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.trained_steps = checkpoint.get(
            "trained_steps",
            checkpoint.get("trained_step", checkpoint.get("learn_steps", self.trained_steps)),
        )
        self.learn_steps = checkpoint.get("learn_steps", self.learn_steps)
        logger.info(
            "Loaded DQN checkpoint from %s: trained_steps=%d learn_steps=%d epsilon=%.4f schema=%s",
            path,
            self.trained_steps,
            self.learn_steps,
            self.epsilon,
            self.ACTION_SCHEMA,
        )

    def buffered_steps(self) -> int:
        """Return the number of transitions currently buffered for an update."""
        return len(self.replay_buffer)

    def new_rollout(self) -> "SharedReplayCollector":
        """Return a rollout collector for one trajectory/client.

        DQN uses a single order-independent replay buffer, so per-trajectory
        isolation is unnecessary; every collector forwards to this shared agent.
        The collector exists only to give the orchestrator a uniform interface.
        """
        return SharedReplayCollector(self)


class SharedReplayCollector:
    """Rollout collector that forwards to a shared replay-buffer agent."""

    def __init__(self, agent: DQNCandidateAgent) -> None:
        self.agent = agent

    def choose_action(self, raw_state: dict, training: bool = True) -> dict:
        return self.agent.choose_action(raw_state, training=training)

    def remember(self, *args, **kwargs) -> None:
        return self.agent.remember(*args, **kwargs)
