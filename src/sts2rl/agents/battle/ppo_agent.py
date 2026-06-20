"""Simple candidate-action PPO battle agent."""

from __future__ import annotations

import logging
import random

import torch
from torch import nn
from torch.distributions import Categorical

from sts2rl.agents.battle.dqn_agent import BattleDQNAgent


logger = logging.getLogger(__name__)


def layer_init(layer: nn.Linear, std: float = 1.0, bias_const: float = 0.0) -> nn.Linear:
  """Initialize a linear layer following the small CleanRL PPO convention."""
  nn.init.orthogonal_(layer.weight, std)
  nn.init.constant_(layer.bias, bias_const)
  return layer


class BattlePPOPolicy(nn.Module):
  """Actor-critic network for variable candidate-action sets."""

  def __init__(self, state_size: int, state_action_size: int, hidden_size: int = 256):
    super().__init__()
    self.actor = nn.Sequential(
      layer_init(nn.Linear(state_action_size, hidden_size)),
      nn.Tanh(),
      layer_init(nn.Linear(hidden_size, hidden_size)),
      nn.Tanh(),
      layer_init(nn.Linear(hidden_size, 1), std=0.01),
    )
    self.critic = nn.Sequential(
      layer_init(nn.Linear(state_size, hidden_size)),
      nn.Tanh(),
      layer_init(nn.Linear(hidden_size, hidden_size)),
      nn.Tanh(),
      layer_init(nn.Linear(hidden_size, 1), std=1.0),
    )

  def action_logits(self, state_actions):
    """Return one logit per encoded state/action candidate."""
    return self.actor(state_actions).squeeze(-1)

  def value(self, states):
    """Return state values."""
    return self.critic(states).squeeze(-1)


class BattlePPOAgent(BattleDQNAgent):
  """Battle agent that samples legal candidate actions with PPO."""

  ACTION_SCHEMA = "candidate_action_ppo_v2"

  def __init__(
    self,
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
    device=None,
  ):
    super().__init__(
      gamma=gamma,
      epsilon=0.0,
      learning_rate=learning_rate,
      hidden_size=hidden_size,
      batch_size=rollout_steps,
      device=device,
    )
    self.rollout_steps = rollout_steps
    self.minibatch_size = minibatch_size
    self.update_epochs = update_epochs
    self.gae_lambda = gae_lambda
    self.clip_coef = clip_coef
    self.ent_coef = ent_coef
    self.vf_coef = vf_coef
    self.max_grad_norm = max_grad_norm

    self.model = BattlePPOPolicy(
      self.state_size,
      self.model_input_size,
      hidden_size,
    ).to(self.device)
    self.target_model = None
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)
    self.replay_buffer = []
    self._pending_transition = None

  def choose_action(self, raw_state: dict, training: bool = True) -> dict:
    """Choose a legal battle action from the PPO policy distribution."""
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
        "prob": None,
        "value": None,
        "valid_action_count": 0,
      }
      return action

    state_vector = self.encode_state(raw_state)
    candidate_vectors = [
      self.encode_action(raw_state, candidate["action"])
      for candidate in candidates
    ]
    inputs = [
      state_vector + candidate_vector
      for candidate_vector in candidate_vectors
    ]
    with torch.no_grad():
      input_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
      state_tensor = torch.tensor([state_vector], dtype=torch.float32, device=self.device)
      logits = self.model.action_logits(input_tensor)
      value = float(self.model.value(state_tensor).item())
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
    self._pending_transition = {
      "state": list(state_vector),
      "candidate_vectors": [list(vector) for vector in candidate_vectors],
      "action_index": action_index,
      "action_vector": list(candidate_vectors[action_index]),
      "logprob": logprob,
      "value": value,
    }
    self.last_action_selection = {
      "method": method,
      "reason": "candidate_policy",
      "training": training,
      "epsilon": self.epsilon,
      "action_key": candidate["action_key"],
      "q": float(logits_list[action_index]),
      "prob": float(probs[action_index]),
      "value": value,
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
    """Store one PPO rollout transition."""
    state = list(state)
    action_vector = list(action_vector)
    pending = self._pending_transition
    if (
      pending is None
      or pending["state"] != state
      or pending["action_vector"] != action_vector
    ):
      pending = self._transition_from_encoded_action(state, action_vector)

    self.replay_buffer.append({
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
    })
    self.trained_steps += 1
    self._pending_transition = None

  def train_step(self) -> float | None:
    """Run a PPO update when the rollout buffer is full."""
    if len(self.replay_buffer) < self.rollout_steps:
      return None

    rollout = list(self.replay_buffer)
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
    rewards = [transition["reward"] for transition in rollout]
    dones = [transition["done"] for transition in rollout]
    values = torch.tensor(
      [transition["value"] for transition in rollout],
      dtype=torch.float32,
      device=self.device,
    )

    with torch.no_grad():
      bootstrap_value = self._bootstrap_value(rollout[-1])
      advantages = torch.zeros(len(rollout), dtype=torch.float32, device=self.device)
      last_gae = 0.0
      for index in reversed(range(len(rollout))):
        if index == len(rollout) - 1:
          next_value = bootstrap_value
        else:
          next_value = values[index + 1]
        next_non_terminal = 0.0 if dones[index] else 1.0
        delta = rewards[index] + self.gamma * next_value * next_non_terminal - values[index]
        last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        advantages[index] = last_gae
      returns = advantages + values
      if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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

    self.replay_buffer.clear()
    self.learn_steps += 1
    return last_loss

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
      logits = self.model.action_logits(input_tensor)
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
    return self.model.action_logits(state_action)

  def _transition_from_encoded_action(self, state: list[float], action_vector: list[float]) -> dict:
    with torch.no_grad():
      state_tensor = torch.tensor([state], dtype=torch.float32, device=self.device)
      input_tensor = torch.tensor(
        [state + action_vector],
        dtype=torch.float32,
        device=self.device,
      )
      logits = self.model.action_logits(input_tensor)
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
        state + candidate_vector
        for candidate_vector in transition["candidate_vectors"]
      ]
      input_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
      logits = self.model.action_logits(input_tensor)
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


PPOBattleAgent = BattlePPOAgent
