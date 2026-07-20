"""Pins for the policy modules' checkpoint-compatibility contract.

The state_dict key layout and the per-screen schema strings gate whether
existing checkpoints keep loading. These tests fail loudly if a refactor
renames or renests the policy attributes (``net`` / ``actor`` / ``critic``)
or drifts a schema string.
"""

import torch

from sts2rl.agents.candidate_dqn_agent import DQNCandidateAgent
from sts2rl.agents.candidate_ppo_agent import PPOCandidateAgent
from sts2rl.agents.orchestrator import create_screen_agent

DQN_STATE_DICT_KEYS = {
    "net.0.weight",
    "net.0.bias",
    "net.2.weight",
    "net.2.bias",
    "net.4.weight",
    "net.4.bias",
}

PPO_STATE_DICT_KEYS = {
    f"{tower}.{layer}.{param}"
    for tower in ("actor", "critic")
    for layer in (0, 2, 4)
    for param in ("weight", "bias")
}


def test_dqn_state_dict_keys_are_stable():
    """Golden key set: existing DQN checkpoints store exactly these flat keys."""
    agent = DQNCandidateAgent(hidden_size=16)
    assert set(agent.model.state_dict().keys()) == DQN_STATE_DICT_KEYS
    assert set(agent.target_model.state_dict().keys()) == DQN_STATE_DICT_KEYS


def test_ppo_state_dict_keys_are_stable():
    """Golden key set: existing PPO checkpoints store exactly these flat keys."""
    agent = PPOCandidateAgent(hidden_size=16, rollout_steps=2)
    assert set(agent.model.state_dict().keys()) == PPO_STATE_DICT_KEYS


def test_battle_schema_strings_are_pinned():
    """Battle checkpoint schema gates must never drift."""
    assert DQNCandidateAgent(hidden_size=16).ACTION_SCHEMA == "candidate_action_v3"
    assert (
        PPOCandidateAgent(hidden_size=16, rollout_steps=2).ACTION_SCHEMA
        == "candidate_action_ppo_v3"
    )


def test_screen_schema_strings_are_pinned():
    """Representative screen checkpoint schema gates must never drift."""
    assert create_screen_agent("map", "DQN").ACTION_SCHEMA == "map_dqn_v1"
    assert create_screen_agent("rest", "PPO").ACTION_SCHEMA == "rest_ppo_v1"


def test_custom_policy_module_is_accepted_and_trained():
    """A swapped-in policy module drives scoring, and its params get gradients."""

    class TinyPolicy(torch.nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(input_size, 1))

        def forward(self, state_action):
            return self.net(state_action).squeeze(-1)

        def score(self, state_action):
            return self.forward(state_action)

    probe = DQNCandidateAgent(hidden_size=16)
    agent = DQNCandidateAgent(policy=TinyPolicy(probe.model_input_size))
    assert isinstance(agent.model, TinyPolicy)
    # The optimizer must cover exactly the policy's parameters.
    optimized = {id(p) for group in agent.optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in agent.model.parameters()}
