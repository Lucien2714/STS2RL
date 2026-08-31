"""Candidate-action PPO behavior tests."""

import torch

from sts2rl.agents import CandidatePPOAgent, PPOConfig, Transition


class StubFeatureEncoder:
    """Small deterministic encoder used only to exercise PPO mechanics."""

    state_dim = 4
    action_dim = 4

    def encode_state(self, state):
        state_types = {"map": 0.0, "game_over": 1.0}
        options = state.get("map", {}).get("next_options", [])
        return torch.tensor(
            [
                state_types.get(state.get("state_type"), -1.0),
                float(len(options)),
                float(state.get("run", {}).get("floor", 0)),
                1.0,
            ],
            dtype=torch.float32,
        )

    def encode_action(self, state, action):
        del state
        params = action.get_params()
        return torch.tensor(
            [
                1.0 if action.get_type() == "choose_map_node" else 0.0,
                float(params.get("index", 0)),
                1.0 if action.get_type() == "end_turn" else 0.0,
                1.0,
            ],
            dtype=torch.float32,
        )


def test_ppo_samples_only_from_current_candidates_and_updates_on_terminal_step():
    torch.manual_seed(7)
    agent = CandidatePPOAgent(
        StubFeatureEncoder(),
        config=PPOConfig(hidden_dim=16, rollout_size=8, update_epochs=1),
    )
    state = {
        "state_type": "map",
        "map": {"next_options": [{"index": 2}, {"index": 5}]},
    }

    action = agent.choose_action(state)

    assert action.to_dict() in [
        {"type": "choose_map_node", "index": 2},
        {"type": "choose_map_node", "index": 5},
    ]

    agent.observe(
        Transition(
            state=state,
            action=action,
            reward=1.0,
            next_state={"state_type": "game_over"},
            done=True,
        )
    )

    assert agent.last_update["rollout_steps"] == 1.0
    assert agent.last_update["value_loss"] >= 0.0


def test_ppo_evaluation_is_deterministic_and_does_not_require_observe():
    agent = CandidatePPOAgent(
        StubFeatureEncoder(),
        config=PPOConfig(hidden_dim=8),
    )
    agent.train(False)
    state = {
        "state_type": "map",
        "map": {"next_options": [{"index": 0}, {"index": 1}]},
    }

    first = agent.choose_action(state).to_dict()
    second = agent.choose_action(state).to_dict()

    assert first == second
