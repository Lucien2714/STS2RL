"""Candidate-action PPO behavior tests."""

import torch

from sts2rl.agents import CandidatePPOAgent, PPOConfig, Transition


def test_ppo_samples_only_from_current_candidates_and_updates_on_terminal_step():
    torch.manual_seed(7)
    agent = CandidatePPOAgent(
        config=PPOConfig(hidden_dim=16, rollout_size=8, update_epochs=1)
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
    agent = CandidatePPOAgent(config=PPOConfig(hidden_dim=8))
    agent.train(False)
    state = {
        "state_type": "map",
        "map": {"next_options": [{"index": 0}, {"index": 1}]},
    }

    first = agent.choose_action(state).to_dict()
    second = agent.choose_action(state).to_dict()

    assert first == second
