"""End-to-end structured candidate PPO tests."""

from __future__ import annotations

import pytest
import torch

from sts2rl.actions import GameAction
from sts2rl.agents import CandidatePPOAgent, PPOConfig, Transition
from sts2rl.encoder import (
    EncoderConfig,
    GameEncoder,
    GameTokenizer,
    GameVocabulary,
    TokenizedDecision,
    TokenizedState,
)
from sts2rl.env import GameObservation


def _map_state(option_count: int = 2) -> dict[str, object]:
    options = [
        {"index": index, "col": index, "row": 1}
        for index in range(option_count)
    ]
    nodes: list[dict[str, object]] = [
        {
            "col": 0,
            "row": 0,
            "type": "Start",
            "children": [[index, 1] for index in range(option_count)],
        }
    ]
    nodes.extend(
        {
            "col": index,
            "row": 1,
            "type": "Monster" if index % 2 == 0 else "Event",
            "children": [[0, 2]],
        }
        for index in range(option_count)
    )
    return {
        "state_type": "map",
        "run": {"floor": 1},
        "player": {
            "character": "The Ironclad",
            "hp": 70,
            "max_hp": 80,
            "gold": 50,
            "relics": [],
            "potions": [],
            "status": [],
        },
        "map": {
            "current_position": {"col": 0, "row": 0},
            "visited": [{"col": 0, "row": 0}],
            "next_options": options,
            "nodes": nodes,
            "bosses": [{"col": 0, "row": 2}],
        },
    }


def _observation(state: dict[str, object]) -> GameObservation:
    detail = None
    if state.get("state_type") != "game_over":
        detail = {
            "state_type": "player_detail",
            "player": {"deck_count": 0, "deck": []},
        }
    return GameObservation(state, detail)


def _agent(
    *,
    rollout_size: int = 256,
    update_epochs: int = 1,
) -> CandidatePPOAgent:
    vocabulary = GameVocabulary.from_bundled_data()
    tokenizer = GameTokenizer(vocabulary)
    encoder = GameEncoder(
        vocabulary,
        EncoderConfig(hidden_dim=16, entity_heads=4, entity_ff_dim=32),
    )
    return CandidatePPOAgent(
        tokenizer=tokenizer,
        game_encoder=encoder,
        config=PPOConfig(
            rollout_size=rollout_size,
            update_epochs=update_epochs,
        ),
    )


def test_ppo_defaults_match_long_horizon_run_config():
    config = PPOConfig()

    assert config.gamma == 0.999
    assert config.gae_lambda == 0.98
    assert not hasattr(config, "hidden_dim")


def test_training_samples_only_current_candidates_and_updates_encoder_on_terminal():
    torch.manual_seed(30)
    agent = _agent(rollout_size=8)
    state = _map_state(2)
    observation = _observation(state)
    before = agent.game_encoder.entity_encoder.state_token.detach().clone()

    action = agent.choose_action(observation)

    assert action.to_dict() in [
        {"type": "choose_map_node", "index": 0},
        {"type": "choose_map_node", "index": 1},
    ]
    agent.observe(
        Transition(
            state=observation,
            action=action,
            reward=1.0,
            next_state=_observation({"state_type": "game_over"}),
            done=True,
        )
    )

    assert agent.last_update["rollout_steps"] == 1.0
    assert agent.last_update["value_loss"] >= 0.0
    assert not torch.equal(
        before,
        agent.game_encoder.entity_encoder.state_token.detach(),
    )


def test_rollout_keeps_cpu_tokens_next_state_and_original_candidates():
    torch.manual_seed(31)
    agent = _agent(rollout_size=20)
    state = _map_state(3)
    observation = _observation(state)
    action = agent.choose_action(observation)
    agent.observe(
        Transition(
            state=observation,
            action=action,
            reward=0.25,
            next_state=_observation(_map_state(1)),
            done=False,
        )
    )

    step = agent._rollout[0]
    assert isinstance(step.decision, TokenizedDecision)
    assert isinstance(step.next_state, TokenizedState)
    assert len(step.candidates) == 3
    assert step.decision.state.global_categorical.device.type == "cpu"
    assert step.old_log_probability.device.type == "cpu"
    assert step.old_value.device.type == "cpu"


@pytest.mark.parametrize("done", [True, False])
def test_terminal_and_nonterminal_bootstrap_are_distinct(done: bool):
    torch.manual_seed(32)
    agent = _agent(rollout_size=20)
    state = _map_state(1)
    observation = _observation(state)
    action = agent.choose_action(observation)
    assert agent._pending is not None
    old_value = agent._pending.value.item()
    agent.update = lambda: {}  # type: ignore[method-assign]
    next_observation = (
        _observation({"state_type": "game_over"})
        if done
        else _observation(_map_state(1))
    )
    agent.observe(
        Transition(
            state=observation,
            action=action,
            reward=2.0,
            next_state=next_observation,
            done=done,
        )
    )

    advantages, _ = agent._advantages_and_returns()
    if done:
        expected = 2.0 - old_value
    else:
        with torch.no_grad():
            next_value = agent.game_encoder.value(
                agent._rollout[0].next_state.to(agent.device)
            ).item()
        expected = 2.0 + agent.config.gamma * next_value - old_value
    assert advantages[0].item() == pytest.approx(expected)


def test_truncated_finish_updates_nonterminal_rollout():
    torch.manual_seed(33)
    agent = _agent(rollout_size=20)
    observation = _observation(_map_state(1))
    action = agent.choose_action(observation)
    next_observation = _observation(_map_state(1))
    agent.observe(
        Transition(
            state=observation,
            action=action,
            reward=0.5,
            next_state=next_observation,
            done=False,
        )
    )

    agent.finish_episode(next_observation, truncated=True)

    assert agent.last_update["rollout_steps"] == 1.0
    assert not agent._rollout


def test_observed_action_must_match_sampled_candidate():
    torch.manual_seed(34)
    agent = _agent()
    observation = _observation(_map_state(2))
    sampled = agent.choose_action(observation)
    other_index = 1 - int(sampled.params["index"])

    with pytest.raises(ValueError, match="does not match"):
        agent.observe(
            Transition(
                state=observation,
                action=GameAction("choose_map_node", index=other_index),
                reward=0.0,
                next_state=_observation(_map_state(1)),
                done=False,
            )
        )


def test_evaluation_is_deterministic_and_does_not_collect_rollouts():
    torch.manual_seed(35)
    agent = _agent()
    agent.eval()
    observation = _observation(_map_state(3))

    first = agent.choose_action(observation).to_dict()
    second = agent.choose_action(observation).to_dict()

    assert first == second
    assert agent._pending is None
    assert not agent._rollout
