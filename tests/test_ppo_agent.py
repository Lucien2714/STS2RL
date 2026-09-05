"""End-to-end structured candidate PPO tests."""

from __future__ import annotations

import pytest
import torch

from sts2rl.agents import CandidatePPOAgent, PPOConfig, Transition
from sts2rl.encoder import (
    EncoderConfig,
    GameEncoder,
    GameTokenizer,
    GameVocabulary,
    TokenizedDecision,
)
from sts2rl.env import GameObservation


def _map_state(option_count: int = 2) -> dict[str, object]:
    options = [
        {"index": index, "col": index, "row": 1} for index in range(option_count)
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
    return GameObservation(state)


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
    assert agent.environment_steps == 1
    assert agent.optimizer_updates == 1
    assert not torch.equal(
        before,
        agent.game_encoder.entity_encoder.state_token.detach(),
    )


def test_update_metrics_are_drained_once():
    torch.manual_seed(36)
    agent = _agent(rollout_size=8)
    observation = _observation(_map_state(2))
    action = agent.choose_action(observation)
    agent.observe(
        Transition(
            state=observation,
            action=action,
            reward=1.0,
            next_state=_observation({"state_type": "game_over"}),
            done=True,
        )
    )

    metrics = agent.drain_update_metrics()

    assert len(metrics) == 1
    assert metrics[0]["environment_steps"] == 1.0
    assert metrics[0]["optimizer_update"] == 1.0
    assert agent.drain_update_metrics() == ()


def test_agent_checkpoint_round_trip_restores_logits_and_optimizer():
    torch.manual_seed(37)
    agent = _agent(rollout_size=8)
    observation = _observation(_map_state(2))
    action = agent.choose_action(observation)
    agent.observe(
        Transition(
            state=observation,
            action=action,
            reward=1.0,
            next_state=_observation({"state_type": "game_over"}),
            done=True,
        )
    )
    candidates = agent.action_provider.require_candidates(observation.raw_state)
    decision = agent.tokenizer.tokenize_decision(observation, candidates)
    with torch.no_grad():
        expected = agent.game_encoder.policy_value(decision).logits.clone()
    checkpoint = agent.checkpoint_state()

    restored = _agent(rollout_size=8)
    restored.load_checkpoint_state(checkpoint)
    with torch.no_grad():
        actual = restored.game_encoder.policy_value(decision).logits

    assert torch.equal(actual, expected)
    assert restored.optimizer.state


def test_checkpoint_requires_clean_boundary_and_abort_discards_partial_work():
    torch.manual_seed(38)
    agent = _agent(rollout_size=20)
    observation = _observation(_map_state(2))
    action = agent.choose_action(observation)

    with pytest.raises(RuntimeError, match="unobserved action"):
        agent.checkpoint_state()

    agent.observe(
        Transition(
            state=observation,
            action=action,
            reward=0.0,
            next_state=_observation(_map_state(1)),
            done=False,
        )
    )
    with pytest.raises(RuntimeError, match="non-empty rollout"):
        agent.checkpoint_state()

    agent.abort_episode()

    assert agent._pending is None
    assert not agent._rollout
    assert set(agent.checkpoint_state()) == {"encoder", "optimizer"}
    assert agent.environment_steps == 1


def test_rollout_keeps_cpu_tokens_and_defers_next_state_tokenization():
    torch.manual_seed(31)
    agent = _agent(rollout_size=20)
    state = _map_state(3)
    observation = _observation(state)
    next_observation = _observation(_map_state(2))
    action = agent.choose_action(observation)
    agent.observe(
        Transition(
            state=observation,
            action=action,
            reward=0.25,
            next_state=next_observation,
            done=False,
        )
    )

    step = agent._rollout[0]
    assert isinstance(step.decision, TokenizedDecision)
    assert len(step.decision.actions) == 3
    assert step.next_observation is next_observation
    assert step.decision.state.global_categorical.device.type == "cpu"
    assert step.old_log_probability.device.type == "cpu"
    assert step.old_value.device.type == "cpu"


@pytest.mark.parametrize("done", [True, False])
def test_terminal_and_nonterminal_bootstrap_are_distinct(done: bool):
    torch.manual_seed(32)
    agent = _agent(rollout_size=20)
    state = _map_state(2)
    observation = _observation(state)
    action = agent.choose_action(observation)
    assert agent._pending is not None
    old_value = agent._pending.value.item()
    agent.update = lambda: {}  # type: ignore[method-assign]
    next_observation = (
        _observation({"state_type": "game_over"})
        if done
        else _observation(_map_state(2))
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
                agent.tokenizer.tokenize_state(
                    agent._rollout[0].next_observation
                ).to(agent.device)
            ).item()
        expected = 2.0 + agent.config.gamma * next_value - old_value
    assert advantages[0].item() == pytest.approx(expected)


def test_truncated_finish_updates_nonterminal_rollout():
    torch.manual_seed(33)
    agent = _agent(rollout_size=20)
    observation = _observation(_map_state(2))
    action = agent.choose_action(observation)
    next_observation = _observation(_map_state(2))
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


def test_forced_single_candidate_is_executed_without_entering_the_rollout():
    torch.manual_seed(34)
    agent = _agent(rollout_size=20)
    observation = _observation(_map_state(1))

    action = agent.choose_action(observation)
    agent.observe(
        Transition(
            state=observation,
            action=action,
            reward=3.0,
            next_state=_observation(_map_state(2)),
            done=False,
        )
    )

    assert action.to_dict() == {"type": "choose_map_node", "index": 0}
    assert agent._pending is None
    assert agent._rollout == []
    assert agent.environment_steps == 1
    assert agent._carried_reward == 3.0


def test_forced_step_reward_is_folded_into_the_preceding_decision():
    torch.manual_seed(39)
    agent = _agent(rollout_size=20)
    decision_observation = _observation(_map_state(2))
    forced_observation = _observation(_map_state(1))

    action = agent.choose_action(decision_observation)
    agent.observe(
        Transition(
            state=decision_observation,
            action=action,
            reward=1.0,
            next_state=forced_observation,
            done=False,
        )
    )
    forced_action = agent.choose_action(forced_observation)
    terminal = _observation({"state_type": "game_over"})
    agent.update = lambda: {}  # type: ignore[method-assign]
    agent.observe(
        Transition(
            state=forced_observation,
            action=forced_action,
            reward=4.0,
            next_state=terminal,
            done=True,
        )
    )

    assert len(agent._rollout) == 1
    assert agent._rollout[0].reward == pytest.approx(5.0)
    assert agent._rollout[0].done is True
    assert agent._rollout[0].next_observation is terminal
    assert agent.environment_steps == 2


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
