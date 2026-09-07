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
    agent = _agent(rollout_size=1)
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
    agent = _agent(rollout_size=1)
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
    agent = _agent(rollout_size=1)
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

    restored = _agent(rollout_size=1)
    restored.load_checkpoint_state(checkpoint)
    with torch.no_grad():
        actual = restored.game_encoder.policy_value(decision).logits

    assert torch.equal(actual, expected)
    assert restored.optimizer.state


def test_saving_is_a_snapshot_that_does_not_disturb_the_rollout():
    """Demanding an empty rollout is what forced updates on 13 transitions."""
    torch.manual_seed(38)
    agent = _agent(rollout_size=20)
    observation = _observation(_map_state(2))
    action = agent.choose_action(observation)

    # Mid-decision on one lane, and mid-rollout after it: both are fine to save.
    payload = agent.checkpoint_state()
    assert set(payload) == {"encoder", "optimizer"}

    agent.observe(
        Transition(
            state=observation,
            action=action,
            reward=0.0,
            next_state=_observation(_map_state(1)),
            done=False,
        )
    )
    agent.checkpoint_state()

    assert len(agent._lane(0).steps) == 1
    assert agent.optimizer_updates == 0


def test_a_saved_snapshot_is_detached_from_later_training():
    """Another client can be mid-optimizer-step while the file is written."""
    torch.manual_seed(38)
    agent = _agent(rollout_size=2)
    payload = agent.checkpoint_state()
    before = {name: tensor.clone() for name, tensor in payload["encoder"].items()}

    for _ in range(2):
        _step(agent, 0, reward=1.0, done=False)

    assert agent.optimizer_updates == 1
    for name, tensor in payload["encoder"].items():
        assert torch.equal(tensor, before[name])


def test_loading_still_refuses_an_agent_with_work_in_flight():
    """Overwriting the weights a pending decision was made under is real damage."""
    torch.manual_seed(38)
    agent = _agent()
    payload = agent.checkpoint_state()
    agent.choose_action(_observation(_map_state(2)))

    with pytest.raises(RuntimeError, match="unobserved action"):
        agent.load_checkpoint_state(payload)


def test_aborting_discards_partial_work():
    torch.manual_seed(38)
    agent = _agent(rollout_size=20)
    observation = _observation(_map_state(2))
    action = agent.choose_action(observation)
    agent.observe(
        Transition(
            state=observation,
            action=action,
            reward=0.0,
            next_state=_observation(_map_state(1)),
            done=False,
        )
    )

    agent.abort_episode()

    assert agent._lane(0).pending is None
    assert not agent._lane(0).steps
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

    step = agent._lane(0).steps[0]
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
    assert agent._lane(0).pending is not None
    old_value = agent._lane(0).pending.value.item()
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

    advantages, _ = agent._advantages_and_returns(agent._lane(0).steps)
    if done:
        expected = 2.0 - old_value
    else:
        with torch.no_grad():
            next_value = agent.game_encoder.value(
                agent.tokenizer.tokenize_state(
                    agent._lane(0).steps[0].next_observation
                ).to(agent.device)
            ).item()
        expected = 2.0 + agent.config.gamma * next_value - old_value
    assert advantages[0].item() == pytest.approx(expected)


def test_finishing_an_episode_keeps_the_rollout_for_the_next_one():
    """Episodes are far shorter than a rollout, so they must accumulate."""
    torch.manual_seed(33)
    agent = _agent(rollout_size=20)
    for done in (True, False, True):
        observation = _observation(_map_state(2))
        action = agent.choose_action(observation)
        next_observation = _observation(_map_state(2))
        agent.observe(
            Transition(
                state=observation,
                action=action,
                reward=0.5,
                next_state=next_observation,
                done=done,
            )
        )
        agent.finish_episode(next_observation, truncated=not done)

    assert len(agent._lane(0).steps) == 3
    assert agent.last_update == {}
    assert [step.done for step in agent._lane(0).steps] == [True, False, True]

    agent.update()

    assert agent.last_update["rollout_steps"] == 3.0
    assert not agent._lane(0).steps


def test_a_terminal_inside_the_rollout_cuts_the_return_there():
    """Accumulating across episodes is only safe if GAE stops at a terminal."""
    torch.manual_seed(44)
    agent = _agent(rollout_size=20)
    for done in (True, False):
        observation = _observation(_map_state(2))
        action = agent.choose_action(observation)
        agent.observe(
            Transition(
                state=observation,
                action=action,
                reward=1.0,
                next_state=_observation(_map_state(2)),
                done=done,
            )
        )

    values = [float(step.old_value) for step in agent._lane(0).steps]
    advantages, _ = agent._advantages_and_returns(agent._lane(0).steps)

    # the first step ends an episode, so its advantage must not see the second
    assert advantages[0].item() == pytest.approx(1.0 - values[0])


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
    assert agent._lane(0).pending is None
    assert agent._lane(0).steps == []
    assert agent.environment_steps == 1
    assert agent._lane(0).carried_reward == 3.0


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

    assert len(agent._lane(0).steps) == 1
    assert agent._lane(0).steps[0].reward == pytest.approx(5.0)
    assert agent._lane(0).steps[0].done is True
    assert agent._lane(0).steps[0].next_observation is terminal
    assert agent.environment_steps == 2


def test_evaluation_is_deterministic_and_does_not_collect_rollouts():
    torch.manual_seed(35)
    agent = _agent()
    agent.eval()
    observation = _observation(_map_state(3))

    first = agent.choose_action(observation).to_dict()
    second = agent.choose_action(observation).to_dict()

    assert first == second
    assert agent._lane(0).pending is None
    assert not agent._lane(0).steps


def _step(agent, lane: int, *, reward: float, done: bool) -> None:
    """Run one full decision on one lane."""
    observation = _observation(_map_state(2))
    action = agent.choose_action(observation, lane=lane)
    agent.observe(
        Transition(
            state=observation,
            action=action,
            reward=reward,
            next_state=_observation(_map_state(2)),
            done=done,
            info={},
        ),
        lane=lane,
    )


def test_lanes_keep_their_pending_decisions_apart():
    """Two clients decide concurrently; neither may consume the other's."""
    agent = _agent()
    first = _observation(_map_state(2))
    second = _observation(_map_state(3))

    agent.choose_action(first, lane=0)
    agent.choose_action(second, lane=1)

    assert agent._lane(0).pending is not None
    assert agent._lane(1).pending is not None
    assert agent._lane(0).pending is not agent._lane(1).pending


def test_a_lane_still_rejects_two_decisions_without_an_observation():
    agent = _agent()
    agent.choose_action(_observation(_map_state(2)), lane=1)

    with pytest.raises(RuntimeError, match="observe"):
        agent.choose_action(_observation(_map_state(2)), lane=1)


def test_the_rollout_fills_from_every_lane_together():
    agent = _agent(rollout_size=4)

    for lane in (0, 1):
        for _ in range(2):
            _step(agent, lane, reward=1.0, done=False)

    # Four steps across two lanes reached rollout_size, so the update ran.
    assert agent.optimizer_updates == 1
    assert agent.last_update["lanes"] == 2.0
    assert agent.last_update["rollout_steps"] == 4.0


def test_advantage_never_flows_from_one_lane_into_another():
    """The whole reason lanes exist: GAE walks a trajectory, not a list.

    A lane's tail is normally mid-episode -- the rollout fills while every
    client is still playing -- so ``done`` masking does not protect it.  The
    backward walk would carry the next lane's trace straight into it.
    """
    agent = _agent()

    _step(agent, 0, reward=1.0, done=False)
    _step(agent, 0, reward=1.0, done=False)
    # A large reward on the other lane is only visible in lane 0 if the walk
    # ran over the concatenation instead of per lane.
    _step(agent, 1, reward=100.0, done=False)

    per_lane, _ = agent._advantages_and_returns(agent._lane(0).steps)
    concatenated, _ = agent._advantages_and_returns(
        agent._lane(0).steps + agent._lane(1).steps
    )

    assert float(concatenated[1]) > float(per_lane[-1]) + 10.0
    assert len(per_lane) == 2


def test_update_computes_the_advantage_each_lane_would_get_alone():
    agent = _agent(rollout_size=1000)

    _step(agent, 0, reward=1.0, done=False)
    _step(agent, 1, reward=100.0, done=False)
    expected, _ = agent._advantages_and_returns(agent._lane(0).steps)

    # Recomputing after adding the loud lane must not move lane 0.
    _step(agent, 1, reward=100.0, done=False)
    actual, _ = agent._advantages_and_returns(agent._lane(0).steps)

    assert float(actual[0]) == pytest.approx(float(expected[0]))


def test_a_forced_step_is_absorbed_into_its_own_lane():
    agent = _agent()
    _step(agent, 0, reward=1.0, done=False)
    _step(agent, 1, reward=1.0, done=False)

    # Lane 1 takes a forced step worth 5.0; lane 0 must not see it.
    agent._lane(1).forced_action = True
    observation = _observation(_map_state(2))
    agent.observe(
        Transition(
            state=observation,
            action=agent._lane(1).steps[-1].decision.candidates[0]
            if hasattr(agent._lane(1).steps[-1].decision, "candidates")
            else None,
            reward=5.0,
            next_state=observation,
            done=False,
            info={},
        ),
        lane=1,
    )

    assert agent._lane(1).steps[-1].reward == pytest.approx(6.0)
    assert agent._lane(0).steps[-1].reward == pytest.approx(1.0)


def test_a_lane_view_binds_every_call_to_its_lane():
    agent = _agent()
    view = agent.lane_view(2)

    observation = _observation(_map_state(2))
    action = view.choose_action(observation)
    view.observe(
        Transition(
            state=observation,
            action=action,
            reward=1.0,
            next_state=observation,
            done=False,
            info={},
        )
    )

    assert len(agent._lane(2).steps) == 1
    assert agent._lane(0).steps == []


def test_the_update_hook_fires_where_the_rollout_is_empty():
    """The trainer checkpoints here, which is why nothing has to be flushed."""
    agent = _agent(rollout_size=2)
    seen: list[int] = []
    agent.on_update = lambda: seen.append(agent._rollout_length())

    for _ in range(2):
        _step(agent, 0, reward=1.0, done=False)

    assert seen == [0]


def test_aborting_a_lane_discards_its_whole_trajectory():
    """Steps recorded on the way to a crash are the least trustworthy ones."""
    agent = _agent(rollout_size=1000)
    for _ in range(3):
        _step(agent, 0, reward=1.0, done=False)
    agent.choose_action(_observation(_map_state(2)), lane=0)

    agent.abort_lane(0)

    assert agent._lane(0).steps == []
    assert agent._lane(0).pending is None
    assert agent._lane(0).carried_reward == 0.0


def test_aborting_one_lane_leaves_the_other_clients_alone():
    agent = _agent(rollout_size=1000)
    _step(agent, 0, reward=1.0, done=False)
    _step(agent, 1, reward=1.0, done=False)
    _step(agent, 1, reward=1.0, done=False)

    agent.abort_lane(0)

    assert agent._lane(0).steps == []
    assert len(agent._lane(1).steps) == 2


def test_an_aborted_lane_does_not_break_the_next_update():
    agent = _agent(rollout_size=1000)
    _step(agent, 0, reward=1.0, done=False)
    _step(agent, 1, reward=1.0, done=False)
    agent.abort_lane(0)

    metrics = agent.update()

    assert metrics["lanes"] == 1.0
    assert metrics["rollout_steps"] == 1.0
