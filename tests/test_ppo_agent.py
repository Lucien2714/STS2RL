"""End-to-end structured candidate PPO tests."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from sts2rl.actions import GameAction
from sts2rl.agents import CandidatePPOAgent, NoLegalActionsError, PPOConfig, Transition
from sts2rl.agents.ppo import STATE_TYPES, _mixture_log_probability, _policy_step
from sts2rl.encoder import (
    EncoderConfig,
    GameEncoder,
    GameTokenizer,
    GameVocabulary,
    TokenizedDecision,
)
from sts2rl.encoder.game_encoder import PolicyValueOutput
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


TEST_ENCODER = EncoderConfig(hidden_dim=16, entity_heads=4, entity_ff_dim=32)


def _agent(
    *,
    rollout_size: int = 256,
    update_epochs: int = 1,
    target_kl: float | None = None,
    minibatch_size: int = 32,
    hold_open_steps: bool = False,
    exploration: Mapping[str, float] | tuple[tuple[str, float], ...] = (),
    learning_rate: float = 3e-4,
    reference_kl: float = 0.0,
    reference_kl_screens: Mapping[str, float] | tuple[tuple[str, float], ...] = (),
    reference: nn.Module | None = None,
    encoder: nn.Module | None = None,
) -> CandidatePPOAgent:
    vocabulary = GameVocabulary.from_bundled_data()
    tokenizer = GameTokenizer(vocabulary)
    if encoder is None:
        encoder = GameEncoder(vocabulary, TEST_ENCODER)
    return CandidatePPOAgent(
        tokenizer=tokenizer,
        game_encoder=encoder,
        config=PPOConfig(
            learning_rate=learning_rate,
            rollout_size=rollout_size,
            update_epochs=update_epochs,
            target_kl=target_kl,
            minibatch_size=minibatch_size,
            exploration=exploration,
            reference_kl_coefficient=reference_kl,
            reference_kl_screens=reference_kl_screens,
        ),
        hold_open_steps=hold_open_steps,
        reference_encoder=reference,
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
    assert set(payload) == {"encoder", "optimizer", "return_scale"}

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
    assert set(agent.checkpoint_state()) == {"encoder", "optimizer", "return_scale"}
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


def test_forced_single_candidate_is_recorded_as_a_critic_only_step():
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
    assert len(agent._lane(0).steps) == 1
    step = agent._lane(0).steps[0]
    assert len(step.decision.actions) == 1
    assert float(step.old_log_probability) == 0.0
    assert step.reward == 3.0
    assert agent.environment_steps == 1


def test_forced_step_preserves_its_own_reward_and_terminal():
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

    assert len(agent._lane(0).steps) == 2
    first, forced = agent._lane(0).steps
    assert first.reward == pytest.approx(1.0)
    assert first.done is False
    assert forced.reward == pytest.approx(4.0)
    assert forced.done is True
    assert forced.next_observation is terminal
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


def test_a_forced_step_is_recorded_in_its_own_lane():
    agent = _agent()
    _step(agent, 0, reward=1.0, done=False)
    _step(agent, 1, reward=1.0, done=False)

    # Lane 1 takes a forced step worth 5.0; lane 0 must not see it.
    observation = _observation(_map_state(1))
    action = agent.choose_action(observation, lane=1)
    agent.observe(
        Transition(
            state=observation,
            action=action,
            reward=5.0,
            next_state=observation,
            done=False,
            info={},
        ),
        lane=1,
    )

    assert [step.reward for step in agent._lane(1).steps] == [1.0, 5.0]
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


def test_the_return_scale_grows_with_the_returns_it_sees():
    """Episode returns here go from about +/-2 to +57 as the policy improves."""
    agent = _agent(rollout_size=1000)
    small = agent._return_scale.scale

    agent._return_scale.update(torch.full((64,), 40.0))

    assert agent._return_scale.scale > small * 5


def test_stored_trajectory_values_are_invariant_to_later_scale_changes():
    """Changing units after sampling must not rewrite that value estimate."""
    agent = _agent(rollout_size=4)
    for done in (False, False, True):
        _step(agent, 0, reward=1.0, done=done)
    steps = agent._lane(0).steps
    before, _ = agent._advantages_and_returns(steps)

    agent._return_scale.update(torch.full((256,), 30.0))
    after, _ = agent._advantages_and_returns(steps)

    assert agent._return_scale.scale > 5.0
    assert torch.equal(before, after)


def test_a_sampled_value_is_converted_to_reward_units_before_storage():
    agent = _agent(rollout_size=1000)
    agent._return_scale.update(torch.full((256,), 10.0))
    observation = _observation(_map_state(2))
    decision = agent.tokenizer.tokenize_decision(
        observation, agent.action_provider.require_candidates(observation.raw_state)
    )
    with torch.no_grad():
        raw_value = agent.game_encoder.policy_value(decision).value.item()
        raw_value *= agent._return_scale.scale
    _step(agent, 0, reward=1.0, done=False)
    step = agent._lane(0).steps[0]
    assert float(step.old_value) == pytest.approx(raw_value)
    step.done = True
    agent._return_scale.update(torch.full((256,), 100.0))
    advantages, returns = agent._advantages_and_returns([step])
    assert float(advantages[0]) == pytest.approx(1.0 - raw_value, rel=1e-4)
    assert float(returns[0]) == pytest.approx(1.0)


def test_the_scale_survives_a_checkpoint():
    """A resumed critic reading its own predictions at the wrong scale is worse
    than one that was never saved."""
    agent = _agent()
    agent._return_scale.update(torch.full((256,), 12.0))
    saved = agent.checkpoint_state()
    scale = agent._return_scale.scale

    restored = _agent()
    restored.load_checkpoint_state(saved)

    assert restored._return_scale.scale == pytest.approx(scale)


def test_an_untouched_scale_leaves_the_reward_units_alone():
    """Before any update the critic is in the reward's own units."""
    agent = _agent()

    assert agent._return_scale.scale == pytest.approx(1.0)


def test_a_forced_step_never_reaches_back_across_an_episode_boundary():
    """The rollout spans episodes, so the step before a forced one may be a terminal.

    Merging into it erased the only terminal GAE had to anchor on, credited the
    new episode's reward to the old episode's last decision, and replaced its
    next observation with a state from another run.
    """
    torch.manual_seed(39)
    agent = _agent(rollout_size=20)
    agent.update = lambda: {}  # type: ignore[method-assign]
    decision_observation = _observation(_map_state(2))
    terminal = _observation({"state_type": "game_over"})

    action = agent.choose_action(decision_observation)
    agent.observe(
        Transition(
            state=decision_observation,
            action=action,
            reward=1.0,
            next_state=terminal,
            done=True,
        )
    )

    # A new episode whose first decision is forced, which is the usual shape:
    # the first map of an act offers exactly one node.
    next_episode = _observation(_map_state(1))
    agent.reset(next_episode)
    forced_action = agent.choose_action(next_episode)
    agent.observe(
        Transition(
            state=next_episode,
            action=forced_action,
            reward=4.0,
            next_state=_observation(_map_state(2)),
            done=False,
        )
    )

    ended = agent._lane(0).steps[0]
    assert ended.done is True, "the terminal must survive the next episode's forced step"
    assert ended.reward == pytest.approx(1.0), "its reward must stay its own episode's"
    assert ended.next_observation is terminal
    assert agent._lane(0).steps[1].reward == pytest.approx(4.0)
    assert agent._lane(0).steps[1].done is False


def test_an_initial_forced_reward_is_not_added_to_a_later_decision():
    torch.manual_seed(39)
    agent = _agent(rollout_size=20)
    agent.update = lambda: {}  # type: ignore[method-assign]
    first = _observation(_map_state(2))
    terminal = _observation({"state_type": "game_over"})
    action = agent.choose_action(first)
    agent.observe(
        Transition(state=first, action=action, reward=1.0, next_state=terminal, done=True)
    )

    forced = _observation(_map_state(1))
    agent.reset(forced)
    agent.observe(
        Transition(
            state=forced,
            action=agent.choose_action(forced),
            reward=4.0,
            next_state=first,
            done=False,
        )
    )
    agent.observe(
        Transition(
            state=first,
            action=agent.choose_action(first),
            reward=2.0,
            next_state=terminal,
            done=True,
        )
    )

    assert len(agent._lane(0).steps) == 3
    assert [step.reward for step in agent._lane(0).steps] == [1.0, 4.0, 2.0]


def _external_step(agent, index: int, *, reward: float, done: bool) -> None:
    observation = _observation(_map_state(3))
    candidates = agent.action_provider.require_candidates(observation.raw_state)
    agent.choose_external(observation, candidates[index])
    agent.observe(
        Transition(
            state=observation,
            action=candidates[index],
            reward=reward,
            next_state=_observation(_map_state(3)),
            done=done,
        )
    )


def test_an_external_action_is_recorded_for_the_critic_only():
    torch.manual_seed(51)
    agent = _agent(rollout_size=50)
    _external_step(agent, 2, reward=1.5, done=False)

    step = agent._lane(0).steps[0]
    assert step.action_index == 2
    assert step.policy_trainable is False
    assert len(step.decision.actions) == 3
    assert step.reward == 1.5
    assert agent.environment_steps == 1


def test_external_actions_train_the_critic_but_never_the_policy():
    torch.manual_seed(52)
    agent = _agent(rollout_size=4, update_epochs=1)
    for index in range(4):
        _external_step(agent, index % 3, reward=2.0, done=index == 3)

    metrics = agent.last_update
    assert metrics["policy_loss"] == 0.0
    assert metrics["entropy"] == 0.0
    assert metrics["value_loss"] > 0.0


def test_an_external_action_must_be_one_of_the_candidates():
    agent = _agent()
    observation = _observation(_map_state(2))
    foreign = agent.action_provider.require_candidates(_observation(_map_state(3)).raw_state)[2]
    with pytest.raises(ValueError):
        agent.choose_external(observation, foreign)
    assert agent._lane(0).pending is None


def test_a_lane_view_passes_external_actions_to_its_lane():
    agent = _agent()
    view = agent.lane_view(3)
    observation = _observation(_map_state(2))
    action = agent.action_provider.require_candidates(observation.raw_state)[1]
    assert view.choose_external(observation, action).to_dict() == action.to_dict()
    assert agent._lane(3).pending is not None
    assert agent._lane(0).pending is None



def _fill_policy_rollout(agent, size: int) -> None:
    """Sampled map decisions with large, varied rewards, so an update moves the policy."""
    for index in range(size):
        observation = _observation(_map_state(3))
        action = agent.choose_action(observation)
        agent.observe(
            Transition(
                state=observation,
                action=action,
                reward=10.0 if action.params.get("index") == index % 3 else -10.0,
                next_state=_observation(_map_state(3)),
                done=index % 4 == 3,
            )
        )


def test_without_a_kl_limit_every_epoch_and_minibatch_runs():
    torch.manual_seed(61)
    agent = _agent(rollout_size=32, update_epochs=4, minibatch_size=8)
    _fill_policy_rollout(agent, 32)
    metrics = agent.last_update
    assert metrics["optimizer_steps"] == 16
    assert metrics["kl_early_stop"] == 0.0


def test_a_kl_limit_stops_the_update_before_the_step_that_crosses_it():
    torch.manual_seed(61)
    agent = _agent(rollout_size=32, update_epochs=4, minibatch_size=8, target_kl=1e-9)
    _fill_policy_rollout(agent, 32)
    metrics = agent.last_update
    # The first minibatch sees the unchanged policy (KL 0) and steps; the next one
    # measures a moved policy and stops the update there.
    assert metrics["optimizer_steps"] == 1
    assert metrics["kl_early_stop"] == 1.0
    assert metrics["approx_kl"] > 1.5e-9


def test_the_kl_limit_must_be_positive():
    with pytest.raises(ValueError, match="target_kl"):
        PPOConfig(target_kl=0.0)



def _macro_step(agent, lane: int, *, reward: float, done: bool = False) -> None:
    observation = _observation(_map_state(2))
    action = agent.choose_action(observation, lane=lane)
    agent.observe(
        Transition(state=observation, action=action, reward=reward,
                   next_state=_observation(_map_state(2)), done=done),
        lane=lane,
    )


def _unrecorded_fight_step(agent, lane: int, *, reward: float, done: bool = False, marker: int = 0) -> None:
    observation = _observation(_map_state(3))
    action = agent.action_provider.require_candidates(observation.raw_state)[0]
    agent.choose_external(observation, action, lane=lane, record=False)
    agent.observe(
        Transition(state=observation, action=action, reward=reward,
                   next_state=_observation(_map_state(2 + marker)), done=done),
        lane=lane,
    )


def test_an_unrecorded_fight_adds_its_reward_to_the_decision_before_it():
    agent = _agent()
    _macro_step(agent, 0, reward=1.0)
    _unrecorded_fight_step(agent, 0, reward=-0.01)
    _unrecorded_fight_step(agent, 0, reward=10.0, marker=1)
    steps = agent._lane(0).steps
    assert len(steps) == 1
    assert steps[0].reward == pytest.approx(1.0 - 0.01 + 10.0)
    # The decision's transition now ends where the fight ended.
    assert len(steps[0].next_observation.raw_state["map"]["next_options"]) == 3
    assert agent.environment_steps == 3


def test_a_death_in_an_unrecorded_fight_ends_the_decision_before_it():
    agent = _agent()
    _macro_step(agent, 0, reward=1.0)
    _unrecorded_fight_step(agent, 0, reward=-0.01, done=True)
    step = agent._lane(0).steps[0]
    assert step.done and step.episode_end


def test_an_unrecorded_fight_never_extends_a_terminal_and_carries_its_reward_forward():
    agent = _agent()
    _macro_step(agent, 0, reward=1.0, done=True)
    _unrecorded_fight_step(agent, 0, reward=5.0)
    terminal = agent._lane(0).steps[0]
    assert terminal.reward == 1.0 and terminal.done
    _macro_step(agent, 0, reward=2.0)
    assert agent._lane(0).steps[1].reward == pytest.approx(2.0 + 5.0)


def test_unrecorded_fights_stay_in_their_own_lane():
    agent = _agent()
    _macro_step(agent, 0, reward=1.0)
    _macro_step(agent, 1, reward=1.0)
    _unrecorded_fight_step(agent, 1, reward=7.0)
    assert agent._lane(0).steps[0].reward == 1.0
    assert agent._lane(1).steps[0].reward == 8.0


def test_a_rollout_of_only_macro_decisions_fills_with_their_own_choices():
    agent = _agent(rollout_size=4)
    for _ in range(5):
        _macro_step(agent, 0, reward=1.0)
        _unrecorded_fight_step(agent, 0, reward=0.5)
    # Four completed decisions filled the rollout when the fifth closed the
    # fourth; the fights never entered it.  The fifth was held, so its fight
    # reached it rather than the decision after it.
    assert agent.optimizer_updates == 1
    assert agent.last_update["rollout_steps"] == 4.0
    assert agent.last_update["held_steps"] == 1.0
    assert agent.environment_steps == 10
    assert [step.reward for step in agent._lane(0).steps] == [pytest.approx(1.5)]
    assert agent._lane(0).carried_reward == 0.0


def _rewards_seen_by_gae(agent) -> list[list[tuple[float, bool]]]:
    """Record the (reward, done) of every trajectory an update walks."""
    seen: list[list[tuple[float, bool]]] = []
    original = agent._advantages_and_returns

    def spy(steps, tail_value=None):
        seen.append([(step.reward, step.done) for step in steps])
        return original(steps, tail_value=tail_value)

    agent._advantages_and_returns = spy
    return seen


def _fill_while_lane_one_fights(agent) -> None:
    """Lane 1 enters a fight; lane 0 then fills the rollout and ends its episode."""
    _macro_step(agent, 1, reward=1.0)
    _unrecorded_fight_step(agent, 1, reward=0.5)
    for _ in range(3):
        _macro_step(agent, 0, reward=1.0)
    _macro_step(agent, 0, reward=1.0, done=True)


def test_an_update_keeps_the_decision_before_a_running_fight():
    """The rollout fills on one lane while another is mid-fight.

    Training that lane's last decision then trained it without the fight; the
    clear left nothing to fold into, and the rest of the fight -- floors, the
    boss -- was paid to the lane's next decision.
    """
    torch.manual_seed(71)
    agent = _agent(rollout_size=4, hold_open_steps=True)
    _fill_while_lane_one_fights(agent)

    assert agent.optimizer_updates == 1
    assert agent.last_update["rollout_steps"] == 4.0
    assert agent.last_update["held_steps"] == 1.0
    held = agent._lane(1).steps
    assert len(held) == 1 and held[0].reward == pytest.approx(1.5)

    _unrecorded_fight_step(agent, 1, reward=10.0, marker=1)
    _macro_step(agent, 1, reward=2.0)

    rewards = [step.reward for step in agent._lane(1).steps]
    assert rewards == [pytest.approx(11.5), pytest.approx(2.0)]
    assert len(agent._lane(1).steps[0].next_observation.raw_state["map"]["next_options"]) == 3


def test_a_lane_with_no_fight_is_consumed_whole_and_unaffected():
    torch.manual_seed(72)
    agent = _agent(rollout_size=4, hold_open_steps=True)
    seen = _rewards_seen_by_gae(agent)
    _fill_while_lane_one_fights(agent)

    # Lane 0 alone was walked, completely, with its own rewards and terminal.
    assert seen == [[(1.0, False), (1.0, False), (1.0, False), (1.0, True)]]
    assert agent._lane(0).steps == []
    assert agent.last_update["lanes"] == 1.0


def test_a_death_after_the_update_ends_the_held_decision():
    """The death used to be dropped: nothing was left to mark as terminal."""
    torch.manual_seed(73)
    agent = _agent(rollout_size=4, hold_open_steps=True)
    _fill_while_lane_one_fights(agent)

    _unrecorded_fight_step(agent, 1, reward=-0.01, done=True)

    held = agent._lane(1).steps[0]
    assert held.done and held.episode_end
    assert held.reward == pytest.approx(1.0 + 0.5 - 0.01)
    assert agent._lane(1).carried_reward == 0.0


def test_the_held_step_is_trained_next_time_with_its_whole_fight():
    torch.manual_seed(74)
    agent = _agent(rollout_size=4, hold_open_steps=True)
    _fill_while_lane_one_fights(agent)
    _unrecorded_fight_step(agent, 1, reward=10.0)
    _unrecorded_fight_step(agent, 1, reward=-0.01, done=True)
    seen = _rewards_seen_by_gae(agent)

    for _ in range(2):
        _macro_step(agent, 0, reward=1.0)
    _macro_step(agent, 0, reward=1.0, done=True)

    assert agent.optimizer_updates == 2
    assert agent.last_update["rollout_steps"] == 4.0
    assert agent.last_update["held_steps"] == 0.0
    assert seen[0] == [(1.0, False), (1.0, False), (1.0, True)]
    assert seen[1] == [(pytest.approx(1.0 + 0.5 + 10.0 - 0.01), True)]
    assert agent._rollout_length() == 0


def test_the_step_before_a_held_one_bootstraps_from_its_sampled_value():
    """The same value the walk would read with the held step in the batch;
    only the trace stops there, as the held step's reward is not known yet."""
    torch.manual_seed(75)
    agent = _agent(rollout_size=1000, hold_open_steps=True)
    _macro_step(agent, 0, reward=1.0)
    _macro_step(agent, 0, reward=2.0)
    _macro_step(agent, 0, reward=3.0)
    _unrecorded_fight_step(agent, 0, reward=0.5)
    first, second, held = agent._lane(0).steps
    values = [float(step.old_value) for step in (first, second, held)]
    gamma, lam = agent.config.gamma, agent.config.gae_lambda

    advantages, _ = agent._advantages_and_returns([first, second], tail_value=held.old_value)

    expected_second = 2.0 + gamma * values[2] - values[1]
    expected_first = 1.0 + gamma * values[1] - values[0] + gamma * lam * expected_second
    assert float(advantages[1]) == pytest.approx(expected_second, rel=1e-5)
    assert float(advantages[0]) == pytest.approx(expected_first, rel=1e-5)


def test_an_update_with_only_held_steps_trains_nothing():
    agent = _agent(rollout_size=1000, hold_open_steps=True)
    _macro_step(agent, 0, reward=1.0)
    _unrecorded_fight_step(agent, 0, reward=0.5)

    assert agent.update() == {}
    assert agent.optimizer_updates == 0
    assert len(agent._lane(0).steps) == 1


def test_an_episode_end_closes_the_last_step_and_a_pending_decision_does_not():
    agent = _agent(rollout_size=1000, hold_open_steps=True)
    _macro_step(agent, 0, reward=1.0)
    _macro_step(agent, 1, reward=1.0)
    _macro_step(agent, 2, reward=1.0)
    agent.choose_action(_observation(_map_state(2)), lane=0)
    agent.reset(_observation(_map_state(2)), lane=1)

    assert agent._trainable_length() == 1  # Lane 1 alone is closed.
    assert agent.update()["held_steps"] == 2.0
    assert len(agent._lane(0).steps) == 1
    assert len(agent._lane(2).steps) == 1


def test_a_discarded_decision_then_a_fight_still_folds_into_the_held_step():
    """A pending decision can be refused and discarded; the lane may then fight.

    Had the pending decision closed the step, the update would have taken it,
    and the fight's reward would have gone to the next decision, its death
    nowhere.
    """
    agent = _agent(rollout_size=1000, hold_open_steps=True)
    _macro_step(agent, 0, reward=1.0)
    agent.choose_action(_observation(_map_state(2)), lane=0)
    agent.update()
    agent.discard_decision(lane=0)
    _unrecorded_fight_step(agent, 0, reward=0.5, done=True)

    steps = agent._lane(0).steps
    assert len(steps) == 1
    assert steps[0].reward == pytest.approx(1.5)
    assert steps[0].done and steps[0].episode_end
    assert agent._lane(0).carried_reward == 0.0


def test_two_lanes_that_both_see_a_full_rollout_update_once():
    """The threshold is checked again under the lock that selects the steps."""
    agent = _agent(rollout_size=2)
    hooks: list[int] = []
    agent.on_update = lambda: hooks.append(agent.optimizer_updates)
    _macro_step(agent, 0, reward=1.0)
    _macro_step(agent, 1, reward=1.0)  # Fills the rollout: update 1.

    # A second lane that judged the rollout full before the first update ran.
    agent._update_if_ready()

    assert agent.optimizer_updates == 1
    assert hooks == [1]


def test_a_count_that_drops_below_the_threshold_skips_the_update():
    agent = _agent(rollout_size=1000, hold_open_steps=True)
    hooks: list[int] = []
    agent.on_update = lambda: hooks.append(agent.optimizer_updates)
    _macro_step(agent, 0, reward=1.0)
    _macro_step(agent, 0, reward=1.0, done=True)
    _macro_step(agent, 1, reward=1.0)
    _macro_step(agent, 1, reward=1.0)
    agent.config = dataclasses.replace(agent.config, rollout_size=3)
    assert agent._trainable_length() == 3  # The second lane judges it full ...
    agent.abort_lane(0)  # ... and before its update, the first lane's work goes.

    agent._update_if_ready()

    assert agent.optimizer_updates == 0
    assert hooks == []


def test_a_truncation_that_completes_the_rollout_updates():
    """Otherwise a run that ends here saves a final checkpoint without the batch."""
    agent = _agent(rollout_size=4, hold_open_steps=True)
    hooks: list[int] = []
    agent.on_update = lambda: hooks.append(agent.optimizer_updates)
    for _ in range(4):
        _macro_step(agent, 0, reward=1.0)
    assert agent._trainable_length() == 3
    assert agent.optimizer_updates == 0

    agent.finish_episode(_observation(_map_state(2)), truncated=True)

    assert agent.optimizer_updates == 1
    assert agent.last_update["rollout_steps"] == 4.0
    assert hooks == [1]


def test_a_reset_that_completes_the_rollout_updates():
    agent = _agent(rollout_size=4, hold_open_steps=True)
    for _ in range(4):
        _macro_step(agent, 0, reward=1.0)

    agent.reset(_observation(_map_state(2)))

    assert agent.optimizer_updates == 1
    assert agent._rollout_length() == 0


def test_a_kl_stop_on_the_first_minibatch_still_reports_the_batch_size():
    torch.manual_seed(77)
    agent = _agent(rollout_size=1000, update_epochs=2, minibatch_size=4, target_kl=1e-3)
    for _ in range(6):
        _macro_step(agent, 0, reward=1.0, done=True)
    # Pretend the rollout came from a far-away policy: every ratio is e.
    for step in agent._lane(0).steps:
        step.old_log_probability = step.old_log_probability - 1.0

    metrics = agent.update()

    assert metrics["optimizer_steps"] == 0.0
    assert metrics["kl_early_stop"] == 1.0
    assert metrics["rollout_steps"] == 6.0


def test_a_held_step_survives_more_than_one_update():
    torch.manual_seed(78)
    agent = _agent(rollout_size=2, hold_open_steps=True)
    _macro_step(agent, 1, reward=1.0)
    _unrecorded_fight_step(agent, 1, reward=0.5)
    held = agent._lane(1).steps[0]
    for _ in range(2):
        _macro_step(agent, 0, reward=1.0)
        _macro_step(agent, 0, reward=1.0, done=True)
        _unrecorded_fight_step(agent, 1, reward=1.0)

    assert agent.optimizer_updates == 2
    assert agent._lane(1).steps == [held]
    assert held.reward == pytest.approx(1.0 + 0.5 + 2.0)


def test_a_held_value_keeps_its_reward_units_across_a_scale_change():
    torch.manual_seed(79)
    agent = _agent(rollout_size=2, hold_open_steps=True)
    _macro_step(agent, 1, reward=1.0)
    _unrecorded_fight_step(agent, 1, reward=0.5)
    held = agent._lane(1).steps[0]
    value = held.old_value.clone()
    scale = agent._return_scale.scale

    _macro_step(agent, 0, reward=40.0)
    _macro_step(agent, 0, reward=40.0, done=True)

    assert agent._return_scale.scale > scale * 5
    assert torch.equal(held.old_value, value)


def test_initializing_refuses_an_agent_with_a_held_step():
    agent = _agent(rollout_size=1000, hold_open_steps=True)
    _macro_step(agent, 0, reward=1.0)
    _unrecorded_fight_step(agent, 0, reward=0.5)
    agent.update()
    saved = agent.checkpoint_state()

    with pytest.raises(RuntimeError, match="non-empty rollout"):
        agent.initialize_from(saved["encoder"], saved["return_scale"])


def test_an_update_bootstraps_the_trained_prefix_from_the_held_value():
    agent = _agent(rollout_size=1000, hold_open_steps=True)
    _macro_step(agent, 0, reward=1.0)
    _macro_step(agent, 0, reward=2.0)
    _unrecorded_fight_step(agent, 0, reward=0.5)
    held = agent._lane(0).steps[-1]
    tails: list[object] = []
    original = agent._advantages_and_returns

    def spy(steps, tail_value=None):
        tails.append(tail_value)
        return original(steps, tail_value=tail_value)

    agent._advantages_and_returns = spy
    agent.update()

    assert len(tails) == 1 and tails[0] is held.old_value


def test_an_unrecorded_step_turns_holding_on():
    agent = _agent(rollout_size=1000)
    assert agent.hold_open_steps is False
    _macro_step(agent, 0, reward=1.0)
    _unrecorded_fight_step(agent, 0, reward=0.5)
    assert agent.hold_open_steps is True


def test_a_held_step_is_work_in_flight_for_loading_and_is_lost_on_abort():
    agent = _agent(rollout_size=1000, hold_open_steps=True)
    _macro_step(agent, 0, reward=1.0)
    _unrecorded_fight_step(agent, 0, reward=0.5)
    agent.update()
    saved = agent.checkpoint_state()  # Saving reads only; the held step stays.
    assert len(agent._lane(0).steps) == 1

    with pytest.raises(RuntimeError, match="non-empty rollout"):
        agent.load_checkpoint_state(saved)
    agent.abort_lane(0)
    agent.load_checkpoint_state(saved)


@pytest.mark.parametrize("hold", [False, True])
def test_with_every_last_step_closed_holding_changes_nothing(hold: bool):
    """No fight in progress: the same batch, the same targets, the same weights."""

    def run(hold_open_steps: bool):
        torch.manual_seed(76)
        agent = _agent(rollout_size=1000, hold_open_steps=hold_open_steps)
        walked: list[tuple[torch.Tensor, torch.Tensor]] = []
        original = agent._advantages_and_returns

        def spy(steps, tail_value=None):
            walked.append(original(steps, tail_value=tail_value))
            return walked[-1]

        agent._advantages_and_returns = spy
        _macro_step(agent, 0, reward=1.0)
        _macro_step(agent, 0, reward=2.0, done=True)
        _macro_step(agent, 1, reward=3.0)
        # Lane 1's last step is closed by a truncation.
        agent.finish_episode(_observation(_map_state(2)), truncated=True, lane=1)
        return agent, agent.update(), walked

    reference, reference_metrics, reference_walked = run(False)
    agent, metrics, walked = run(hold)

    assert metrics["held_steps"] == 0.0
    assert metrics == reference_metrics
    for (advantages, targets), (expected_advantages, expected_targets) in zip(
        walked, reference_walked, strict=True
    ):
        assert torch.equal(advantages, expected_advantages)
        assert torch.equal(targets, expected_targets)
    for name, tensor in reference.game_encoder.state_dict().items():
        assert torch.equal(agent.game_encoder.state_dict()[name], tensor), name


def _node(index: int) -> dict[str, object]:
    return {"type": "choose_map_node", "index": index}


def test_an_excluded_action_is_never_sampled_and_the_decision_is_the_reduced_set():
    """The stored log probability must be over what was really offered.

    The update re-encodes the stored decision, so recording the reduced set
    makes the ratio compare the same distribution at both ends.
    """
    torch.manual_seed(80)
    agent = _agent(rollout_size=100)
    observation = _observation(_map_state(3))
    refused = GameAction("choose_map_node", index=1)

    for _ in range(20):
        action = agent.choose_action(observation, exclude=[refused])
        pending = agent._lanes[0].pending
        assert action.to_dict() != _node(1)
        assert len(pending.decision.actions) == 2
        offered = [_node(0), _node(2)]
        assert offered[pending.action_index] == action.to_dict()
        with torch.no_grad():
            logits = agent.game_encoder.policy_value(pending.decision).logits
        expected = torch.log_softmax(logits, dim=-1)[pending.action_index]
        assert torch.allclose(pending.log_probability, expected, atol=1e-6)
        agent.discard_decision()


def test_a_decision_made_under_an_exclusion_trains_like_any_other():
    torch.manual_seed(81)
    agent = _agent(rollout_size=2)
    observation = _observation(_map_state(3))
    for exclude in ([GameAction("choose_map_node", index=0)], []):
        action = agent.choose_action(observation, exclude=exclude)
        agent.observe(
            Transition(
                state=observation,
                action=action,
                reward=1.0,
                next_state=_observation(_map_state(3)),
                done=False,
            )
        )

    assert agent.optimizer_updates == 1


def test_a_set_reduced_to_one_candidate_is_a_forced_step():
    torch.manual_seed(82)
    agent = _agent()
    observation = _observation(_map_state(2))

    action = agent.choose_action(
        observation, exclude=[GameAction("choose_map_node", index=0)]
    )

    assert action.to_dict() == _node(1)
    pending = agent._lanes[0].pending
    assert len(pending.decision.actions) == 1
    assert float(pending.log_probability) == 0.0


def test_excluding_every_candidate_raises_and_leaves_nothing_pending():
    """The runner reads this as "nothing new to try" and asks without it."""
    agent = _agent()
    observation = _observation(_map_state(2))
    everything = [GameAction("choose_map_node", index=i) for i in range(2)]

    with pytest.raises(NoLegalActionsError):
        agent.choose_action(observation, exclude=everything)

    assert agent._lanes[0].pending is None
    agent.choose_action(observation)


def test_evaluation_takes_the_best_candidate_that_is_left():
    agent = _agent()
    agent.eval()
    observation = _observation(_map_state(3))
    best = agent.choose_action(observation)

    second = agent.choose_action(observation, exclude=[best])

    assert second.to_dict() != best.to_dict()
    assert agent.choose_action(observation, exclude=()).to_dict() == best.to_dict()


def test_no_exclusion_is_the_same_choice_as_before():
    observation = _observation(_map_state(3))
    choices = []
    for exclude in (None, ()):
        torch.manual_seed(83)
        agent = _agent()
        torch.manual_seed(84)
        if exclude is None:
            action = agent.choose_action(observation)
        else:
            action = agent.choose_action(observation, exclude=exclude)
        pending = agent._lanes[0].pending
        choices.append((action.to_dict(), pending.action_index, pending.log_probability))

    assert choices[0][:2] == choices[1][:2]
    assert torch.equal(choices[0][2], choices[1][2])


def test_a_lane_view_passes_the_exclusion_through():
    torch.manual_seed(85)
    agent = _agent()
    view = agent.lane_view(3)
    observation = _observation(_map_state(2))

    action = view.choose_action(
        observation, exclude=[GameAction("choose_map_node", index=1)]
    )

    assert action.to_dict() == _node(0)
    assert len(agent._lanes[3].pending.decision.actions) == 1


def _explored_decision(agent, observation, *, explored: bool, lane: int = 0, exclude=()):
    """Choose until the pending decision came from the wanted branch of the mixture."""
    for _ in range(200):
        action = agent.choose_action(observation, lane=lane, exclude=exclude)
        if agent._lane(lane).pending.explored is explored:
            return action
        agent.discard_decision(lane=lane)
    raise AssertionError(f"no decision with explored={explored} in 200 draws")


def _log_policy(agent, decision: TokenizedDecision, action_index: int) -> torch.Tensor:
    """log pi(a) under the agent's current weights."""
    with torch.no_grad():
        logits = agent.game_encoder.policy_value(decision).logits
    return torch.log_softmax(logits, dim=-1)[action_index]


def _reference_run(exploration=(), **options):
    """A seeded rollout of eight map decisions, and the one update it fills.

    ``options`` go to ``_agent`` over the defaults here: a reference policy
    and its coefficients, or a minibatch size, for the runs pinned beside the
    one without them.
    """
    torch.manual_seed(90)
    agent = _agent(
        **{
            "rollout_size": 8, "update_epochs": 1, "minibatch_size": 8,
            "exploration": exploration, **options,
        }
    )
    actions = []
    for index in range(8):
        observation = _observation(_map_state(3))
        action = agent.choose_action(observation)
        actions.append(action.params["index"])
        agent.observe(
            Transition(
                state=observation,
                action=action,
                reward=10.0 if action.params["index"] == index % 3 else -10.0,
                next_state=_observation(_map_state(3)),
                done=index % 4 == 3,
            )
        )
    return agent, actions


def test_without_exploration_the_path_is_the_one_from_before_it_existed():
    """Pinned from the code before exploration existed, on the same seed: the
    actions, one update's losses, the weights, and the random numbers left."""
    agent, actions = _reference_run(())

    assert actions == [0, 0, 2, 1, 1, 0, 1, 2]
    metrics = agent.last_update
    assert metrics["loss"] == pytest.approx(0.5453976988792419, rel=1e-5)
    assert metrics["value_loss"] == pytest.approx(1.1118905544281006, rel=1e-5)
    assert metrics["entropy"] == pytest.approx(1.0547552108764648, rel=1e-5)
    assert metrics["gradient_norm"] == pytest.approx(4.400908946990967, rel=1e-5)
    assert metrics["approx_kl"] == 0.0
    assert metrics["explored_steps"] == 0.0
    assert metrics["explored_positive_advantage"] == 0.0
    weights = sum(float(p.detach().sum()) for p in agent.game_encoder.parameters())
    assert weights == pytest.approx(-65.32838867467945, rel=1e-5)
    # The generator moved exactly as far as the eight samples took it.
    assert torch.rand(2).tolist() == pytest.approx([0.817125678062439, 0.0004968047142028809])


def test_a_rate_on_a_screen_never_visited_changes_nothing():
    """Nothing is drawn for a screen whose rate is zero, so the run is the same."""
    reference, reference_actions = _reference_run(())
    reference_rng = torch.get_rng_state()
    agent, actions = _reference_run({"rest_site": 0.3})

    assert actions == reference_actions
    assert torch.equal(torch.get_rng_state(), reference_rng)
    for key, value in reference.last_update.items():
        assert agent.last_update[key] == value, key
    assert agent.last_update["explore/rest_site"] == 0.0
    for name, tensor in reference.game_encoder.state_dict().items():
        assert torch.equal(agent.game_encoder.state_dict()[name], tensor), name


def test_an_explored_screen_stores_the_mixture_and_the_policy_apart():
    """log mu_old(a) is what the ratio starts from; log pi_old(a) feeds policy_kl."""
    torch.manual_seed(91)
    agent = _agent(exploration={"map": 0.3})
    observation = _observation(_map_state(3))

    for explored in (False, True):
        _explored_decision(agent, observation, explored=explored)
        pending = agent._lane(0).pending
        log_pi = _log_policy(agent, pending.decision, pending.action_index)
        assert pending.explored is explored
        assert pending.epsilon == 0.3
        assert pending.state_type == "map"
        assert torch.allclose(pending.policy_log_probability, log_pi, atol=1e-6)
        expected = torch.log(0.7 * log_pi.exp() + 0.3 / 3)
        assert torch.allclose(pending.log_probability, expected, atol=1e-6)
        agent.discard_decision()


def test_a_screen_without_a_rate_stores_the_policy_alone():
    torch.manual_seed(92)
    agent = _agent(exploration={"rest_site": 0.3})
    observation = _observation(_map_state(3))

    agent.choose_action(observation)

    pending = agent._lane(0).pending
    assert pending.explored is False
    assert pending.epsilon == 0.0
    assert torch.equal(pending.log_probability, pending.policy_log_probability)
    assert torch.allclose(
        pending.log_probability,
        _log_policy(agent, pending.decision, pending.action_index),
        atol=1e-6,
    )


def test_an_update_starts_at_ratio_one_on_explored_steps():
    torch.manual_seed(93)
    agent = _agent(rollout_size=1000, update_epochs=1, minibatch_size=64, exploration={"map": 0.5})
    for index in range(8):
        observation = _observation(_map_state(3))
        action = _explored_decision(agent, observation, explored=index % 2 == 0)
        agent.observe(
            Transition(
                state=observation,
                action=action,
                reward=float(index),
                next_state=_observation(_map_state(3)),
                done=index == 7,
            )
        )

    for step in agent._lane(0).steps:
        new = _mixture_log_probability(
            _log_policy(agent, step.decision, step.action_index),
            step.epsilon,
            len(step.decision.actions),
        )
        assert float(torch.exp(new - step.old_log_probability)) == pytest.approx(1.0, abs=1e-6)
    metrics = agent.update()

    assert metrics["optimizer_steps"] == 1.0
    assert metrics["approx_kl"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["policy_kl"] == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("reward", [50.0, -50.0])
def test_an_explored_step_moves_the_policy_the_way_its_advantage_points(reward: float):
    """The mixture's gradient reaches the policy through (1 - epsilon) * pi."""
    torch.manual_seed(94)
    agent = _agent(rollout_size=1000, update_epochs=1, minibatch_size=8, exploration={"map": 0.5})
    # The policy term alone, so the direction is the surrogate's and nothing else's.
    agent.config = dataclasses.replace(
        agent.config, value_coefficient=0.0, entropy_coefficient=0.0
    )
    observation = _observation(_map_state(3))
    action = _explored_decision(agent, observation, explored=True)
    agent.observe(
        Transition(
            state=observation,
            action=action,
            reward=reward,
            next_state=_observation({"state_type": "game_over"}),
            done=True,
        )
    )
    step = agent._lane(0).steps[0]
    before = float(_log_policy(agent, step.decision, step.action_index))

    agent.update()

    after = float(_log_policy(agent, step.decision, step.action_index))
    assert (after > before) == (reward > 0)


def _collected_by(step, pi_old: float, epsilon: float) -> tuple[float, float]:
    """Rewrite a stored step as if a policy giving its action ``pi_old`` had
    collected it, through the mixture the sampler really draws from.

    Returns ``(pi_old, mu_old)``, the pair the update's ratios start from.
    """
    mu_old = (1.0 - epsilon) * pi_old + epsilon / len(step.decision.actions)
    step.old_policy_log_probability = torch.tensor(math.log(pi_old))
    step.old_log_probability = torch.tensor(math.log(mu_old))
    return pi_old, mu_old


def _policy_probability(agent, step) -> float:
    """pi(a) under the agent's current weights, for a stored step."""
    return math.exp(float(_log_policy(agent, step.decision, step.action_index)))


def _policy_kl_term(pi_old: float, mu_old: float, pi_new: float) -> float:
    """w * ((q - 1) - log q): the importance-weighted estimator, by hand."""
    weight, ratio = pi_old / mu_old, pi_new / pi_old
    return weight * ((ratio - 1.0) - math.log(ratio))


def test_policy_kl_weights_each_step_by_the_policy_share_of_its_sample():
    """KL(pi_old || pi_new) estimated from mixture samples: each term carries
    w = pi_old(a) / mu_old(a), below 1 for an action the uniform branch
    supplied more of than the policy did, above 1 for one it supplied less."""
    torch.manual_seed(95)
    epsilon, pi_old = 0.5, (0.7, 0.2, 0.1)
    agent = _agent(rollout_size=1000, update_epochs=1, minibatch_size=64, exploration={"map": epsilon})
    observation = _observation(_map_state(3))
    for index in range(4):
        action = agent.choose_action(observation)
        agent.observe(
            Transition(state=observation, action=action, reward=1.0, next_state=observation, done=index == 3)
        )
    expected = []
    for step in agent._lane(0).steps:
        _, mu_old = _collected_by(step, pi_old[step.action_index], epsilon)
        expected.append(
            _policy_kl_term(pi_old[step.action_index], mu_old, _policy_probability(agent, step))
        )

    metrics = agent.update()

    assert metrics["policy_kl"] == pytest.approx(sum(expected) / len(expected), rel=1e-4)


def test_policy_kl_stays_finite_when_the_policy_had_abandoned_the_action():
    """w = pi_old / mu_old underflows and q = pi_new / pi_old overflows in
    float32 once the policy has recovered an action it had given 1e-52; taken
    apart they make 0 * inf, and the estimator must not report NaN for a real
    number."""
    torch.manual_seed(104)
    epsilon = 0.5
    agent = _agent(rollout_size=1000, update_epochs=1, minibatch_size=64, exploration={"map": epsilon})
    observation = _observation(_map_state(3))
    action = _explored_decision(agent, observation, explored=True)
    agent.observe(
        Transition(state=observation, action=action, reward=1.0, next_state=observation, done=True)
    )
    step = agent._lane(0).steps[0]
    pi_old, mu_old = _collected_by(step, math.exp(-120.0), epsilon)
    pi_new = _policy_probability(agent, step)
    expected = _policy_kl_term(pi_old, mu_old, pi_new)
    # Nearly all of the term is pi_new(a) / mu_old(a); the rest is 1e-50.
    assert expected == pytest.approx(pi_new / mu_old, rel=1e-6)

    metrics = agent.update()

    assert math.isfinite(metrics["policy_kl"])
    assert metrics["policy_kl"] == pytest.approx(expected, rel=1e-4)


def test_without_exploration_the_policy_kl_is_the_approximate_kl():
    """Every weight is exactly one and both ratios are the policy's own."""
    torch.manual_seed(96)
    agent = _agent(rollout_size=8, update_epochs=2, minibatch_size=8)
    _fill_policy_rollout(agent, 8)

    assert agent.last_update["approx_kl"] > 0.0
    assert agent.last_update["policy_kl"] == agent.last_update["approx_kl"]


def test_a_held_step_keeps_its_exploration_across_an_update():
    torch.manual_seed(97)
    agent = _agent(rollout_size=4, hold_open_steps=True, exploration={"map": 0.5})
    observation = _observation(_map_state(3))
    action = _explored_decision(agent, observation, explored=True, lane=1)
    agent.observe(
        Transition(state=observation, action=action, reward=1.0, next_state=observation, done=False),
        lane=1,
    )
    _unrecorded_fight_step(agent, 1, reward=0.5)
    held = agent._lane(1).steps[0]

    for _ in range(3):
        _macro_step(agent, 0, reward=1.0)
    _macro_step(agent, 0, reward=1.0, done=True)

    assert agent.optimizer_updates == 1
    assert agent._lane(1).steps == [held]
    assert held.explored is True
    assert held.epsilon == 0.5
    assert held.state_type == "map"


def test_discarding_or_aborting_drops_the_exploration_with_the_decision():
    torch.manual_seed(98)
    agent = _agent(exploration={"map": 0.5})
    observation = _observation(_map_state(3))

    _explored_decision(agent, observation, explored=True)
    agent.discard_decision()
    assert agent._lane(0).pending is None

    action = _explored_decision(agent, observation, explored=True)
    agent.observe(
        Transition(state=observation, action=action, reward=1.0, next_state=observation, done=False)
    )
    assert agent._lane(0).steps[0].explored is True
    _explored_decision(agent, observation, explored=True)
    agent.abort_lane(0)
    assert agent._lane(0).pending is None
    assert agent._lane(0).steps == []


@pytest.mark.parametrize("reward", [100.0, -100.0])
def test_exploration_metrics_count_only_the_steps_the_update_trained(reward: float):
    torch.manual_seed(99)
    agent = _agent(
        rollout_size=1000, hold_open_steps=True, exploration={"map": 0.5, "rest_site": 0.3}
    )
    observation = _observation(_map_state(3))
    # Lane 1: an explored decision, then a fight that holds it open.
    action = _explored_decision(agent, observation, explored=True, lane=1)
    agent.observe(
        Transition(state=observation, action=action, reward=1.0, next_state=observation, done=False),
        lane=1,
    )
    _unrecorded_fight_step(agent, 1, reward=0.5)
    # Lane 0: three explored decisions paid ``reward`` and two the policy drew
    # paid nothing, each its own episode so an advantage is its own reward.
    for explored in (True, False, True, True, False):
        action = _explored_decision(agent, observation, explored=explored)
        agent.observe(
            Transition(
                state=observation,
                action=action,
                reward=reward if explored else 0.0,
                next_state=observation,
                done=True,
            )
        )

    metrics = agent.update()

    assert metrics["rollout_steps"] == 5.0
    assert metrics["held_steps"] == 1.0
    assert metrics["explored_steps"] == 3.0
    assert metrics["explore/map"] == 3.0
    assert metrics["explore/rest_site"] == 0.0
    assert "explore/card_reward" not in metrics
    assert metrics["explored_positive_advantage"] == (1.0 if reward > 0 else 0.0)


def test_evaluation_never_explores_and_draws_nothing():
    torch.manual_seed(100)
    agent = _agent(exploration={"map": 0.9})
    agent.eval()
    observation = _observation(_map_state(3))

    torch.manual_seed(101)
    choices = {agent.choose_action(observation).to_dict()["index"] for _ in range(10)}
    drawn = torch.rand(())
    torch.manual_seed(101)

    assert len(choices) == 1
    assert torch.equal(drawn, torch.rand(()))
    assert agent._lane(0).pending is None


def test_exploration_draws_over_the_reduced_candidate_set():
    """N and both stored probabilities refer to what was really offered."""
    torch.manual_seed(102)
    agent = _agent(exploration={"map": 0.5})
    observation = _observation(_map_state(3))
    refused = GameAction("choose_map_node", index=1)

    for explored in (True, False):
        action = _explored_decision(agent, observation, explored=explored, exclude=[refused])
        pending = agent._lane(0).pending
        assert action.to_dict() != _node(1)
        assert len(pending.decision.actions) == 2
        log_pi = _log_policy(agent, pending.decision, pending.action_index)
        assert torch.allclose(pending.policy_log_probability, log_pi, atol=1e-6)
        expected = torch.log(0.5 * log_pi.exp() + 0.5 / 2)
        assert torch.allclose(pending.log_probability, expected, atol=1e-6)
        agent.discard_decision()

    # Reduced to one candidate it is a forced step, with nothing to explore.
    agent.choose_action(observation, exclude=[GameAction("choose_map_node", index=0), refused])
    pending = agent._lane(0).pending
    assert pending.explored is False
    assert pending.epsilon == 0.0
    assert float(pending.log_probability) == 0.0


def test_the_uniform_branch_reaches_every_candidate():
    torch.manual_seed(103)
    agent = _agent(exploration={"map": 0.9})
    observation = _observation(_map_state(3))

    seen = set()
    for _ in range(60):
        seen.add(_explored_decision(agent, observation, explored=True).to_dict()["index"])
        agent.discard_decision()

    assert seen == {0, 1, 2}


@pytest.mark.parametrize(
    "exploration",
    [
        {"lobby": 0.3},
        (("map", 0.1), ("map", 0.2)),
        {"map": 1.0},
        {"map": -0.1},
        {"map": float("nan")},
        {"map": float("inf")},
    ],
)
def test_an_exploration_rate_names_a_known_screen_once_with_a_rate_below_one(exploration):
    with pytest.raises(ValueError, match="exploration"):
        PPOConfig(exploration=exploration)


def test_exploration_rates_are_normalised_so_order_and_spelling_do_not_count():
    assert PPOConfig(exploration={"map": 0.15, "rest_site": 0.3}) == PPOConfig(
        exploration=[["rest_site", 0.3], ["map", 0.15]]
    )
    config = PPOConfig(exploration={"rest_site": 0.3, "map": 0.15})
    assert config.exploration == (("map", 0.15), ("rest_site", 0.3))
    assert config.exploration_rate("map") == 0.15
    assert config.exploration_rate("card_reward") == 0.0
    assert PPOConfig().exploration == ()


def test_a_vanishing_rate_still_gives_a_finite_mixture_log_probability():
    """epsilon / count underflows to zero for the smallest rate the config
    accepts, and log(0) would raise where log(epsilon) - log(count) is finite."""
    log_probability = _mixture_log_probability(torch.tensor(-1.0), 5e-324, 3)
    assert math.isfinite(float(log_probability))
    assert float(log_probability) == pytest.approx(-1.0)

    torch.manual_seed(105)
    agent = _agent(exploration={"map": 5e-324})
    agent.choose_action(_observation(_map_state(3)))
    pending = agent._lane(0).pending
    assert pending.epsilon == 5e-324
    assert math.isfinite(float(pending.log_probability))


def test_the_ratio_after_a_step_is_the_mixture_at_the_new_logits_over_the_stored_one():
    """One optimizer step moves the logits; the rollout keeps mu_old. Measured
    again without stepping, every ratio is mu_new(a) / mu_old(a) with mu_new
    formed from the new logits, and policy_kl is the policy's own movement."""
    torch.manual_seed(106)
    epsilon = 0.5
    agent = _agent(
        rollout_size=1000, update_epochs=1, minibatch_size=64,
        exploration={"map": epsilon}, learning_rate=0.02,
    )
    observation = _observation(_map_state(3))
    for index in range(6):
        action = _explored_decision(agent, observation, explored=index % 2 == 0)
        agent.observe(
            Transition(
                state=observation,
                action=action,
                reward=10.0 if index % 2 else -10.0,
                next_state=observation,
                done=index == 5,
            )
        )
    steps = list(agent._lane(0).steps)
    agent.update()
    # The same rollout again, under a limit the moved policy cannot meet: the
    # first minibatch measures it and stops before stepping.
    agent.config = dataclasses.replace(agent.config, target_kl=1e-12)
    agent._lane(0).steps.extend(steps)

    metrics = agent.update()

    assert metrics["kl_early_stop"] == 1.0 and metrics["optimizer_steps"] == 0.0
    ratios, terms = [], []
    for step in steps:
        pi_new = _policy_probability(agent, step)
        mu_new = (1.0 - epsilon) * pi_new + epsilon / len(step.decision.actions)
        pi_old = math.exp(float(step.old_policy_log_probability))
        mu_old = math.exp(float(step.old_log_probability))
        ratios.append(mu_new / mu_old)
        terms.append(_policy_kl_term(pi_old, mu_old, pi_new))
    assert all(abs(ratio - 1.0) > 1e-3 for ratio in ratios)
    assert metrics["approx_kl"] == pytest.approx(
        sum((r - 1.0) - math.log(r) for r in ratios) / len(ratios), rel=1e-3
    )
    assert metrics["policy_kl"] == pytest.approx(sum(terms) / len(terms), rel=1e-3)


@pytest.mark.parametrize(
    ("ratio", "reward", "moves"),
    [(1.5, 50.0, False), (1.5, -50.0, True), (0.5, -50.0, False), (0.5, 50.0, True)],
)
def test_the_clip_binds_on_the_mixture_ratio(ratio: float, reward: float, moves: bool):
    """Past 1 +/- clip_ratio on the side the advantage favours, the surrogate
    is the clipped constant: no gradient, and the step moves nothing."""
    torch.manual_seed(107)
    agent = _agent(rollout_size=1000, update_epochs=1, minibatch_size=8, exploration={"map": 0.5})
    # The policy term alone, so nothing else can move a weight.
    agent.config = dataclasses.replace(
        agent.config, value_coefficient=0.0, entropy_coefficient=0.0
    )
    observation = _observation(_map_state(3))
    action = _explored_decision(agent, observation, explored=True)
    agent.observe(
        Transition(
            state=observation,
            action=action,
            reward=reward,
            next_state=_observation({"state_type": "game_over"}),
            done=True,
        )
    )
    step = agent._lane(0).steps[0]
    # The policy has not moved, so mu_new is the stored mu_old: shifting the
    # stored value makes the update see exactly ``ratio``.
    step.old_log_probability = step.old_log_probability - math.log(ratio)
    before = {name: tensor.clone() for name, tensor in agent.game_encoder.state_dict().items()}

    agent.update()

    changed = any(
        not torch.equal(agent.game_encoder.state_dict()[name], tensor)
        for name, tensor in before.items()
    )
    assert changed is moves


@pytest.mark.parametrize("target_kl", [0.02, 0.2])
def test_the_kl_stop_reads_the_mixture_ratio_while_policy_kl_reads_the_policy(target_kl: float):
    """Four times likelier under the current policy than under the one that
    collected it, the action's mixture ratio moved far less than its policy
    ratio: the stop fires on the former, the metric reports the latter."""
    torch.manual_seed(108)
    epsilon = 0.5
    agent = _agent(
        rollout_size=1000, update_epochs=1, minibatch_size=8,
        target_kl=target_kl, exploration={"map": epsilon},
    )
    observation = _observation(_map_state(3))
    action = _explored_decision(agent, observation, explored=True)
    agent.observe(
        Transition(state=observation, action=action, reward=1.0, next_state=observation, done=True)
    )
    step = agent._lane(0).steps[0]
    pi_new = _policy_probability(agent, step)
    pi_old, mu_old = _collected_by(step, pi_new / 4.0, epsilon)
    mu_new = (1.0 - epsilon) * pi_new + epsilon / 3
    mixture = (mu_new / mu_old - 1.0) - math.log(mu_new / mu_old)
    policy = _policy_kl_term(pi_old, mu_old, pi_new)

    metrics = agent.update()

    assert metrics["approx_kl"] == pytest.approx(mixture, rel=1e-4)
    assert metrics["policy_kl"] == pytest.approx(policy, rel=1e-4)
    assert policy > 1.5 * target_kl, "the policy's own movement is over the limit either way"
    assert metrics["kl_early_stop"] == float(mixture > 1.5 * target_kl)
    assert metrics["kl_early_stop"] == (1.0 if target_kl == 0.02 else 0.0)
    assert metrics["optimizer_steps"] == 1.0 - metrics["kl_early_stop"]


def test_an_update_uses_the_reduced_candidate_count_of_an_excluded_decision():
    torch.manual_seed(109)
    epsilon = 0.5
    agent = _agent(rollout_size=1000, update_epochs=1, minibatch_size=8, exploration={"map": epsilon})
    observation = _observation(_map_state(3))
    refused = GameAction("choose_map_node", index=1)
    action = _explored_decision(agent, observation, explored=True, exclude=[refused])
    agent.observe(
        Transition(state=observation, action=action, reward=1.0, next_state=observation, done=True)
    )
    step = agent._lane(0).steps[0]
    assert len(step.decision.actions) == 2
    # Formed over the full set of three instead, the unmoved policy would not
    # give ratio 1, so a zero KL below says N was the reduced count.
    log_pi = _log_policy(agent, step.decision, step.action_index)
    wrong = float(torch.exp(_mixture_log_probability(log_pi, epsilon, 3) - step.old_log_probability))
    assert abs(wrong - 1.0) > 1e-3

    metrics = agent.update()

    assert metrics["approx_kl"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["policy_kl"] == pytest.approx(0.0, abs=1e-7)


def test_decisions_crossing_an_update_are_trained_with_their_original_probabilities():
    """A pending decision and a held open step both outlive an update. The next
    one trains them against the probabilities they were sampled under -- so
    their ratio is no longer 1 there, and the estimator stays finite."""
    torch.manual_seed(110)
    agent = _agent(rollout_size=4, hold_open_steps=True, exploration={"map": 0.5})
    observation = _observation(_map_state(3))
    # Lane 1: an explored decision awaiting its observation.
    pending_action = _explored_decision(agent, observation, explored=True, lane=1)
    pending = agent._lane(1).pending
    sampled = (pending.log_probability.clone(), pending.policy_log_probability.clone())
    # Lane 2: an explored decision, then a fight that holds it open.
    held_action = _explored_decision(agent, observation, explored=True, lane=2)
    agent.observe(
        Transition(state=observation, action=held_action, reward=1.0, next_state=observation, done=False),
        lane=2,
    )
    _unrecorded_fight_step(agent, 2, reward=0.5)
    held = agent._lane(2).steps[0]
    held_sampled = (held.old_log_probability.clone(), held.old_policy_log_probability.clone())
    # Lane 0 fills the rollout with the policy's own draws, and the policy moves.
    for index in range(4):
        action = _explored_decision(agent, observation, explored=False)
        agent.observe(
            Transition(state=observation, action=action, reward=float(index), next_state=observation, done=index == 3)
        )
    assert agent.optimizer_updates == 1
    assert agent.last_update["explored_steps"] == 0.0

    agent.observe(
        Transition(state=observation, action=pending_action, reward=1.0, next_state=observation, done=True),
        lane=1,
    )
    crossed = agent._lane(1).steps[0]
    _unrecorded_fight_step(agent, 2, reward=-0.01, done=True)
    for index in range(2):
        action = _explored_decision(agent, observation, explored=False)
        agent.observe(
            Transition(state=observation, action=action, reward=1.0, next_state=observation, done=index == 1)
        )

    assert agent.optimizer_updates == 2
    metrics = agent.last_update
    assert metrics["rollout_steps"] == 4.0
    assert metrics["explored_steps"] == 2.0
    assert torch.equal(crossed.old_log_probability, sampled[0])
    assert torch.equal(crossed.old_policy_log_probability, sampled[1])
    assert torch.equal(held.old_log_probability, held_sampled[0])
    assert torch.equal(held.old_policy_log_probability, held_sampled[1])
    assert crossed.epsilon == held.epsilon == 0.5
    assert crossed.explored and held.explored
    # Sampled before the first update and trained after it: a moved ratio.
    assert metrics["approx_kl"] > 0.0
    assert math.isfinite(metrics["policy_kl"]) and metrics["policy_kl"] > 0.0


def test_folding_and_closing_an_episode_leave_the_exploration_record_alone():
    torch.manual_seed(111)
    agent = _agent(rollout_size=1000, hold_open_steps=True, exploration={"map": 0.5})
    observation = _observation(_map_state(3))

    for lane, truncated in ((0, False), (1, True)):
        action = _explored_decision(agent, observation, explored=True, lane=lane)
        agent.observe(
            Transition(state=observation, action=action, reward=1.0, next_state=observation, done=False),
            lane=lane,
        )
        step = agent._lane(lane).steps[0]
        recorded = (
            step.old_log_probability.clone(),
            step.old_policy_log_probability.clone(),
            step.epsilon,
            step.explored,
        )
        if truncated:
            agent.finish_episode(observation, truncated=True, lane=lane)  # _close_episode
            assert step.episode_end and step.bootstrap_value is not None
        else:
            _unrecorded_fight_step(agent, lane, reward=-0.01, done=True)  # _fold, a terminal
            assert step.done and step.episode_end
        assert torch.equal(step.old_log_probability, recorded[0])
        assert torch.equal(step.old_policy_log_probability, recorded[1])
        assert (step.epsilon, step.explored) == (recorded[2], recorded[3]) == (0.5, True)


# --------------------------------------------------------------- reference KL


class _ToyEncoder(nn.Module):
    """An actor whose logits are a parameter, so a probability can be set by hand.

    ``policy_value`` serves the first ``len(decision.actions)`` logits and one
    learnable scalar as the value; the tokenizer still tokenizes real states.
    """

    def __init__(self, logits) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))
        self.value_bias = nn.Parameter(torch.zeros(()))

    def policy_value(self, decision: TokenizedDecision) -> PolicyValueOutput:
        return PolicyValueOutput(self.logits[: len(decision.actions)], self.value_bias, None)

    def value(self, state) -> torch.Tensor:
        return self.value_bias


def _reference_encoder(seed: int = 7) -> GameEncoder:
    """A second encoder of the test shape with its own weights, like the CLI's.

    Built under a forked generator, so the caller's random numbers are the
    same with or without a reference.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return GameEncoder(GameVocabulary.from_bundled_data(), TEST_ENCODER)


def _zero_advantages(agent) -> None:
    """Make every GAE walk return zero advantages, so with the value and entropy
    coefficients at zero only the reference term can move the policy."""
    original = agent._advantages_and_returns

    def spy(steps, tail_value=None):
        advantages, returns = original(steps, tail_value=tail_value)
        return torch.zeros_like(advantages), returns

    agent._advantages_and_returns = spy


def _reference_kl_on(agent, steps) -> float:
    """Mean KL(pi_ref || pi_theta) over stored decisions, under the current weights."""
    total = 0.0
    for step in steps:
        with torch.no_grad():
            log_pi = torch.log_softmax(agent.game_encoder.policy_value(step.decision).logits, dim=-1)
        reference = step.reference_log_probabilities
        total += float((reference.exp() * (reference - log_pi)).sum())
    return total / len(steps)


def _sampled_decisions(agent, count: int, observation=None, reward: float = 1.0) -> list:
    """``count`` sampled decisions on one lane, the last one ending its episode."""
    observation = observation or _observation(_map_state(3))
    for index in range(count):
        action = agent.choose_action(observation)
        agent.observe(
            Transition(
                state=observation,
                action=action,
                reward=reward,
                next_state=observation,
                done=index == count - 1,
            )
        )
    return list(agent._lane(0).steps)


BASELINE = Path(__file__).parent / "fixtures" / "ppo_baseline_85c6703.json"


def _digest(tensor: torch.Tensor) -> dict:
    """A tensor as the baseline records it: its bytes' sha256, shape and dtype."""
    tensor = tensor.detach().cpu().contiguous()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
    }


def _assert_matches_baseline(agent, actions: list[int], baseline: dict) -> None:
    """The run is the recorded one bit for bit: the actions, the generator's
    state, every parameter and Adam tensor, and every recorded metric."""
    assert actions == baseline["actions"]
    assert _digest(torch.get_rng_state()) == baseline["rng"]
    parameters = agent.game_encoder.state_dict()
    assert set(parameters) == set(baseline["parameters"])
    for name, tensor in parameters.items():
        assert _digest(tensor) == baseline["parameters"][name], name
    optimizer = agent.optimizer.state_dict()
    assert json.loads(json.dumps(optimizer["param_groups"])) == baseline["optimizer"]["param_groups"]
    assert {str(index) for index in optimizer["state"]} == set(baseline["optimizer"]["state"])
    for index, entry in optimizer["state"].items():
        recorded = baseline["optimizer"]["state"][str(index)]
        assert set(entry) == set(recorded)
        for name, value in entry.items():
            assert _digest(value) == recorded[name], (index, name)
    assert baseline["metrics"].items() <= agent.last_update.items()


@pytest.mark.parametrize("key", ["plain", "explore_map"])
def test_without_a_reference_the_run_is_bit_identical_to_the_base_commit(key: str):
    """``fixtures/ppo_baseline_85c6703.json`` is the seeded run of
    ``_reference_run`` under commit 85c6703, before the reference KL existed:
    the actions, every parameter tensor, every Adam tensor, the generator's
    state and the update's metrics, with each tensor as the sha256 of its
    bytes -- bit-exact, as ``torch.equal`` would be, in a few kilobytes of
    JSON. Captured by extracting that commit's ``src`` and ``tests`` with
    ``git archive`` and running ``_reference_run`` from them, with and without
    an exploration rate on the screen the run visits."""
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))["runs"][key]

    agent, actions = _reference_run(baseline["exploration"] or ())

    assert agent.reference_encoder is None
    assert agent.config.reference_kl_coefficient == 0.0
    assert agent.config.reference_kl_screens == ()
    _assert_matches_baseline(agent, actions, baseline)
    assert agent.last_update == baseline["metrics"]


def test_the_reference_term_pulls_the_policy_toward_the_reference():
    """With zero advantages and no value or entropy term, the loss is the
    reference KL alone, and updates on the same stored decisions reduce it."""
    torch.manual_seed(120)
    agent = _agent(
        rollout_size=1000, update_epochs=1, minibatch_size=64, learning_rate=0.01,
        reference_kl=1.0, reference=_reference_encoder(),
    )
    agent.config = dataclasses.replace(
        agent.config, value_coefficient=0.0, entropy_coefficient=0.0
    )
    _zero_advantages(agent)
    steps = _sampled_decisions(agent, 6)
    before = _reference_kl_on(agent, steps)
    assert before > 0.0

    seen = []
    for _ in range(5):
        if not agent._lane(0).steps:
            agent._lane(0).steps.extend(steps)
        metrics = agent.update()
        seen.append((metrics, _reference_kl_on(agent, steps)))

    # The first update measured the KL before stepping: what ``before`` is.
    assert seen[0][0]["reference_kl"] == pytest.approx(before, rel=1e-4)
    assert seen[0][0]["reference_steps"] == 6.0
    assert seen[-1][1] < before
    assert seen[-1][0]["policy_loss"] == 0.0


def test_an_abandoned_action_the_reference_favours_gets_its_probability_back():
    """pi_theta(a) ~ 4e-7 against pi_ref(a) = 0.5: the surrogate would have to
    sample the action to move it, while the reference term's gradient on its
    logit is pi_theta - pi_ref from the first update."""
    torch.manual_seed(121)
    actor = _ToyEncoder([-14.0, 0.0, 0.0])
    reference = _ToyEncoder([math.log(2.0), 0.0, 0.0])  # softmax: 0.5, 0.25, 0.25
    agent = _agent(
        rollout_size=1000, update_epochs=1, minibatch_size=64, learning_rate=0.05,
        reference_kl=1.0, reference=reference, encoder=actor,
    )
    agent.config = dataclasses.replace(
        agent.config, value_coefficient=0.0, entropy_coefficient=0.0
    )
    _zero_advantages(agent)
    steps = _sampled_decisions(agent, 4, reward=0.0)
    assert torch.allclose(
        steps[0].reference_log_probabilities, torch.log(torch.tensor([0.5, 0.25, 0.25]))
    )
    probabilities = [float(torch.softmax(actor.logits.detach(), dim=-1)[0])]
    assert probabilities[0] < 1e-5

    for _ in range(10):
        agent._lane(0).steps.extend(steps)
        agent.update()
        probabilities.append(float(torch.softmax(actor.logits.detach(), dim=-1)[0]))

    assert all(later > earlier for earlier, later in zip(probabilities, probabilities[1:]))


def test_forced_and_external_steps_carry_no_reference_distribution():
    torch.manual_seed(122)
    agent = _agent(rollout_size=1000, reference_kl=0.5, reference=_reference_encoder())
    forced = _observation(_map_state(1))
    agent.choose_action(forced)
    assert agent._lane(0).pending.reference_log_probabilities is None
    agent.discard_decision()

    observation = _observation(_map_state(3))
    agent.choose_external(
        observation, agent.action_provider.require_candidates(observation.raw_state)[1]
    )
    assert agent._lane(0).pending.reference_log_probabilities is None
    agent.discard_decision()

    agent.choose_action(observation)
    assert agent._lane(0).pending.reference_log_probabilities is not None


def test_a_minibatch_of_external_steps_alone_has_no_reference_term():
    torch.manual_seed(123)
    agent = _agent(rollout_size=4, update_epochs=1, reference_kl=0.5, reference=_reference_encoder())
    for index in range(4):
        _external_step(agent, index % 3, reward=1.0, done=index == 3)

    metrics = agent.last_update
    assert agent.optimizer_updates == 1
    assert metrics["reference_kl"] == 0.0
    assert metrics["reference_steps"] == 0.0
    assert metrics["reference_kl_mean"] == 0.0
    assert metrics["value_loss"] > 0.0


def test_the_reference_distribution_is_over_the_reduced_candidate_set():
    """The update re-encodes the stored decision, so the reference must be
    over the same candidates: what was really offered, after ``exclude``."""
    torch.manual_seed(124)
    reference = _reference_encoder()
    agent = _agent(reference_kl=0.5, reference=reference)
    observation = _observation(_map_state(3))

    agent.choose_action(observation, exclude=[GameAction("choose_map_node", index=1)])

    pending = agent._lane(0).pending
    assert len(pending.decision.actions) == 2
    stored = pending.reference_log_probabilities
    assert stored.shape == (2,)
    assert stored.dtype == torch.float32
    assert stored.device.type == "cpu" and not stored.requires_grad
    with torch.no_grad():
        expected = torch.log_softmax(reference.policy_value(pending.decision).logits, dim=-1)
    assert torch.allclose(stored, expected, atol=1e-6)
    assert float(stored.exp().sum()) == pytest.approx(1.0, abs=1e-5)


def test_a_reference_distribution_of_the_wrong_length_is_refused():
    torch.manual_seed(125)
    agent = _agent(rollout_size=1000, reference_kl=0.5, reference=_reference_encoder())
    _step(agent, 0, reward=1.0, done=True)
    step = agent._lane(0).steps[0]
    assert len(step.decision.actions) == 2
    step.reference_log_probabilities = torch.log(torch.full((3,), 1 / 3))

    with pytest.raises(RuntimeError, match="3 candidates for a decision with 2"):
        agent.update()


def test_a_policy_step_without_a_reference_distribution_is_refused_while_the_term_is_on():
    torch.manual_seed(125)
    agent = _agent(rollout_size=1000, reference_kl=0.5, reference=_reference_encoder())
    _step(agent, 0, reward=1.0, done=True)
    agent._lane(0).steps[0].reference_log_probabilities = None

    with pytest.raises(RuntimeError, match="no reference distribution"):
        agent.update()


def test_decisions_crossing_an_update_keep_their_reference_distributions():
    """A pending decision and a held open step both outlive an update, and are
    trained by the next one against the distribution they were sampled beside."""
    torch.manual_seed(126)
    agent = _agent(rollout_size=4, hold_open_steps=True, reference_kl=0.5, reference=_reference_encoder())
    observation = _observation(_map_state(3))
    pending_action = agent.choose_action(observation, lane=1)
    pending = agent._lane(1).pending
    pending_reference = pending.reference_log_probabilities.clone()
    held_action = agent.choose_action(observation, lane=2)
    agent.observe(
        Transition(state=observation, action=held_action, reward=1.0, next_state=observation, done=False),
        lane=2,
    )
    _unrecorded_fight_step(agent, 2, reward=0.5)
    held = agent._lane(2).steps[0]
    held_reference = held.reference_log_probabilities.clone()

    for index in range(4):
        _step(agent, 0, reward=float(index), done=index == 3)

    assert agent.optimizer_updates == 1
    assert agent.last_update["reference_steps"] == 4.0
    assert agent._lane(1).pending is pending
    assert agent._lane(2).steps == [held]
    assert torch.equal(pending.reference_log_probabilities, pending_reference)
    assert torch.equal(held.reference_log_probabilities, held_reference)

    agent.observe(
        Transition(state=observation, action=pending_action, reward=1.0, next_state=observation, done=True),
        lane=1,
    )
    _unrecorded_fight_step(agent, 2, reward=-0.01, done=True)
    crossed = agent._lane(1).steps[0]
    assert torch.equal(crossed.reference_log_probabilities, pending_reference)
    for index in range(2):
        _step(agent, 0, reward=1.0, done=index == 1)

    assert agent.optimizer_updates == 2
    assert agent.last_update["rollout_steps"] == 4.0
    assert agent.last_update["reference_steps"] == 4.0


def test_folding_and_closing_an_episode_leave_the_reference_distribution_alone():
    torch.manual_seed(127)
    agent = _agent(rollout_size=1000, hold_open_steps=True, reference_kl=0.5, reference=_reference_encoder())
    observation = _observation(_map_state(3))

    for lane, truncated in ((0, False), (1, True)):
        action = agent.choose_action(observation, lane=lane)
        agent.observe(
            Transition(state=observation, action=action, reward=1.0, next_state=observation, done=False),
            lane=lane,
        )
        step = agent._lane(lane).steps[0]
        recorded = step.reference_log_probabilities.clone()
        if truncated:
            agent.finish_episode(observation, truncated=True, lane=lane)  # _close_episode
            assert step.episode_end and step.bootstrap_value is not None
        else:
            _unrecorded_fight_step(agent, lane, reward=-0.01, done=True)  # _fold, a terminal
            assert step.done and step.episode_end
        assert torch.equal(step.reference_log_probabilities, recorded)


def test_discarding_aborting_and_resetting_drop_the_reference_distribution_with_the_step():
    torch.manual_seed(128)
    agent = _agent(rollout_size=1000, reference_kl=0.5, reference=_reference_encoder())
    observation = _observation(_map_state(3))

    agent.choose_action(observation)
    agent.discard_decision()
    assert agent._lane(0).pending is None

    _step(agent, 0, reward=1.0, done=False)
    agent.choose_action(observation)
    agent.reset(observation)
    assert agent._lane(0).pending is None
    assert agent._lane(0).steps[0].reference_log_probabilities is not None

    agent.choose_action(observation)
    agent.abort_lane(0)
    assert agent._lane(0).pending is None and agent._lane(0).steps == []

    _step(agent, 1, reward=1.0, done=False)
    agent.choose_action(observation, lane=1)
    agent.abort_episode()
    assert agent._lanes == {}


def test_an_update_leaves_the_reference_bit_identical_and_outside_the_optimizer():
    torch.manual_seed(129)
    reference = _reference_encoder()
    before = {name: tensor.clone() for name, tensor in reference.state_dict().items()}
    agent = _agent(rollout_size=8, update_epochs=2, minibatch_size=4, reference_kl=0.5, reference=reference)
    assert not reference.training and agent.game_encoder.training

    _fill_policy_rollout(agent, 8)

    assert agent.optimizer_updates == 1
    assert agent.last_update["reference_kl"] > 0.0
    for name, tensor in reference.state_dict().items():
        assert torch.equal(tensor, before[name]), name
    optimized = {id(p) for group in agent.optimizer.param_groups for p in group["params"]}
    assert not any(id(p) in optimized for p in reference.parameters())
    assert not any(p.requires_grad for p in reference.parameters())
    assert all(p.grad is None for p in reference.parameters())
    # ``train`` switches the actor; the reference stays in eval mode.
    agent.train(True)
    assert not reference.training


def test_a_reference_must_be_its_own_module_with_a_coefficient_to_use_it():
    vocabulary = GameVocabulary.from_bundled_data()
    actor = GameEncoder(vocabulary, TEST_ENCODER)
    with pytest.raises(ValueError, match="share memory"):
        CandidatePPOAgent(
            GameTokenizer(vocabulary), actor,
            config=PPOConfig(reference_kl_coefficient=0.5), reference_encoder=actor,
        )
    with pytest.raises(ValueError, match="positive reference_kl_coefficient"):
        CandidatePPOAgent(
            GameTokenizer(vocabulary), actor, reference_encoder=_reference_encoder()
        )


def test_a_reference_built_on_the_actor_storage_is_refused():
    """``nn.Parameter(actor_parameter.detach())`` is a new object on the same
    memory: identity would pass it, and every update would move the reference."""
    vocabulary = GameVocabulary.from_bundled_data()
    actor = GameEncoder(vocabulary, TEST_ENCODER)
    reference = _reference_encoder()
    name, parameter = next(iter(actor.named_parameters()))
    owner, _, attribute = name.rpartition(".")
    module = reference.get_submodule(owner) if owner else reference
    setattr(module, attribute, nn.Parameter(parameter.detach()))
    assert getattr(module, attribute) is not parameter

    with pytest.raises(ValueError, match="share memory"):
        CandidatePPOAgent(
            GameTokenizer(vocabulary), actor,
            config=PPOConfig(reference_kl_coefficient=0.5), reference_encoder=reference,
        )


def test_a_logit_gap_of_sixty_gives_a_finite_reference_kl_and_finite_gradients():
    torch.manual_seed(130)
    actor = _ToyEncoder([60.0, 0.0, 0.0])
    reference = _ToyEncoder([0.0, 60.0, 0.0])
    agent = _agent(
        rollout_size=1000, update_epochs=1, minibatch_size=8,
        reference_kl=1.0, reference=reference, encoder=actor,
    )
    _sampled_decisions(agent, 1)

    metrics = agent.update()

    assert math.isfinite(metrics["reference_kl"])
    assert metrics["reference_kl"] == pytest.approx(60.0, abs=1e-3)
    assert math.isfinite(metrics["loss"]) and math.isfinite(metrics["gradient_norm"])
    assert all(bool(torch.isfinite(p.grad).all()) for p in actor.parameters())
    assert all(bool(torch.isfinite(p).all()) for p in actor.parameters())


def test_on_an_explored_screen_the_reference_kl_reads_the_policy_while_the_ratio_reads_the_mixture():
    torch.manual_seed(131)
    epsilon, ratio = 0.5, 1.5
    actor = _ToyEncoder([1.0, 0.0, -1.0])
    reference = _ToyEncoder([math.log(2.0), 0.0, 0.0])
    agent = _agent(
        rollout_size=1000, update_epochs=1, minibatch_size=8, exploration={"map": epsilon},
        reference_kl=0.3, reference=reference, encoder=actor,
    )
    observation = _observation(_map_state(3))
    action = _explored_decision(agent, observation, explored=True)
    agent.observe(
        Transition(state=observation, action=action, reward=1.0, next_state=observation, done=True)
    )
    step = agent._lane(0).steps[0]
    # The policy has not moved, so shifting the stored mixture probability
    # makes the update see exactly ``ratio`` on the mixture.
    step.old_log_probability = step.old_log_probability - math.log(ratio)
    pi = torch.softmax(actor.logits.detach(), dim=-1)
    pi_ref = torch.tensor([0.5, 0.25, 0.25])
    mixture = (1.0 - epsilon) * pi + epsilon / 3
    expected = float((pi_ref * (pi_ref.log() - pi.log())).sum())
    against_the_mixture = float((pi_ref * (pi_ref.log() - mixture.log())).sum())
    assert abs(expected - against_the_mixture) > 1e-3

    metrics = agent.update()

    assert metrics["reference_steps"] == 1.0
    assert metrics["reference_kl"] == pytest.approx(expected, rel=1e-5)
    assert metrics["approx_kl"] == pytest.approx((ratio - 1.0) - math.log(ratio), rel=1e-4)


def test_evaluation_needs_no_reference_but_training_does():
    """Evaluation builds the agent from a saved plan whose coefficient is
    positive and never samples; only a sampled decision needs the artifact."""
    torch.manual_seed(132)
    agent = _agent(reference_kl=0.5)
    agent.eval()
    observation = _observation(_map_state(3))

    assert agent.choose_action(observation).to_dict() in [_node(i) for i in range(3)]
    assert agent._lane(0).pending is None

    agent.train(True)
    with pytest.raises(RuntimeError, match="reference policy"):
        agent.choose_action(observation)
    assert agent._lane(0).pending is None
    # A forced step samples nothing and needs none either.
    agent.choose_action(_observation(_map_state(1)))
    assert agent._lane(0).pending.reference_log_probabilities is None


def test_building_an_agent_with_a_reference_leaves_the_generator_alone():
    reference = _reference_encoder()
    vocabulary = GameVocabulary.from_bundled_data()
    torch.manual_seed(133)
    encoder = GameEncoder(vocabulary, TEST_ENCODER)
    state = torch.get_rng_state()

    CandidatePPOAgent(
        GameTokenizer(vocabulary), encoder,
        config=PPOConfig(reference_kl_coefficient=0.5), reference_encoder=reference,
    )

    assert torch.equal(torch.get_rng_state(), state)


def test_reference_metrics_count_the_policy_steps_of_each_minibatch():
    torch.manual_seed(134)
    agent = _agent(rollout_size=1000, update_epochs=1, minibatch_size=3, reference_kl=0.5, reference=_reference_encoder())
    results = []
    original = agent._optimize_minibatch

    def spy(*args, **kwargs):
        results.append(original(*args, **kwargs))
        return results[-1]

    agent._optimize_minibatch = spy
    for _ in range(4):
        _step(agent, 0, reward=1.0, done=False)
    forced = _observation(_map_state(1))
    agent.observe(
        Transition(state=forced, action=agent.choose_action(forced), reward=1.0, next_state=forced, done=False)
    )
    _external_step(agent, 1, reward=1.0, done=True)

    metrics = agent.update()

    assert metrics["rollout_steps"] == 6.0
    assert len(results) == 2
    assert sum(result["reference_steps"] for result in results) == 4.0
    assert all(result["reference_kl"] >= 0.0 for result in results)
    assert metrics["reference_kl"] == results[-1]["reference_kl"]
    assert metrics["reference_steps"] == results[-1]["reference_steps"]
    weighted = sum(r["reference_kl"] * r["reference_steps"] for r in results) / 4.0
    assert metrics["reference_kl_mean"] == pytest.approx(weighted)


def test_a_kl_stop_on_the_first_minibatch_still_reports_the_reference_mean():
    torch.manual_seed(135)
    agent = _agent(
        rollout_size=1000, update_epochs=2, minibatch_size=4, target_kl=1e-3,
        reference_kl=0.5, reference=_reference_encoder(),
    )
    for _ in range(4):
        _step(agent, 0, reward=1.0, done=True)
    for step in agent._lane(0).steps:
        step.old_log_probability = step.old_log_probability - 1.0

    metrics = agent.update()

    assert metrics["kl_early_stop"] == 1.0 and metrics["optimizer_steps"] == 0.0
    assert metrics["reference_kl_mean"] == 0.0
    assert "reference_kl" not in metrics
    # Nothing stepped, so no screen was measured and none gets a key.
    assert not any(key.startswith(("reference_kl/", "reference_steps/")) for key in metrics)


@pytest.mark.parametrize("coefficient", [-0.1, float("nan"), float("inf"), True])
def test_the_reference_coefficient_is_a_finite_non_negative_number(coefficient):
    with pytest.raises(ValueError, match="reference_kl_coefficient"):
        PPOConfig(reference_kl_coefficient=coefficient)


def test_the_reference_term_gradient_on_the_logits_is_beta_times_pi_minus_pi_ref_over_the_policy_steps():
    """d/dlogits of beta * mean_i KL(pi_ref_i || pi_theta_i) on one step is
    beta * (pi_theta - pi_ref) / N, N the policy steps of the minibatch; a
    forced step in the same minibatch is not one of them."""
    torch.manual_seed(136)
    beta = 0.7
    agent = _agent(
        rollout_size=1000, update_epochs=1, minibatch_size=64,
        reference_kl=beta, reference=_reference_encoder(),
    )
    agent.config = dataclasses.replace(
        agent.config, value_coefficient=0.0, entropy_coefficient=0.0
    )
    _zero_advantages(agent)
    # Candidate counts 2, 3 and 4, so each forward pass of the update is told
    # apart by the length of its logits.
    for count in (2, 3, 4):
        observation = _observation(_map_state(count))
        action = agent.choose_action(observation)
        agent.observe(
            Transition(state=observation, action=action, reward=0.0, next_state=observation, done=False)
        )
    forced = _observation(_map_state(1))
    agent.observe(
        Transition(state=forced, action=agent.choose_action(forced), reward=0.0, next_state=forced, done=True)
    )
    by_length = {
        len(step.decision.actions): step for step in agent._lane(0).steps if _policy_step(step)
    }
    captured: list[torch.Tensor] = []
    original = agent.game_encoder.policy_value

    def spy(decision):
        output = original(decision)
        output.logits.retain_grad()
        captured.append(output.logits)
        return output

    agent.game_encoder.policy_value = spy

    metrics = agent.update()

    assert metrics["reference_steps"] == 3.0 and metrics["rollout_steps"] == 4.0
    assert metrics["policy_loss"] == 0.0
    assert len(captured) == 3
    for logits in captured:
        step = by_length[logits.numel()]
        expected = beta * (torch.softmax(logits.detach(), dim=-1) - step.reference_log_probabilities.exp()) / 3
        assert torch.allclose(logits.grad, expected, atol=1e-6), logits.numel()


@pytest.mark.parametrize(("reward", "ratio"), [(50.0, 1.0), (-50.0, 1.0), (50.0, 1.5), (-50.0, 0.5)])
def test_the_reference_term_ignores_the_advantage_sign_and_the_ratio(reward: float, ratio: float):
    """The same stored step with the advantage flipped or the ratio moved off
    one reports the same reference KL: the pull is toward the reference,
    whatever the surrogate makes of the action."""
    torch.manual_seed(137)
    actor = _ToyEncoder([1.0, 0.0, -1.0])
    reference = _ToyEncoder([math.log(2.0), 0.0, 0.0])
    agent = _agent(
        rollout_size=1000, update_epochs=1, minibatch_size=8,
        reference_kl=0.3, reference=reference, encoder=actor,
    )
    _sampled_decisions(agent, 1, reward=reward)
    step = agent._lane(0).steps[0]
    step.old_log_probability = step.old_log_probability - math.log(ratio)
    step.old_policy_log_probability = step.old_policy_log_probability - math.log(ratio)
    pi = torch.softmax(actor.logits.detach(), dim=-1)
    pi_ref = torch.tensor([0.5, 0.25, 0.25])
    expected = float((pi_ref * (pi_ref.log() - pi.log())).sum())

    metrics = agent.update()

    assert metrics["reference_kl"] == pytest.approx(expected, rel=1e-5)
    assert metrics["approx_kl"] == pytest.approx((ratio - 1.0) - math.log(ratio), abs=1e-6)
    assert (metrics["policy_loss"] < 0) == (reward > 0)


def test_the_reference_distribution_follows_the_reduced_candidate_order():
    """Excluding the middle of three candidates leaves [0, 2] in that order,
    and the stored reference is pi_ref over exactly those two, in that order:
    the entries of the full distribution for 0 and 2, renormalised, and not
    the first two or the last two of it."""
    torch.manual_seed(138)
    reference = _reference_encoder()
    agent = _agent(reference_kl=0.5, reference=reference)
    observation = _observation(_map_state(3))
    candidates = agent.action_provider.require_candidates(observation.raw_state)
    with torch.no_grad():
        full = reference.policy_value(agent.tokenizer.tokenize_decision(observation, candidates)).logits

    action = agent.choose_action(observation, exclude=[GameAction("choose_map_node", index=1)])

    pending = agent._lane(0).pending
    assert action.to_dict() in (_node(0), _node(2))
    assert len(pending.decision.actions) == 2
    stored = pending.reference_log_probabilities
    assert torch.allclose(stored, torch.log_softmax(full[[0, 2]], dim=-1), atol=1e-5)
    for wrong in ([0, 1], [1, 2], [2, 0]):
        assert not torch.allclose(stored, torch.log_softmax(full[wrong], dim=-1), atol=1e-5), wrong


def test_a_played_checkpoint_needs_no_reference():
    """``PPOConfig.without_reference`` is what the snapshot and boss scripts
    build a saved plan's agent with: sampled or argmaxed, never updated, so
    the artifact the run pulled toward is nothing to them."""
    config = PPOConfig(
        reference_kl_coefficient=0.3,
        reference_kl_screens={"shop": 0.0, "map": 0.05},
        exploration={"map": 0.2},
        rollout_size=10**9,
    )
    played = config.without_reference()
    assert played.reference_kl_coefficient == 0.0
    assert played.reference_kl_screens == ()
    assert not played.uses_reference
    assert (
        dataclasses.replace(
            played, reference_kl_coefficient=0.3, reference_kl_screens=config.reference_kl_screens
        )
        == config
    )
    assert PPOConfig().without_reference() == PPOConfig()

    torch.manual_seed(139)
    vocabulary = GameVocabulary.from_bundled_data()
    agent = CandidatePPOAgent(GameTokenizer(vocabulary), GameEncoder(vocabulary, TEST_ENCODER), config=played)
    agent.train(True)
    agent.choose_action(_observation(_map_state(3)))

    assert agent.reference_encoder is None
    assert agent._lane(0).pending.reference_log_probabilities is None


def test_a_coefficient_that_overflows_the_weighted_term_fails_before_the_backward_pass():
    """Every term finite, the sum not: beta * KL overflows float32, and the
    check names the term before any gradient exists."""
    torch.manual_seed(140)
    actor = _ToyEncoder([1.0, 0.0, -1.0])
    agent = _agent(
        rollout_size=1000, update_epochs=1, minibatch_size=8,
        reference_kl=1e300, reference=_ToyEncoder([math.log(2.0), 0.0, 0.0]), encoder=actor,
    )
    _sampled_decisions(agent, 1)
    before = actor.logits.detach().clone()

    with pytest.raises(RuntimeError, match=r"loss is not finite.*beta \* reference_kl=inf"):
        agent.update()

    assert actor.logits.grad is None
    assert torch.equal(actor.logits.detach(), before)
    assert agent.optimizer_updates == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_reference_distribution_that_is_not_finite_is_refused(bad: float):
    torch.manual_seed(141)
    agent = _agent(rollout_size=1000, reference_kl=0.5, reference=_reference_encoder())
    _step(agent, 0, reward=1.0, done=True)
    agent._lane(0).steps[0].reference_log_probabilities = torch.tensor([bad, math.log(0.5)])

    with pytest.raises(RuntimeError, match="reference distribution is not finite"):
        agent.update()


@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_a_non_finite_gradient_norm_refuses_the_step(beta: float):
    """Adam must never step on NaN: the clip raises instead of scaling by it."""
    torch.manual_seed(142)
    actor = _ToyEncoder([1.0, 0.0, -1.0])
    agent = _agent(
        rollout_size=1000, update_epochs=1, minibatch_size=8, encoder=actor,
        reference_kl=beta, reference=_ToyEncoder([0.0, 0.0, 0.0]) if beta else None,
    )
    _sampled_decisions(agent, 1)
    actor.logits.register_hook(lambda gradient: gradient * float("nan"))
    before = actor.logits.detach().clone()

    with pytest.raises(RuntimeError, match="non-finite"):
        agent.update()

    assert torch.equal(actor.logits.detach(), before)
    assert agent.optimizer_updates == 0
    assert not agent.optimizer.state


# ------------------------------------------- per-screen reference coefficients

REFERENCE_BASELINE = Path(__file__).parent / "fixtures" / "ppo_reference_baseline_5f64073.json"


def _player() -> dict[str, object]:
    return {
        "character": "The Ironclad", "hp": 40, "max_hp": 80, "gold": 50,
        "relics": [], "potions": [], "status": [],
    }


def _rest_site_state() -> dict[str, object]:
    """A rest site offering rest and smith: two candidates."""
    return {
        "state_type": "rest_site",
        "run": {"floor": 6},
        "player": _player(),
        "rest_site": {
            "options": [{"index": 0, "id": "REST"}, {"index": 1, "id": "SMITH"}],
            "can_proceed": False,
        },
    }


def _card_reward_state(count: int = 2) -> dict[str, object]:
    """A card reward of ``count`` cards and a skip: ``count + 1`` candidates."""
    return {
        "state_type": "card_reward",
        "run": {"floor": 6},
        "player": _player(),
        "card_reward": {
            "cards": [
                {"index": index, "id": "UPPERCUT", "type": "Attack", "cost": "2", "rarity": "Uncommon"}
                for index in range(count)
            ],
            "can_skip": True,
        },
    }


def _decide(agent, observation, *, reward: float = 0.0, done: bool = False):
    """One sampled decision on lane 0, observed; returns its rollout step."""
    action = agent.choose_action(observation)
    agent.observe(
        Transition(state=observation, action=action, reward=reward, next_state=observation, done=done)
    )
    return agent._lane(0).steps[-1]


@pytest.mark.parametrize("key", ["reference", "reference_minibatch_5", "plain"])
def test_without_per_screen_coefficients_the_run_is_bit_identical_to_the_one_before_them(key: str):
    """``fixtures/ppo_reference_baseline_5f64073.json`` is ``_reference_run``
    under commit 5f64073, before a coefficient could be set per screen, with
    a reference policy at a global coefficient of 0.1 and without one: the
    same record as ``ppo_baseline_85c6703.json``, captured the same way. An
    empty mapping must leave that path alone -- the run in progress on it
    resumes onto the same experiment -- so the actions, the generator, every
    parameter and every Adam tensor are checked bit for bit, and the metrics
    gain only the per-screen keys. The minibatch of 5 is the run that tells
    beta times the mean KL from the sum of the weighted terms over the count:
    over 8 steps the division is exact and the two coincide, over 5 they
    differ by an ulp that Adam's moments keep."""
    baseline = json.loads(REFERENCE_BASELINE.read_text(encoding="utf-8"))["runs"][key]
    beta = baseline["reference_kl_coefficient"]

    agent, actions = _reference_run(
        reference_kl=beta, reference=_reference_encoder() if beta else None, **baseline["options"]
    )

    assert agent.config.reference_kl_coefficient == beta
    assert agent.config.reference_kl_screens == ()
    assert agent.config.uses_reference is (beta > 0)
    _assert_matches_baseline(agent, actions, baseline)
    added = set(agent.last_update) - set(baseline["metrics"])
    if beta:
        # The eight decisions were all on the map, so that is the one screen
        # measured: its figures are the update's own, over all eight steps.
        assert added == {"reference_kl/map", "reference_steps/map"}
        assert agent.last_update["reference_steps/map"] == 8.0
        assert agent.last_update["reference_kl/map"] == pytest.approx(
            baseline["metrics"]["reference_kl_mean"], rel=1e-6
        )
    else:
        assert added == set()


def test_a_screen_with_a_coefficient_stores_the_reference_and_a_screen_at_zero_does_not():
    """rest_site at 0.1 and card_reward at 0, no global coefficient: the
    rest-site step carries pi_ref and is pulled on 0.1 * KL over the
    minibatch's two policy steps; the card-reward step carries None, is not a
    reference step, and the update asks nothing of it."""
    torch.manual_seed(150)
    agent = _agent(
        rollout_size=1000, update_epochs=1, minibatch_size=64,
        reference_kl_screens={"rest_site": 0.1, "card_reward": 0.0}, reference=_reference_encoder(),
    )
    config = agent.config
    assert config.reference_kl_for("rest_site") == 0.1
    assert config.reference_kl_for("card_reward") == 0.0 and config.reference_kl_for("map") == 0.0
    assert config.uses_reference and config.reference_screens() == ("rest_site",)
    agent.config = dataclasses.replace(config, value_coefficient=0.0, entropy_coefficient=0.0)
    _zero_advantages(agent)
    rest = _decide(agent, _observation(_rest_site_state()))
    card = _decide(agent, _observation(_card_reward_state()), done=True)
    assert rest.reference_log_probabilities is not None
    assert rest.reference_log_probabilities.shape == (2,)
    assert card.reference_log_probabilities is None
    kl = _reference_kl_on(agent, [rest])

    metrics = agent.update()

    assert metrics["rollout_steps"] == 2.0
    assert metrics["reference_steps"] == 1.0
    assert metrics["reference_kl"] == pytest.approx(kl, rel=1e-5)
    assert metrics["reference_kl_mean"] == pytest.approx(kl, rel=1e-5)
    assert metrics["policy_loss"] == 0.0
    assert metrics["loss"] == pytest.approx(0.1 * kl / 2, abs=1e-6)
    assert {key for key in metrics if key.startswith(("reference_kl/", "reference_steps/"))} == {
        "reference_kl/rest_site",
        "reference_steps/rest_site",
    }
    assert metrics["reference_kl/rest_site"] == pytest.approx(kl, rel=1e-5)
    assert metrics["reference_steps/rest_site"] == 1.0


def test_a_zero_override_switches_the_reference_off_on_its_screen_only():
    torch.manual_seed(151)
    agent = _agent(
        rollout_size=1000, update_epochs=1, minibatch_size=64,
        reference_kl=0.3, reference_kl_screens={"map": 0.0}, reference=_reference_encoder(),
    )
    config = agent.config
    assert config.reference_kl_for("map") == 0.0 and config.reference_kl_for("rest_site") == 0.3
    assert config.uses_reference
    assert config.reference_screens() == tuple(screen for screen in STATE_TYPES if screen != "map")
    map_step = _decide(agent, _observation(_map_state(3)))
    rest_step = _decide(agent, _observation(_rest_site_state()), done=True)
    assert map_step.reference_log_probabilities is None
    assert rest_step.reference_log_probabilities is not None

    metrics = agent.update()

    assert metrics["rollout_steps"] == 2.0 and metrics["reference_steps"] == 1.0
    assert "reference_kl/map" not in metrics
    assert metrics["reference_kl/rest_site"] == pytest.approx(metrics["reference_kl"])
    # Every screen at 0 under a positive global coefficient is no reference at all.
    silent = PPOConfig(
        reference_kl_coefficient=0.3, reference_kl_screens={screen: 0.0 for screen in STATE_TYPES}
    )
    assert not silent.uses_reference and silent.reference_screens() == ()
    vocabulary = GameVocabulary.from_bundled_data()
    with pytest.raises(ValueError, match="positive reference_kl_coefficient"):
        CandidatePPOAgent(
            GameTokenizer(vocabulary), GameEncoder(vocabulary, TEST_ENCODER),
            config=silent, reference_encoder=_reference_encoder(),
        )


def test_an_override_alone_switches_the_reference_on_there_only():
    torch.manual_seed(152)
    config = PPOConfig(reference_kl_screens={"rest_site": 0.2})
    assert config.reference_kl_coefficient == 0.0 and config.uses_reference
    assert config.reference_screens() == ("rest_site",)
    # With a reference the rest-site decision carries it and the map one does not ...
    agent = _agent(reference_kl_screens={"rest_site": 0.2}, reference=_reference_encoder())
    agent.choose_action(_observation(_map_state(3)))
    assert agent._lane(0).pending.reference_log_probabilities is None
    agent.discard_decision()
    agent.choose_action(_observation(_rest_site_state()))
    assert agent._lane(0).pending.reference_log_probabilities is not None
    # ... and without one, only the screen that needs it is refused.
    bare = _agent(reference_kl_screens={"rest_site": 0.2})
    bare.choose_action(_observation(_map_state(3)))
    assert bare._lane(0).pending.reference_log_probabilities is None
    bare.discard_decision()
    with pytest.raises(RuntimeError, match="reference policy"):
        bare.choose_action(_observation(_rest_site_state()))
    assert bare._lane(0).pending is None


def test_the_weighted_term_averages_each_screen_coefficient_times_its_kl_over_the_policy_steps():
    """map at 0.1, rest_site at 0.3 and card_reward on the global 0: the term
    is (0.1 * KL_map + 0.3 * KL_rest) / 3 over the minibatch's three policy
    steps, and the gradient on each step's logits is
    beta_s * (pi_theta - pi_ref) / 3 -- zero on the card reward."""
    torch.manual_seed(153)
    coefficients = {"map": 0.1, "rest_site": 0.3}
    agent = _agent(
        rollout_size=1000, update_epochs=1, minibatch_size=64,
        reference_kl_screens=coefficients, reference=_reference_encoder(),
    )
    agent.config = dataclasses.replace(
        agent.config, value_coefficient=0.0, entropy_coefficient=0.0
    )
    _zero_advantages(agent)
    # Candidate counts 4, 2 and 3, so each forward pass of the update is told
    # apart by the length of its logits; the forced step is no policy step.
    steps = [
        _decide(agent, _observation(_map_state(4))),
        _decide(agent, _observation(_rest_site_state())),
        _decide(agent, _observation(_card_reward_state(2))),
    ]
    forced = _observation(_map_state(1))
    agent.observe(
        Transition(state=forced, action=agent.choose_action(forced), reward=0.0, next_state=forced, done=True)
    )
    assert [step.state_type for step in steps] == ["map", "rest_site", "card_reward"]
    assert steps[2].reference_log_probabilities is None
    kl = {step.state_type: _reference_kl_on(agent, [step]) for step in steps[:2]}
    by_length = {len(step.decision.actions): step for step in steps}
    captured: list[torch.Tensor] = []
    original = agent.game_encoder.policy_value

    def spy(decision):
        output = original(decision)
        output.logits.retain_grad()
        captured.append(output.logits)
        return output

    agent.game_encoder.policy_value = spy

    metrics = agent.update()

    assert metrics["rollout_steps"] == 4.0 and metrics["reference_steps"] == 2.0
    assert metrics["policy_loss"] == 0.0
    assert metrics["loss"] == pytest.approx((0.1 * kl["map"] + 0.3 * kl["rest_site"]) / 3, abs=1e-6)
    assert metrics["reference_kl"] == pytest.approx((kl["map"] + kl["rest_site"]) / 2, rel=1e-5)
    assert metrics["reference_kl/map"] == pytest.approx(kl["map"], rel=1e-5)
    assert metrics["reference_kl/rest_site"] == pytest.approx(kl["rest_site"], rel=1e-5)
    assert metrics["reference_steps/map"] == metrics["reference_steps/rest_site"] == 1.0
    assert "reference_kl/card_reward" not in metrics and "reference_steps/card_reward" not in metrics
    assert len(captured) == 3
    for logits in captured:
        step = by_length[logits.numel()]
        beta = coefficients.get(step.state_type, 0.0)
        pi = torch.softmax(logits.detach(), dim=-1)
        expected = (
            beta * (pi - step.reference_log_probabilities.exp()) / 3 if beta else torch.zeros_like(pi)
        )
        gradient = logits.grad if logits.grad is not None else torch.zeros_like(pi)
        assert torch.allclose(gradient, expected, atol=1e-6), step.state_type


@pytest.mark.parametrize(
    "screens",
    [
        {"lobby": 0.1},
        (("map", 0.1), ("map", 0.2)),
        {"map": -0.1},
        {"map": float("nan")},
        {"map": float("inf")},
    ],
)
def test_a_per_screen_coefficient_names_a_known_screen_once_with_a_finite_non_negative_number(screens):
    with pytest.raises(ValueError, match="reference_kl_screens"):
        PPOConfig(reference_kl_screens=screens)


def _spy_minibatches(agent) -> list[dict[str, float]]:
    """Record every minibatch's own metrics as ``update`` first sees them.

    Snapshots, because the update's own metrics are the last minibatch's
    dict with the update's figures written over it.
    """
    results: list[dict[str, float]] = []
    original = agent._optimize_minibatch

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        results.append(dict(result))
        return result

    agent._optimize_minibatch = spy
    return results


def _per_screen_keys(metrics: Mapping[str, float]) -> set[str]:
    return {key for key in metrics if key.startswith(("reference_kl/", "reference_steps/"))}


def test_a_screen_measured_in_an_earlier_minibatch_keeps_its_figures():
    """One rest-site and one map decision in minibatches of one: whichever
    minibatch steps last reports only its own screen, and the update still
    carries the other screen's KL and count. A screen with no reference step
    in the update has no key at all."""
    torch.manual_seed(154)
    agent = _agent(
        rollout_size=1000, update_epochs=1, minibatch_size=1,
        reference_kl=0.1, reference=_reference_encoder(),
    )
    results = _spy_minibatches(agent)
    _decide(agent, _observation(_rest_site_state()))
    _decide(agent, _observation(_map_state(3)), done=True)

    metrics = agent.update()

    assert len(results) == 2
    by_screen = {
        screen: result
        for result in results
        for screen in ("rest_site", "map")
        if f"reference_kl/{screen}" in result
    }
    assert set(by_screen) == {"rest_site", "map"}
    assert all(result["reference_steps"] == 1.0 for result in results)
    for screen, result in by_screen.items():
        assert metrics[f"reference_kl/{screen}"] == result[f"reference_kl/{screen}"]
        assert metrics[f"reference_steps/{screen}"] == 1.0
    assert _per_screen_keys(metrics) == {
        "reference_kl/rest_site", "reference_steps/rest_site", "reference_kl/map", "reference_steps/map",
    }
    # The update-level keys keep their meaning: the last minibatch's own
    # figures, and the weighted mean over both.
    assert metrics["reference_steps"] == 1.0
    assert metrics["reference_kl_mean"] == pytest.approx(
        sum(result["reference_kl"] for result in results) / 2, rel=1e-6
    )


def test_a_screen_kl_is_weighted_by_its_steps_in_each_minibatch():
    """Three rest-site decisions in minibatches of two: the one that holds two
    weighs twice the one that holds one, and the policy moves between them."""
    torch.manual_seed(155)
    agent = _agent(
        rollout_size=1000, update_epochs=1, minibatch_size=2, learning_rate=0.02,
        reference_kl=0.1, reference=_reference_encoder(),
    )
    results = _spy_minibatches(agent)
    for index in range(3):
        _decide(agent, _observation(_rest_site_state()), reward=float(index), done=index == 2)

    metrics = agent.update()

    assert len(results) == 2
    assert sorted(result["reference_steps/rest_site"] for result in results) == [1.0, 2.0]
    weighted = sum(r["reference_kl/rest_site"] * r["reference_steps/rest_site"] for r in results) / 3
    unweighted = sum(r["reference_kl/rest_site"] for r in results) / 2
    assert metrics["reference_steps/rest_site"] == 3.0
    assert metrics["reference_kl/rest_site"] == pytest.approx(weighted, rel=1e-6)
    assert abs(weighted - unweighted) > 1e-5
    # Every step was on the rest site, so the screen's mean is the update's.
    assert metrics["reference_kl/rest_site"] == pytest.approx(metrics["reference_kl_mean"], rel=1e-6)
    assert _per_screen_keys(metrics) == {"reference_kl/rest_site", "reference_steps/rest_site"}


def test_the_reference_is_read_only_on_a_screen_with_a_positive_coefficient():
    """The reference's forward pass is skipped, not merely discarded, on a
    screen whose coefficient is 0."""
    torch.manual_seed(156)
    reference = _reference_encoder()
    calls: list[int] = []
    original = reference.policy_value

    def counting(decision):
        calls.append(len(decision.actions))
        return original(decision)

    reference.policy_value = counting
    agent = _agent(reference_kl_screens={"rest_site": 0.1, "card_reward": 0.0}, reference=reference)
    for observation in (_observation(_map_state(3)), _observation(_card_reward_state())):
        agent.choose_action(observation)
        assert agent._lane(0).pending.reference_log_probabilities is None
        agent.discard_decision()
    assert calls == []
    agent.choose_action(_observation(_rest_site_state()))
    assert calls == [2]
    assert agent._lane(0).pending.reference_log_probabilities is not None
    agent.discard_decision()
    # Under a global coefficient, the screen at 0 is the one that reads nothing.
    calls.clear()
    under_global = _agent(reference_kl=0.1, reference_kl_screens={"map": 0.0}, reference=reference)
    under_global.choose_action(_observation(_map_state(3)))
    assert calls == [] and under_global._lane(0).pending.reference_log_probabilities is None
    under_global.discard_decision()
    under_global.choose_action(_observation(_card_reward_state()))
    assert calls == [3]
    under_global.discard_decision()
    under_global.choose_action(_observation(_rest_site_state()))
    assert calls == [3, 2]


def test_evaluation_needs_no_reference_when_only_an_override_enables_it():
    """``sts2rl-eval`` builds the agent from the saved plan without the
    artifact; an override alone must not change that."""
    torch.manual_seed(157)
    agent = _agent(reference_kl_screens={"rest_site": 0.2})
    assert agent.config.uses_reference and agent.reference_encoder is None
    agent.eval()
    observation = _observation(_rest_site_state())

    torch.manual_seed(158)
    choices = {agent.choose_action(observation).to_dict()["index"] for _ in range(5)}
    drawn = torch.rand(())
    torch.manual_seed(158)

    assert len(choices) == 1
    assert torch.equal(drawn, torch.rand(()))
    assert agent._lane(0).pending is None
    # Training on that screen is what needs the artifact.
    agent.train(True)
    with pytest.raises(RuntimeError, match="reference policy"):
        agent.choose_action(observation)
    assert agent._lane(0).pending is None


@pytest.mark.parametrize("beta", ["0.1", True, None])
def test_a_per_screen_coefficient_must_be_a_number(beta):
    with pytest.raises(TypeError, match="reference_kl_screens"):
        PPOConfig(reference_kl_screens={"map": beta})


def test_per_screen_coefficients_are_normalised_so_order_and_spelling_do_not_count():
    assert PPOConfig(reference_kl_screens={"map": 0.03, "rest_site": 0.1}) == PPOConfig(
        reference_kl_screens=[["rest_site", 0.1], ["map", 0.03]]
    )
    config = PPOConfig(
        reference_kl_coefficient=0.05, reference_kl_screens={"rest_site": 0.1, "map": 0.03, "shop": 0}
    )
    assert config.reference_kl_screens == (("map", 0.03), ("rest_site", 0.1), ("shop", 0.0))
    assert config.reference_kl_for("map") == 0.03
    assert config.reference_kl_for("rest_site") == 0.1
    assert config.reference_kl_for("shop") == 0.0
    assert config.reference_kl_for("card_reward") == 0.05
    assert config.reference_screens() == tuple(screen for screen in STATE_TYPES if screen != "shop")
    assert PPOConfig().reference_kl_screens == () and not PPOConfig().uses_reference
    assert PPOConfig(reference_kl_coefficient=0.1).reference_screens() == tuple(STATE_TYPES)
