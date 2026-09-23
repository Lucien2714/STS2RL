"""Numerical checks for trajectory boundaries and reward-unit invariance."""

from dataclasses import replace

import pytest
import torch

from sts2rl.agents import Transition
from sts2rl.env.types import EnvStep
from test_episode_runner import FakeEnv, _runner
from test_ppo_agent import _agent, _map_state, _observation, _step


def _constant_value(agent, value):
    with torch.no_grad():
        for parameter in agent.game_encoder.value_head.parameters():
            parameter.zero_()
        agent.game_encoder.value_head[-1].bias.fill_(value)


def _transition(agent, count, reward, done=False, lane=0):
    state = _observation(_map_state(count))
    action = agent.choose_action(state, lane=lane)
    agent.observe(
        Transition(state, action, reward, state, done), lane=lane
    )
    return state


def test_truncation_bootstraps_final_state_but_never_next_episode():
    agent = _agent(rollout_size=1000)
    _constant_value(agent, 0)
    final = _transition(agent, 2, 0)
    # V(final) = 7; V(next episode) will deliberately be very different.
    _constant_value(agent, 7)
    agent.finish_episode(final, truncated=True)
    _constant_value(agent, 50)
    agent.reset(final)
    _transition(agent, 2, 100, done=True)
    advantages, returns = agent._advantages_and_returns(agent._lane(0).steps)
    assert float(advantages[0]) == pytest.approx(agent.config.gamma * 7)
    assert float(returns[0]) == pytest.approx(agent.config.gamma * 7)


def test_in_flight_value_keeps_sampling_scale_across_an_actual_update():
    agent = _agent(rollout_size=1000)
    agent._return_scale.var = 4.0  # scale = 2
    _constant_value(agent, 3)
    state = _observation(_map_state(2))
    action = agent.choose_action(state, lane=1)  # raw value = 6
    _step(agent, 0, reward=20, done=True)
    agent.update()  # changes both the model and the scale
    assert agent._return_scale.scale > 10
    agent.observe(Transition(state, action, 0, state, True), lane=1)
    advantages, _ = agent._advantages_and_returns(agent._lane(1).steps)
    assert float(advantages[0]) == pytest.approx(-6)


def test_forced_rewards_keep_environment_step_discount():
    agent = _agent(rollout_size=1000)
    agent.config = replace(agent.config, gamma=0.9, gae_lambda=1.0)
    _constant_value(agent, 0)
    _transition(agent, 2, 2)
    _transition(agent, 1, 4, done=True)
    advantages, returns = agent._advantages_and_returns(agent._lane(0).steps)
    assert float(returns[0]) == pytest.approx(2 + 0.9 * 4)
    assert float(advantages[0]) == pytest.approx(5.6)


def test_forced_reward_after_update_is_not_credited_to_next_episode():
    agent = _agent(rollout_size=1)
    _transition(agent, 2, 0)  # flushes the preceding decision
    assert agent.optimizer_updates == 1
    agent.config = replace(agent.config, rollout_size=1000)
    final = _transition(agent, 1, 4, done=True)
    agent.finish_episode(final, truncated=False)
    agent.reset(final)
    _transition(agent, 2, 0, done=True)
    assert agent._lane(0).steps[-1].reward == 0
    assert sum(step.reward for step in agent._lane(0).steps) == 4


def test_gae_matches_a_full_step_reference_with_forced_actions():
    agent = _agent(rollout_size=1000)
    agent.config = replace(agent.config, gamma=0.9, gae_lambda=0.8)
    rewards, values = [2, 4, 8], [1, 3, 5]
    for count, reward, value, done in zip(
        [2, 1, 2], rewards, values, [False, False, True]
    ):
        _constant_value(agent, value)
        _transition(agent, count, reward, done)
    advantages, returns = agent._advantages_and_returns(agent._lane(0).steps)
    expected = [0.0] * 3
    next_value = gae = 0.0
    for index in reversed(range(3)):
        delta = rewards[index] + 0.9 * next_value - values[index]
        gae = delta + 0.9 * 0.8 * gae
        expected[index] = gae
        next_value = values[index]
    assert advantages.tolist() == pytest.approx(expected)
    assert returns.tolist() == pytest.approx(
        [a + v for a, v in zip(expected, values)]
    )


def test_forced_steps_do_not_dilute_policy_advantage_normalization():
    agent = _agent(rollout_size=1000)
    _constant_value(agent, 0)
    for count, reward in [(2, 1), (1, 100), (2, 3)]:
        _transition(agent, count, reward, done=True)
    captured = []

    def capture(indices, steps, advantages, returns):
        captured.append(advantages.tolist())
        return {}

    agent._optimize_minibatch = capture
    agent.update()
    assert captured[0] == pytest.approx([-1, 0, 1])


def test_only_forced_steps_can_train_critic_without_policy_gradient():
    agent = _agent(rollout_size=2)
    _constant_value(agent, 0)
    before = agent.game_encoder.value_head[-1].bias.detach().clone()
    _transition(agent, 1, 2)
    _transition(agent, 1, 4, done=True)
    assert agent.optimizer_updates == 1
    assert agent.last_update['policy_loss'] == 0
    assert agent.last_update['entropy'] == 0
    assert agent.game_encoder.action_type_embedding.weight.grad is None
    assert not torch.equal(before, agent.game_encoder.value_head[-1].bias)


def test_forced_step_after_truncation_does_not_extend_previous_episode():
    agent = _agent(rollout_size=1000)
    _constant_value(agent, 0)
    final = _transition(agent, 2, 1)
    agent.finish_episode(final, truncated=True)
    agent.reset(final)
    _transition(agent, 1, 100)
    assert agent._lane(0).steps[0].reward == 1
    advantages, _ = agent._advantages_and_returns(agent._lane(0).steps)
    assert float(advantages[0]) == pytest.approx(1)


@pytest.mark.parametrize("gamma", [0.0, 0.9, 1.0])
@pytest.mark.parametrize("gae_lambda", [0.0, 0.8, 1.0])
def test_boundary_masks_match_independent_finite_sum(gamma, gae_lambda):
    agent = _agent(rollout_size=1000)
    agent.config = replace(agent.config, gamma=gamma, gae_lambda=gae_lambda)
    # Episode 1 truncates after a forced action; episode 2 terminates;
    # episode 3 ends at a nonterminal rollout boundary.
    rewards = [2, -1, 100, 3]
    values = [1, 4, 20, 5]
    for index, (reward, value) in enumerate(zip(rewards, values)):
        _constant_value(agent, value)
        state = _transition(agent, 1 if index == 1 else 2, reward, index == 2)
        if index == 1:
            _constant_value(agent, 7)
            agent.finish_episode(state, truncated=True)
            agent.reset(state)
        elif index == 2:
            agent.finish_episode(state, truncated=False)
            agent.reset(state)
    _constant_value(agent, 11)
    next_values = [4, 7, 0, 11]
    deltas = [r + gamma * n - v for r, n, v in zip(rewards, next_values, values)]
    # Independent definition: sum discounted TD residuals up to each boundary.
    ends = [1, 1, 2, 3]
    expected = [
        sum((gamma * gae_lambda) ** (j - i) * deltas[j] for j in range(i, end + 1))
        for i, end in enumerate(ends)
    ]
    advantages, returns = agent._advantages_and_returns(agent._lane(0).steps)
    assert advantages.tolist() == pytest.approx(expected)
    assert returns.tolist() == pytest.approx([a + v for a, v in zip(expected, values)])


def test_episode_runner_step_limit_cuts_gae_before_reset():
    class ContinuingEnv(FakeEnv):
        def reset(self, spec=None):
            self.state = _map_state(2)
            return self.state

        def step(self, action):
            self.state = _map_state(2)
            self.state["run"]["floor"] = 2
            return EnvStep(self.state, done=False, info={})

    agent = _agent(rollout_size=1000)
    _constant_value(agent, 0)
    runner = _runner(ContinuingEnv(), agent, max_steps=1)
    assert runner.run().truncation_reason == "step_limit"
    assert runner.run().truncation_reason == "step_limit"
    advantages, _ = agent._advantages_and_returns(agent._lane(0).steps)
    assert advantages.tolist() == pytest.approx([1.5, 1.5])


def test_discarding_a_forced_action_does_not_create_a_transition():
    agent = _agent(rollout_size=1000)
    state = _observation(_map_state(1))
    agent.choose_action(state)
    agent.discard_decision()
    assert agent._lane(0).pending is None
    assert not agent._lane(0).steps
    _transition(agent, 1, 1, done=True)
    assert len(agent._lane(0).steps) == 1


def test_truncation_after_automatic_update_does_not_close_an_old_episode():
    agent = _agent(rollout_size=1)
    state = _transition(agent, 2, 1)
    assert agent.optimizer_updates == 1
    agent.finish_episode(state, truncated=True)
    agent.reset(state)
    assert not agent._lane(0).steps


def test_reset_without_finish_closes_the_previous_trajectory():
    agent = _agent(rollout_size=1000)
    _constant_value(agent, 0)
    state = _transition(agent, 2, 1)
    agent.reset(state)
    _transition(agent, 2, 100, done=True)
    advantages, _ = agent._advantages_and_returns(agent._lane(0).steps)
    assert advantages.tolist() == pytest.approx([1, 100])
