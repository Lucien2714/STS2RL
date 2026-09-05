"""Episode runner integration tests."""

import pytest

from sts2rl.actions import GameAction
from sts2rl.agents import (
    Agent,
    EpisodeRunner,
    LegalActionProvider,
    Transition,
)
from sts2rl.env.types import EnvStep, GameObservation


class RecordingAgent(Agent):
    def __init__(self):
        self.provider = LegalActionProvider()
        self.transitions = []
        self.finished = None
        self.initial = None

    def reset(self, initial_state):
        self.initial = initial_state

    def choose_action(self, state):
        return self.provider.require_candidates(state.raw_state)[0]

    def observe(self, transition: Transition):
        self.transitions.append(transition)

    def finish_episode(self, final_state, truncated):
        self.finished = (final_state, truncated)


class FixedReward:
    def reset(self, raw_state=None):
        self.reset_state = raw_state

    def compute(self, prev_state, next_state, action=None):
        return 1.5, {"type": "fixed", "total": 1.5}

    def action_error_reward(self, error):
        return -1.0, {"type": "error", "error": str(error), "total": -1.0}


class FakeEnv:
    def __init__(self):
        self.state = {"state_type": "map", "map": {"next_options": [{"index": 0}]}}
        self.state_reads = 0

    def reset(self, spec=None):
        return self.state

    def get_state(self):
        self.state_reads += 1
        return self.state

    def step(self, action: GameAction):
        assert action.to_dict() == {"type": "choose_map_node", "index": 0}
        self.state = {"state_type": "game_over"}
        return EnvStep(self.state, done=True, info={"action_error": False})


def _runner(env, agent, **changes):
    options = {"reward_model": FixedReward(), "refresh_backoff_seconds": 0.0}
    options.update(changes)
    return EpisodeRunner(env, agent, **options)


def test_runner_connects_env_reward_and_agent_until_game_over():
    env = FakeEnv()
    agent = RecordingAgent()

    result = _runner(env, agent).run()

    assert result.steps == 1
    assert result.total_reward == 1.5
    assert result.terminated is True
    assert result.truncated is False
    assert agent.transitions == list(result.transitions)
    assert agent.transitions[0].info["reward"]["type"] == "fixed"
    assert isinstance(agent.initial, GameObservation)
    assert agent.transitions[0].state is agent.initial
    assert agent.finished == (agent.transitions[0].next_state, False)


def test_stepping_needs_no_extra_read_because_responses_carry_the_state():
    """Action responses embed the resulting state, so a step issues no GET."""

    class NonterminalEnv(FakeEnv):
        def step(self, action):
            self.state = {
                "state_type": "map",
                "map": {"next_options": [{"index": 0}]},
                "run": {"floor": 2},
            }
            return EnvStep(self.state, done=False, info={"action_error": False})

    env = NonterminalEnv()
    agent = RecordingAgent()

    result = _runner(env, agent, max_steps=3).run()

    assert result.truncated is True
    assert result.steps == 3
    assert env.state_reads == 0
    assert agent.finished == (agent.transitions[-1].next_state, True)


def test_runner_refreshes_transitional_state_before_asking_again():
    class RefreshingEnv(FakeEnv):
        def reset(self, spec=None):
            self.state = {"state_type": "treasure", "treasure": {}}
            return self.state

        def get_state(self):
            self.state_reads += 1
            self.state = {
                "state_type": "map",
                "map": {"next_options": [{"index": 0}]},
            }
            return self.state

    env = RefreshingEnv()
    result = _runner(env, RecordingAgent()).run()

    assert result.terminated is True
    assert env.state_reads == 1


@pytest.mark.parametrize("max_refreshes", [0, 1, 3])
def test_runner_stops_when_state_refreshes_are_exhausted(max_refreshes):
    class StuckEnv(FakeEnv):
        def __init__(self):
            super().__init__()
            self.state = {"state_type": "treasure", "treasure": {}}

        def get_state(self):
            self.state_reads += 1
            return self.state

    env = StuckEnv()
    agent = RecordingAgent()

    result = _runner(env, agent, max_state_refreshes=max_refreshes).run()

    assert result.truncated is True
    assert result.terminated is False
    assert result.steps == 0
    assert env.state_reads == max_refreshes
    assert agent.transitions == []
    assert agent.finished == (agent.initial, True)


@pytest.mark.parametrize("max_refreshes", [0, 1, 3])
def test_runner_can_choose_on_the_last_allowed_attempt(max_refreshes):
    class DelayedEnv(FakeEnv):
        def __init__(self):
            super().__init__()
            self.ready_state = self.state
            if max_refreshes:
                self.state = {"state_type": "treasure", "treasure": {}}

        def get_state(self):
            self.state_reads += 1
            if self.state_reads == max_refreshes:
                self.state = self.ready_state
            return self.state

    env = DelayedEnv()
    result = _runner(env, RecordingAgent(), max_state_refreshes=max_refreshes).run()

    assert result.terminated is True
    assert result.steps == 1
    assert env.state_reads == max_refreshes
