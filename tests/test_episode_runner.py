"""Episode runner integration tests."""

from sts2rl.actions import GameAction
import pytest

from sts2rl.agents import (
    Agent,
    EpisodeRunner,
    LegalActionProvider,
    ObservationError,
    Transition,
)
from sts2rl.env.mcp_client import STS2ClientError
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
        self.detail_calls = 0

    def reset(self, spec=None):
        return self.state

    def get_state(self):
        return self.state

    def get_player_detail(self):
        self.detail_calls += 1
        return {
            "state_type": "player_detail",
            "player": {"deck": [{"id": "UPPERCUT"}]},
        }

    def step(self, action: GameAction):
        assert action.to_dict() == {"type": "choose_map_node", "index": 0}
        self.state = {"state_type": "game_over"}
        return EnvStep(self.state, done=True, info={"action_error": False})


def test_runner_connects_env_reward_and_agent_until_game_over():
    env = FakeEnv()
    agent = RecordingAgent()
    reward = FixedReward()

    result = EpisodeRunner(env, agent, reward_model=reward).run()

    assert result.steps == 1
    assert result.total_reward == 1.5
    assert result.terminated is True
    assert result.truncated is False
    assert agent.transitions == list(result.transitions)
    assert agent.transitions[0].info["reward"]["type"] == "fixed"
    assert isinstance(agent.initial, GameObservation)
    assert agent.initial.player_detail["state_type"] == "player_detail"
    assert agent.transitions[0].state is agent.initial
    assert agent.transitions[0].next_state.player_detail is None
    assert agent.finished == (agent.transitions[0].next_state, False)
    assert env.detail_calls == 1


def test_runner_refreshes_transitional_state_before_asking_again():
    class RefreshingEnv(FakeEnv):
        def reset(self, spec=None):
            self.state = {"state_type": "treasure", "treasure": {}}
            return self.state

        def get_state(self):
            self.state = {
                "state_type": "map",
                "map": {"next_options": [{"index": 0}]},
            }
            return self.state

    result = EpisodeRunner(
        env := RefreshingEnv(),
        RecordingAgent(),
        reward_model=FixedReward(),
    ).run()

    assert result.terminated is True
    assert env.detail_calls == 2


@pytest.mark.parametrize("max_refreshes", [0, 1, 3])
def test_runner_stops_when_state_refreshes_are_exhausted(max_refreshes):
    class StuckEnv(FakeEnv):
        def __init__(self):
            super().__init__()
            self.state = {"state_type": "treasure", "treasure": {}}
            self.refreshes = 0

        def get_state(self):
            self.refreshes += 1
            return self.state

    env = StuckEnv()
    agent = RecordingAgent()
    runner = EpisodeRunner(
        env,
        agent,
        max_state_refreshes=max_refreshes,
        refresh_backoff_seconds=0.0,
    )

    result = runner.run()

    assert result.truncated is True
    assert result.terminated is False
    assert result.steps == 0
    assert env.refreshes == max_refreshes
    assert env.detail_calls == max_refreshes + 1
    assert agent.transitions == []
    assert agent.finished == (agent.initial, True)


@pytest.mark.parametrize("max_refreshes", [0, 1, 3])
def test_runner_can_choose_on_the_last_allowed_attempt(max_refreshes):
    class DelayedEnv(FakeEnv):
        def __init__(self):
            super().__init__()
            self.ready_state = self.state
            self.refreshes = 0
            if max_refreshes:
                self.state = {"state_type": "treasure", "treasure": {}}

        def get_state(self):
            self.refreshes += 1
            if self.refreshes == max_refreshes:
                self.state = self.ready_state
            return self.state

    env = DelayedEnv()
    result = EpisodeRunner(
        env,
        RecordingAgent(),
        reward_model=FixedReward(),
        max_state_refreshes=max_refreshes,
        refresh_backoff_seconds=0.0,
    ).run()

    assert result.terminated is True
    assert result.steps == 1
    assert env.refreshes == max_refreshes


def test_runner_fetches_detail_for_every_nonterminal_next_state():
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

    result = EpisodeRunner(
        env,
        agent,
        reward_model=FixedReward(),
        max_steps=1,
    ).run()

    assert result.truncated is True
    assert env.detail_calls == 2
    assert agent.transitions[0].next_state.player_detail is not None
    assert agent.finished == (agent.transitions[0].next_state, True)


def test_runner_reuses_one_player_detail_snapshot_for_a_whole_battle():
    class BattleEnv(FakeEnv):
        def __init__(self):
            super().__init__()
            self.state = _battle_state()
            self.steps = 0

        def step(self, action):
            self.steps += 1
            self.state = _battle_state()
            return EnvStep(self.state, done=False, info={"action_error": False})

    env = BattleEnv()

    result = EpisodeRunner(
        env,
        RecordingAgent(),
        reward_model=FixedReward(),
        max_steps=4,
    ).run()

    assert result.steps == 4
    assert env.detail_calls == 1


def _battle_state():
    return {
        "state_type": "monster",
        "battle": {
            "turn": "player",
            "is_play_phase": True,
            "enemies": [{"entity_id": "JAW_WORM_0", "hp": 40}],
        },
        "player": {
            "hand": [{"index": 0, "id": "Strike", "can_play": True}],
            "potions": [],
        },
    }


def test_player_detail_failure_terminates_explicitly():
    class BrokenDetailEnv(FakeEnv):
        def get_player_detail(self):
            raise STS2ClientError("detail unavailable")

    with pytest.raises(ObservationError, match="state_type='map'"):
        EpisodeRunner(
            BrokenDetailEnv(),
            RecordingAgent(),
            reward_model=FixedReward(),
        ).run()
