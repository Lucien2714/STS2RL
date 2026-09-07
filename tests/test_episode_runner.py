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


def _deck(count=10):
    return {
        "in_run": True,
        "deck": {
            "count": count,
            "unique_count": 1,
            "upgraded_count": 0,
            "cards": [
                {
                    "id": "STRIKE_IRONCLAD",
                    "name": "Strike",
                    "type": "Attack",
                    "rarity": "Basic",
                    "cost": "1",
                    "quantity": count,
                    "current_upgrade_level": 0,
                }
            ],
        },
    }


class FakeEnv:
    def __init__(self):
        self.state = {"state_type": "map", "map": {"next_options": [{"index": 0}]}}
        self.state_reads = 0
        self.deck_reads = 0

    def reset(self, spec=None):
        return self.state

    def get_state(self):
        self.state_reads += 1
        return self.state

    def get_player_detail(self):
        self.deck_reads += 1
        return _deck()

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


def _battle_state():
    return {
        "state_type": "monster",
        "battle": {
            "turn": "player",
            "is_play_phase": True,
            "enemies": [{"entity_id": "JAW_WORM_0", "hp": 40}],
        },
        "player": {"hand": [{"index": 0, "id": "Strike", "can_play": True}]},
    }


def test_deck_is_read_once_per_battle_but_refreshed_off_battle():
    """The deck cannot change mid-battle, so one read covers the whole fight."""

    class BattleEnv(FakeEnv):
        def __init__(self):
            super().__init__()
            self.state = _battle_state()

        def step(self, action):
            self.state = _battle_state()
            return EnvStep(self.state, done=False, info={"action_error": False})

    env = BattleEnv()
    _runner(env, RecordingAgent(), max_steps=4).run()
    assert env.deck_reads == 1

    off_battle = FakeEnv()

    class MapEnv(FakeEnv):
        def step(self, action):
            self.state = {
                "state_type": "map",
                "map": {"next_options": [{"index": 0}]},
            }
            return EnvStep(self.state, done=False, info={"action_error": False})

    off_battle = MapEnv()
    _runner(off_battle, RecordingAgent(), max_steps=4).run()
    assert off_battle.deck_reads == 5


def test_deck_reaches_the_agent_and_becomes_deck_zone_rows():
    env = FakeEnv()
    agent = RecordingAgent()

    _runner(env, agent).run()

    assert agent.initial.player_detail["deck"]["count"] == 10


def test_a_build_without_the_deck_endpoint_degrades_and_is_not_retried():
    from sts2rl.env.mcp_client import STS2ClientError

    class NoDeckEnv(FakeEnv):
        def get_player_detail(self):
            self.deck_reads += 1
            raise STS2ClientError("HTTP 404: Not found")

        def step(self, action):
            self.state = {
                "state_type": "map",
                "map": {"next_options": [{"index": 0}]},
            }
            return EnvStep(self.state, done=False, info={"action_error": False})

    env = NoDeckEnv()
    agent = RecordingAgent()

    with pytest.warns(RuntimeWarning, match="master deck"):
        result = _runner(env, agent, max_steps=3).run()

    assert result.steps == 3
    assert env.deck_reads == 1
    assert agent.initial.player_detail is None


def test_combat_is_given_far_longer_than_one_animation_to_settle(monkeypatch):
    """A truncated episode leaves the run alive for the next reset to join."""
    import sts2rl.agents.runner as runner_module

    slept: list[float] = []
    monkeypatch.setattr(runner_module.time, "sleep", slept.append)

    assert runner_module.MAX_STATE_REFRESHES >= 10
    total = sum(
        min(runner_module.REFRESH_BACKOFF_SECONDS * (2**attempt),
            runner_module.MAX_REFRESH_BACKOFF_SECONDS)
        for attempt in range(runner_module.MAX_STATE_REFRESHES)
    )
    # The old budget was 1.75s, which multi-enemy turns outlasted.
    assert total > 8.0


def test_the_refresh_backoff_is_capped(monkeypatch):
    """Doubling without a cap would stall a worker for minutes."""
    import sts2rl.agents.runner as runner_module

    longest = min(
        runner_module.REFRESH_BACKOFF_SECONDS * (2**20),
        runner_module.MAX_REFRESH_BACKOFF_SECONDS,
    )

    assert longest == runner_module.MAX_REFRESH_BACKOFF_SECONDS


class RefusingEnv:
    """An env whose screen refuses the action until it finally opens."""

    def __init__(self, refusals: int):
        self.refusals = refusals
        self.steps = 0
        self.screen = {"state_type": "rest_site",
                       "rest_site": {"options": [{"index": 0}], "can_proceed": False},
                       "run": {"floor": 3}}
        self.reused_active_run = False

    def reset(self, spec=None):
        return self.screen

    def get_state(self):
        return self.screen

    def step(self, action):
        self.steps += 1
        if self.refusals:
            self.refusals -= 1
            # Refused, and the screen is exactly as it was.
            return EnvStep(
                raw_state=self.screen,
                done=False,
                info={"action_error": True, "error": "Rest site room is not open"},
            )
        done_state = {"state_type": "game_over", "run": {"floor": 4}}
        return EnvStep(raw_state=done_state, done=True, info={"action_error": False})

    def get_player_detail(self):
        return None


class CountingAgent:
    def __init__(self) -> None:
        self.observed = 0
        self.discarded = 0

    def reset(self, observation) -> None:
        pass

    def choose_action(self, observation):
        return GameAction("choose_rest_option", index=0)

    def observe(self, transition) -> None:
        self.observed += 1

    def discard_decision(self) -> None:
        self.discarded += 1

    def finish_episode(self, final_state, truncated) -> None:
        pass


def test_a_refusal_that_moves_nothing_is_retried_not_learned_from():
    """Recording it would teach that resting at a rest site does nothing."""
    env = RefusingEnv(refusals=3)
    agent = CountingAgent()
    runner = EpisodeRunner(env, agent, refresh_backoff_seconds=0.0)

    result = runner.run()

    assert agent.discarded == 3
    assert agent.observed == 1
    assert result.steps == 1
    assert result.terminated is True


def test_a_screen_that_never_opens_ends_the_episode():
    """Otherwise a deterministic policy spends the whole step budget on it."""
    env = RefusingEnv(refusals=10_000)
    agent = CountingAgent()
    runner = EpisodeRunner(
        env, agent, max_state_refreshes=3, refresh_backoff_seconds=0.0
    )

    result = runner.run()

    assert agent.observed == 0
    assert result.truncated is True
    assert env.steps <= 5
