"""Evaluation scheduling, scoring, and the training-versus-holdout comparison."""

from __future__ import annotations

import time

import pytest

from sts2rl.actions import GameAction
from sts2rl.agents import EpisodeResult, Transition
from sts2rl.env import GameObservation, ResetSpec
from sts2rl.env.mcp_client import STS2ClientError
from sts2rl.training.evaluate import (
    EpisodeScore,
    Evaluator,
    build_schedule,
    format_report,
    summarize,
)


class FakeAgent:
    def __init__(self) -> None:
        self.eval_calls = 0
        self.aborted_lanes: list[int] = []

    def eval(self) -> None:
        self.eval_calls += 1

    def abort_lane(self, lane: int) -> None:
        self.aborted_lanes.append(lane)


class FakeRunner:
    """Returns a floor derived from the seed, so scores are checkable."""

    def __init__(self, agent, floors: dict[str, int], delay: float = 0.0,
                 failures: int = 0, reused_once: bool = False,
                 always_reused: bool = False):
        self.agent = agent
        self.floors = floors
        self.delay = delay
        self.failures = failures
        self.reused_once = reused_once
        self.always_reused = always_reused
        self.seeds: list[str] = []

    def run(self, reset_spec: ResetSpec) -> EpisodeResult:
        self.seeds.append(reset_spec.run_seed)
        if self.delay:
            time.sleep(self.delay)
        if self.failures:
            self.failures -= 1
            raise STS2ClientError("client crashed")
        floor = self.floors[reset_spec.run_seed]
        observation = GameObservation({"state_type": "map", "run": {"floor": floor}})
        transition = Transition(
            state=observation,
            action=GameAction("proceed"),
            reward=1.0,
            next_state=observation,
            done=True,
            info={},
        )
        reused = self.always_reused or self.reused_once
        self.reused_once = False
        return EpisodeResult(
            initial_state=observation.raw_state,
            final_state=observation.raw_state,
            transitions=(transition,) * floor,
            total_reward=float(floor),
            terminated=True,
            truncated=False,
            reused_run=reused,
        )


def _evaluator(runners, agent, **kwargs):
    return Evaluator(runners, agent, ResetSpec(), **kwargs)


def test_the_schedule_covers_both_pools_evenly():
    schedule = build_schedule(("A", "B"), ("X",), episodes_per_seed=2)

    assert schedule.count(("A", "training")) == 2
    assert schedule.count(("X", "holdout")) == 2
    assert len(schedule) == 6


def test_the_schedule_interleaves_so_a_short_run_measures_both():
    """Stopping halfway must leave both pools half measured, not one unmeasured."""
    schedule = build_schedule(("A", "B"), ("X",), episodes_per_seed=3)

    first_round = schedule[:3]

    assert {pool for _, pool in first_round} == {"training", "holdout"}


def test_a_seed_in_both_pools_is_refused():
    """A holdout seed that was trained on measures nothing at all."""
    with pytest.raises(ValueError, match="also trained on"):
        build_schedule(("A", "B"), ("B",), episodes_per_seed=1)


def test_an_empty_schedule_is_refused():
    with pytest.raises(ValueError, match="both seed pools are empty"):
        build_schedule((), (), episodes_per_seed=1)


def test_evaluation_freezes_the_policy_before_playing():
    agent = FakeAgent()
    runner = FakeRunner(agent, {"A": 5})

    _evaluator([runner], agent).run([("A", "training")])

    assert agent.eval_calls == 1


def test_every_scheduled_episode_is_played_once():
    agent = FakeAgent()
    runner = FakeRunner(agent, {"A": 4, "X": 2})

    scores = _evaluator([runner], agent).run(
        [("A", "training"), ("X", "holdout"), ("A", "training")]
    )

    assert sorted(runner.seeds) == ["A", "A", "X"]
    assert len(scores) == 3


def test_clients_share_the_schedule():
    agent = FakeAgent()
    first = FakeRunner(agent, {"A": 3, "B": 3}, delay=0.02)
    second = FakeRunner(agent, {"A": 3, "B": 3}, delay=0.02)

    scores = _evaluator([first, second], agent).run(
        [("A", "training"), ("B", "training")] * 3
    )

    assert len(scores) == 6
    assert first.seeds and second.seeds


def test_a_crashed_episode_is_replayed_rather_than_scored():
    """Scoring an episode the client killed would report a floor nobody reached."""
    agent = FakeAgent()
    runner = FakeRunner(agent, {"A": 7}, failures=1)

    scores = _evaluator([runner], agent).run([("A", "training")])

    assert [score.floor for score in scores] == [7]
    assert runner.seeds == ["A", "A"]
    assert agent.aborted_lanes == [0]


def test_a_client_that_keeps_crashing_fails_the_evaluation():
    agent = FakeAgent()
    runner = FakeRunner(agent, {"A": 7}, failures=99)

    with pytest.raises(STS2ClientError, match="client crashed"):
        _evaluator([runner], agent, max_episode_failures=2).run([("A", "training")])


def test_evaluation_always_uses_custom_mode():
    """Standard single-player ignores a seed, so it would score a random map."""
    agent = FakeAgent()
    runner = FakeRunner(agent, {"A": 3})
    evaluator = Evaluator(runner_list := [runner], agent, ResetSpec())
    del runner_list

    evaluator.run([("A", "training")])

    assert evaluator._spec_for("A").game_mode == "custom"
    assert evaluator._spec_for("A").run_seed == "A"


def _scores(pairs):
    return [
        EpisodeScore(seed=seed, pool=pool, reward=float(floor), floor=floor,
                     steps=floor, terminated=True)
        for seed, pool, floor in pairs
    ]


def test_the_summary_separates_the_pools():
    scores = _scores([("A", "training", 10), ("B", "training", 12),
                      ("X", "holdout", 4)])

    summary = summarize(scores)

    assert summary["training"]["floor"] == 11.0
    assert summary["holdout"]["floor"] == 4.0


def test_the_report_states_the_gap_that_answers_the_question():
    scores = _scores([("A", "training", 10), ("X", "holdout", 4)])

    report = format_report(scores)

    assert "floor gap (training - holdout): +6.00" in report
    assert "memorised" in report


def test_a_generalising_policy_reports_a_gap_near_zero():
    scores = _scores([("A", "training", 9), ("X", "holdout", 9)])

    assert "floor gap (training - holdout): +0.00" in format_report(scores)


def test_the_report_survives_having_nothing_to_report():
    assert format_report([]) == "no episodes were scored"


def test_a_joined_run_is_played_out_but_never_scored():
    """Training leaves its clients mid-run and the game cannot quit one, so a
    leftover has to be cleared by playing it -- but scoring it would file
    another map's result under this seed's name."""
    agent = FakeAgent()
    runner = FakeRunner(agent, {"A": 4}, reused_once=True)

    scores = _evaluator([runner], agent).run([("A", "training")])

    assert runner.seeds == ["A", "A"]
    assert [score.floor for score in scores] == [4]


def test_a_client_that_always_hands_back_someone_elses_run_fails_loudly():
    agent = FakeAgent()
    runner = FakeRunner(agent, {"A": 4}, always_reused=True)

    with pytest.raises(RuntimeError, match="did not start"):
        _evaluator([runner], agent, max_episode_failures=2).run([("A", "training")])
