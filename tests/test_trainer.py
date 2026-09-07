"""Multi-episode trainer lifecycle tests without network access."""

from __future__ import annotations

from dataclasses import replace
import time

import pytest

from sts2rl.actions import GameAction
from sts2rl.agents import EpisodeResult, Transition
from sts2rl.env import GameObservation, ResetSpec
from sts2rl.env.mcp_client import STS2ClientError
from sts2rl.training import EpisodeMetrics, TrainingConfig, TrainingPlan, TrainingState
from sts2rl.training.trainer import Trainer


class FakeAgent:
    def __init__(self) -> None:
        self.environment_steps = 0
        self.optimizer_updates = 0
        self.updates: list[dict[str, float]] = []
        self.abort_count = 0

    def drain_update_metrics(self) -> tuple[dict[str, float], ...]:
        result = tuple(self.updates)
        self.updates.clear()
        return result

    def abort_episode(self) -> None:
        self.abort_count += 1

    def update(self) -> dict[str, float]:
        self.flush_count = getattr(self, "flush_count", 0) + 1
        return {}


class FakeRunner:
    def __init__(
        self,
        agent: FakeAgent,
        *,
        steps: int = 3,
        failure: BaseException | None = None,
    ) -> None:
        self.agent = agent
        self.steps = steps
        self.failure = failure
        self.calls = 0
        self.reset_specs: list[object] = []
        self.reused_run = False

    def run(self, reset_spec: object) -> EpisodeResult:
        self.reset_specs.append(reset_spec)
        self.calls += 1
        self.agent.environment_steps += self.steps
        self.agent.optimizer_updates += 1
        self.agent.updates.append(
            {
                "environment_steps": float(self.agent.environment_steps),
                "optimizer_update": float(self.agent.optimizer_updates),
                "loss": 0.5,
                "rollout_steps": float(self.steps),
            }
        )
        if self.failure is not None:
            raise self.failure
        return _result(
            self.steps,
            action_error=self.calls == 1,
            reused_run=self.reused_run,
        )


class FakeCheckpointManager:
    def __init__(self) -> None:
        self.episodes: list[dict[str, int]] = []
        self.recoveries: list[tuple[str, dict[str, int]]] = []

    def save_progress(
        self,
        agent: object,
        plan: object,
        state: TrainingState,
        logging: object,
    ) -> None:
        del agent, plan, logging
        self.episodes.append(state.to_dict())

    def save_recovery(
        self,
        kind: str,
        agent: object,
        plan: object,
        state: TrainingState,
        logging: object,
    ) -> None:
        del agent, plan, logging
        self.recoveries.append((kind, state.to_dict()))


class FakeMetricsWriter:
    def __init__(self) -> None:
        self.episodes: list[EpisodeMetrics] = []
        self.updates: list[dict[str, float]] = []
        self.events: list[tuple[str, int]] = []
        self.event_payloads: list[tuple[str, dict[str, object]]] = []
        self.flush_count = 0

    def log_episode(self, metrics: EpisodeMetrics) -> None:
        self.episodes.append(metrics)

    def log_ppo_update(self, metrics: dict[str, float]) -> None:
        self.updates.append(metrics)

    def log_event(
        self,
        event_type: str,
        payload: dict[str, object],
        global_step: int,
    ) -> None:
        self.events.append((event_type, global_step))
        self.event_payloads.append((event_type, payload))

    def flush(self) -> None:
        self.flush_count += 1


def _plan(total: int = 3, checkpoint_every: int = 2) -> TrainingPlan:
    return TrainingPlan(
        training=TrainingConfig(
            total_episodes=total,
            checkpoint_every=checkpoint_every,
            tensorboard_enabled=False,
        )
    )


def _result(
    steps: int,
    *,
    action_error: bool,
    error: str = "rejected",
    reused_run: bool = False,
) -> EpisodeResult:
    observation = GameObservation({"state_type": "map", "run": {"floor": 4}})
    info: dict[str, object] = {"action_error": action_error}
    if action_error:
        info["error"] = error
    transition = Transition(
        state=observation,
        action=GameAction("proceed"),
        reward=1.0,
        next_state=observation,
        done=False,
        info=info,
    )
    return EpisodeResult(
        initial_state=observation.raw_state,
        final_state=observation.raw_state,
        transitions=tuple(transition for _ in range(steps)),
        total_reward=2.5,
        terminated=True,
        truncated=False,
        reused_run=reused_run,
    )


def test_trainer_runs_to_target_logs_updates_and_saves_periodic_and_final():
    agent = FakeAgent()
    runner = FakeRunner(agent)
    checkpoints = FakeCheckpointManager()
    metrics = FakeMetricsWriter()
    reported: list[EpisodeMetrics] = []
    trainer = Trainer(
        runner,  # type: ignore[arg-type]
        agent,  # type: ignore[arg-type]
        checkpoints,  # type: ignore[arg-type]
        metrics,  # type: ignore[arg-type]
        _plan(),
        reporter=reported.append,
    )

    state = trainer.train()

    assert state.to_dict() == {
        "completed_episodes": 3,
        "environment_steps": 9,
        "optimizer_updates": 3,
    }
    assert [item["completed_episodes"] for item in checkpoints.episodes] == [2, 3]
    assert len(metrics.episodes) == 3
    assert len(metrics.updates) == 3
    assert metrics.episodes[0].action_errors == 3
    assert metrics.episodes[-1].floor == 4
    assert reported == metrics.episodes


@pytest.mark.parametrize(
    ("failure", "kind", "event"),
    [
        (KeyboardInterrupt(), "interrupted", "training_interrupted"),
        (RuntimeError("broken"), "recovery", "training_failed"),
    ],
)
def test_trainer_aborts_and_checkpoints_interrupt_or_failure(
    failure: BaseException,
    kind: str,
    event: str,
):
    agent = FakeAgent()
    runner = FakeRunner(agent, steps=2, failure=failure)
    checkpoints = FakeCheckpointManager()
    metrics = FakeMetricsWriter()
    trainer = Trainer(
        runner,  # type: ignore[arg-type]
        agent,  # type: ignore[arg-type]
        checkpoints,  # type: ignore[arg-type]
        metrics,  # type: ignore[arg-type]
        _plan(),
    )

    with pytest.raises(type(failure)):
        trainer.train()

    assert agent.abort_count == 1
    assert checkpoints.recoveries == [
        (
            kind,
            {
                "completed_episodes": 0,
                "environment_steps": 2,
                "optimizer_updates": 1,
            },
        )
    ]
    assert metrics.events == [(event, 2)]
    assert len(metrics.updates) == 1


def test_trainer_resume_does_not_repeat_completed_episodes():
    agent = FakeAgent()
    agent.environment_steps = 20
    agent.optimizer_updates = 4
    state = TrainingState(2, 20, 4)
    runner = FakeRunner(agent, steps=1)
    plan = _plan(total=3)
    trainer = Trainer(
        runner,  # type: ignore[arg-type]
        agent,  # type: ignore[arg-type]
        FakeCheckpointManager(),  # type: ignore[arg-type]
        FakeMetricsWriter(),  # type: ignore[arg-type]
        plan,
        state,
    )

    restored = trainer.train()

    assert runner.calls == 1
    assert restored.completed_episodes == 3


def test_trainer_rejects_target_below_restored_progress():
    agent = FakeAgent()
    state = TrainingState(completed_episodes=2)
    trainer = Trainer(
        FakeRunner(agent),  # type: ignore[arg-type]
        agent,  # type: ignore[arg-type]
        FakeCheckpointManager(),  # type: ignore[arg-type]
        FakeMetricsWriter(),  # type: ignore[arg-type]
        replace(_plan(), training=replace(_plan().training, total_episodes=1)),
        state,
    )

    with pytest.raises(ValueError, match="below"):
        trainer.train()

def _payloads(writer: FakeMetricsWriter, event_type: str) -> list[dict[str, object]]:
    return [payload for name, payload in writer.event_payloads if name == event_type]


def test_rejected_actions_are_logged_with_their_messages():
    """A count alone cannot diagnose a rejection after the run has moved on."""
    agent = FakeAgent()
    runner = FakeRunner(agent, steps=2)
    writer = FakeMetricsWriter()
    trainer = Trainer(
        runner,
        agent,
        FakeCheckpointManager(),
        writer,
        _plan(total=1),
    )

    trainer.train()

    events = _payloads(writer, "action_errors")
    assert len(events) == 1
    rejections = events[0]["rejections"]
    assert len(rejections) == 1
    assert rejections[0]["error"] == "rejected"
    assert rejections[0]["count"] == 2
    assert rejections[0]["action"] == {"type": "proceed"}
    assert rejections[0]["state_type"] == "map"


def test_no_action_error_event_when_every_action_was_accepted():
    agent = FakeAgent()
    runner = FakeRunner(agent, steps=2)
    runner.calls = 1  # _result only flags errors on the first call
    writer = FakeMetricsWriter()
    trainer = Trainer(
        runner,
        agent,
        FakeCheckpointManager(),
        writer,
        _plan(total=1),
    )

    trainer.train()

    assert _payloads(writer, "action_errors") == []


def _seeded_plan(seeds: tuple[str, ...], total: int) -> TrainingPlan:
    return TrainingPlan(
        training=TrainingConfig(
            total_episodes=total,
            checkpoint_every=total,
            tensorboard_enabled=False,
            training_seeds=seeds,
        ),
        reset=ResetSpec(game_mode="custom"),
    )


def _seeded_trainer(plan: TrainingPlan, state: TrainingState | None = None):
    agent = FakeAgent()
    runner = FakeRunner(agent)
    return runner, Trainer(
        runner,  # type: ignore[arg-type]
        agent,  # type: ignore[arg-type]
        FakeCheckpointManager(),  # type: ignore[arg-type]
        FakeMetricsWriter(),  # type: ignore[arg-type]
        plan,
        state,
    )


def test_each_episode_runs_the_next_seed_in_the_pool():
    runner, trainer = _seeded_trainer(_seeded_plan(("AAA", "BBB"), total=5))

    trainer.train()

    assert [spec.run_seed for spec in runner.reset_specs] == [
        "AAA",
        "BBB",
        "AAA",
        "BBB",
        "AAA",
    ]


def test_a_resumed_run_picks_up_where_the_cycle_left_off():
    """Restarting the cycle would re-train early seeds and starve the rest."""
    runner, trainer = _seeded_trainer(
        _seeded_plan(("AAA", "BBB", "CCC"), total=5),
        TrainingState(completed_episodes=2),
    )

    trainer.train()

    assert [spec.run_seed for spec in runner.reset_specs] == ["CCC", "AAA", "BBB"]


def test_an_unseeded_run_passes_the_configured_reset_unchanged():
    plan = _plan()
    runner, trainer = _seeded_trainer(plan)

    trainer.train()

    assert all(spec is plan.reset for spec in runner.reset_specs)


def test_metrics_record_which_seed_each_episode_ran():
    runner, trainer = _seeded_trainer(_seeded_plan(("AAA", "BBB"), total=3))

    trainer.train()

    assert [m.seed for m in trainer.metrics_writer.episodes] == ["AAA", "BBB", "AAA"]


def test_a_reused_run_is_flagged_and_reports_no_seed():
    """Joining a run in progress ignores the seed, so claiming one would lie."""
    agent = FakeAgent()
    runner = FakeRunner(agent)
    runner.reused_run = True
    trainer = Trainer(
        runner,  # type: ignore[arg-type]
        agent,  # type: ignore[arg-type]
        FakeCheckpointManager(),  # type: ignore[arg-type]
        FakeMetricsWriter(),  # type: ignore[arg-type]
        _seeded_plan(("AAA",), total=1),
    )

    trainer.train()

    recorded = trainer.metrics_writer.episodes[0]
    assert recorded.reused_run is True
    assert recorded.seed is None


class SlowFakeRunner(FakeRunner):
    """A runner that overlaps with its peers and records when it was inside."""

    def __init__(self, agent: FakeAgent, ledger: list, name: str, delay: float):
        super().__init__(agent)
        self.ledger = ledger
        self.name = name
        self.delay = delay

    def run(self, reset_spec: object) -> EpisodeResult:
        self.ledger.append(("enter", self.name))
        time.sleep(self.delay)
        self.ledger.append(("leave", self.name))
        return super().run(reset_spec)


def _parallel_trainer(runners, plan, state=None):
    agent = runners[0].agent
    return Trainer(
        runners,  # type: ignore[arg-type]
        agent,  # type: ignore[arg-type]
        FakeCheckpointManager(),  # type: ignore[arg-type]
        FakeMetricsWriter(),  # type: ignore[arg-type]
        plan,
        state,
    )


def test_several_clients_share_one_episode_target():
    agent = FakeAgent()
    ledger: list = []
    runners = [
        SlowFakeRunner(agent, ledger, "a", 0.01),
        SlowFakeRunner(agent, ledger, "b", 0.01),
    ]
    trainer = _parallel_trainer(runners, _plan(total=6, checkpoint_every=6))

    state = trainer.train()

    assert state.completed_episodes == 6
    assert runners[0].calls + runners[1].calls == 6
    # Both clients contributed rather than one doing all the work.
    assert runners[0].calls > 0 and runners[1].calls > 0


def test_clients_actually_overlap():
    """Threads exist to buy throughput, so episodes must run concurrently."""
    agent = FakeAgent()
    ledger: list = []
    runners = [
        SlowFakeRunner(agent, ledger, "a", 0.05),
        SlowFakeRunner(agent, ledger, "b", 0.05),
    ]
    _parallel_trainer(runners, _plan(total=4, checkpoint_every=4)).train()

    depth = 0
    peak = 0
    for kind, _ in ledger:
        depth += 1 if kind == "enter" else -1
        peak = max(peak, depth)
    assert peak == 2


def test_a_checkpoint_drains_every_client_first():
    """A checkpoint needs an empty rollout, so no episode may be in flight."""
    agent = FakeAgent()
    ledger: list = []
    runners = [
        SlowFakeRunner(agent, ledger, "a", 0.02),
        SlowFakeRunner(agent, ledger, "b", 0.02),
    ]
    trainer = _parallel_trainer(runners, _plan(total=4, checkpoint_every=2))
    original = trainer._save_checkpoint

    def recording_save() -> None:
        ledger.append(("checkpoint", "-"))
        original()

    trainer._save_checkpoint = recording_save  # type: ignore[method-assign]
    trainer.train()

    depth = 0
    for kind, _ in ledger:
        if kind == "enter":
            depth += 1
        elif kind == "leave":
            depth -= 1
        else:
            assert depth == 0, "a checkpoint ran while an episode was in flight"


def test_parallel_episodes_claim_distinct_seeds():
    agent = FakeAgent()
    ledger: list = []
    runners = [
        SlowFakeRunner(agent, ledger, "a", 0.01),
        SlowFakeRunner(agent, ledger, "b", 0.01),
    ]
    plan = _seeded_plan(("AAA", "BBB", "CCC"), total=6)
    trainer = _parallel_trainer(runners, plan)

    trainer.train()

    seeds = [spec.run_seed for r in runners for spec in r.reset_specs]
    assert sorted(seeds) == ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"]


def test_a_failing_client_stops_training_without_hanging_the_others():
    agent = FakeAgent()
    ledger: list = []
    healthy = SlowFakeRunner(agent, ledger, "a", 0.01)
    broken = SlowFakeRunner(agent, ledger, "b", 0.01)
    broken.failure = RuntimeError("client died")
    trainer = _parallel_trainer([healthy, broken], _plan(total=20, checkpoint_every=20))

    with pytest.raises(RuntimeError, match="client died"):
        trainer.train()

    assert trainer.checkpoint_manager.recoveries


class CrashingRunner(FakeRunner):
    """A runner whose client dies for a fixed number of episodes."""

    def __init__(self, agent, failures: int, delay: float = 0.0):
        super().__init__(agent)
        self.failures = failures
        self.delay = delay

    def run(self, reset_spec: object) -> EpisodeResult:
        self.reset_specs.append(reset_spec)
        if self.delay:
            time.sleep(self.delay)
        if self.failures:
            self.failures -= 1
            raise STS2ClientError("client crashed")
        return super().run(reset_spec)


class LaneTrackingAgent(FakeAgent):
    def __init__(self) -> None:
        super().__init__()
        self.aborted_lanes: list[int] = []

    def abort_lane(self, lane: int) -> None:
        self.aborted_lanes.append(lane)


def _crash_trainer(runners, plan, agent, **kwargs):
    return Trainer(
        runners,  # type: ignore[arg-type]
        agent,  # type: ignore[arg-type]
        FakeCheckpointManager(),  # type: ignore[arg-type]
        FakeMetricsWriter(),  # type: ignore[arg-type]
        plan,
        **kwargs,
    )


def test_a_crashed_client_costs_its_episode_not_the_job():
    """Both clients are slow enough that both are certain to be dispatched."""
    agent = LaneTrackingAgent()
    first = CrashingRunner(agent, failures=1, delay=0.02)
    second = CrashingRunner(agent, failures=1, delay=0.02)
    trainer = _crash_trainer(
        [first, second], _plan(total=6, checkpoint_every=6), agent
    )

    state = trainer.train()

    assert state.completed_episodes == 6
    # Each client dropped the decision left in flight by its crash.
    assert sorted(agent.aborted_lanes) == [0, 1]


def test_a_failed_episode_does_not_consume_the_target():
    """Counting a crash as an episode would end training short of the target."""
    agent = LaneTrackingAgent()
    crashy = CrashingRunner(agent, failures=2, delay=0.01)
    trainer = _crash_trainer(
        [crashy, CrashingRunner(agent, failures=0, delay=0.01)],
        _plan(total=4, checkpoint_every=4),
        agent,
    )

    state = trainer.train()

    assert state.completed_episodes == 4
    assert len(trainer.metrics_writer.episodes) == 4


def test_a_client_that_never_recovers_is_given_up_on():
    agent = LaneTrackingAgent()
    broken = CrashingRunner(agent, failures=999)
    trainer = _crash_trainer(
        [broken], _plan(total=5, checkpoint_every=5), agent, max_episode_failures=2
    )

    with pytest.raises(STS2ClientError, match="client crashed"):
        trainer.train()

    assert len(broken.reset_specs) == 2


def test_the_other_clients_finish_when_one_is_given_up_on():
    agent = LaneTrackingAgent()
    broken = CrashingRunner(agent, failures=999)
    healthy = CrashingRunner(agent, failures=0, delay=0.01)
    trainer = _crash_trainer(
        [broken, healthy], _plan(total=5, checkpoint_every=5), agent,
        max_episode_failures=2,
    )

    state = trainer.train()

    assert state.completed_episodes == 5
    assert healthy.calls == 5


def test_a_programming_error_is_still_fatal_immediately():
    """Only the game boundary is treated as flaky; our own bugs are not."""
    agent = LaneTrackingAgent()
    broken = FakeRunner(agent)
    broken.failure = TypeError("bug in our code")
    trainer = _crash_trainer([broken, FakeRunner(agent)], _plan(total=20, checkpoint_every=20), agent)

    with pytest.raises(TypeError, match="bug in our code"):
        trainer.train()


class UpdatingRunner(FakeRunner):
    """A runner whose episodes produce a controllable number of updates."""

    def __init__(self, agent, updates_per_episode: int):
        super().__init__(agent)
        self.updates_per_episode = updates_per_episode

    def run(self, reset_spec: object) -> EpisodeResult:
        self.reset_specs.append(reset_spec)
        self.calls += 1
        self.agent.environment_steps += self.steps
        for _ in range(self.updates_per_episode):
            self.agent.optimizer_updates += 1
            self.agent.updates.append(
                {
                    "environment_steps": float(self.agent.environment_steps),
                    "optimizer_update": float(self.agent.optimizer_updates),
                    "loss": 0.5,
                    "rollout_steps": 256.0,
                }
            )
        return _result(self.steps, action_error=False)


def test_checkpoints_are_spaced_by_optimizer_updates_not_episodes():
    """Episode length quadruples over a run; an update is a fixed amount."""
    agent = FakeAgent()
    runner = UpdatingRunner(agent, updates_per_episode=1)
    checkpoints = FakeCheckpointManager()
    trainer = Trainer(
        runner,  # type: ignore[arg-type]
        agent,  # type: ignore[arg-type]
        checkpoints,  # type: ignore[arg-type]
        FakeMetricsWriter(),  # type: ignore[arg-type]
        _plan(total=9, checkpoint_every=3),
    )

    trainer.train()

    # Updates 3, 6 and 9 -- and no extra save at the end, since 9 landed there.
    assert len(checkpoints.episodes) == 3


def test_an_episode_that_spans_several_updates_saves_once():
    agent = FakeAgent()
    runner = UpdatingRunner(agent, updates_per_episode=5)
    checkpoints = FakeCheckpointManager()
    trainer = Trainer(
        runner,  # type: ignore[arg-type]
        agent,  # type: ignore[arg-type]
        checkpoints,  # type: ignore[arg-type]
        FakeMetricsWriter(),  # type: ignore[arg-type]
        _plan(total=2, checkpoint_every=2),
    )

    trainer.train()

    assert len(checkpoints.episodes) == 2


def test_a_run_that_never_updates_still_saves_once_at_the_end():
    """Otherwise a short run would finish with nothing on disk."""
    agent = FakeAgent()
    runner = UpdatingRunner(agent, updates_per_episode=0)
    checkpoints = FakeCheckpointManager()
    trainer = Trainer(
        runner,  # type: ignore[arg-type]
        agent,  # type: ignore[arg-type]
        checkpoints,  # type: ignore[arg-type]
        FakeMetricsWriter(),  # type: ignore[arg-type]
        _plan(total=3, checkpoint_every=5),
    )

    trainer.train()

    assert len(checkpoints.episodes) >= 1
