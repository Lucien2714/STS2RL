"""Multi-episode trainer lifecycle tests without network access."""

from __future__ import annotations

from dataclasses import replace

import pytest

from sts2rl.actions import GameAction
from sts2rl.agents import EpisodeResult, Transition
from sts2rl.env import GameObservation, ResetSpec
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

    def save_episode(
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
