"""Multi-episode trainer lifecycle tests without network access."""

from __future__ import annotations

from dataclasses import replace

import pytest

from sts2rl.actions import GameAction
from sts2rl.agents import EpisodeResult, Transition
from sts2rl.env import GameObservation
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

    def run(self, reset_spec: object) -> EpisodeResult:
        del reset_spec
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
        return _result(self.steps, action_error=self.calls == 1)


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
        del payload
        self.events.append((event_type, global_step))

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


def _result(steps: int, *, action_error: bool) -> EpisodeResult:
    observation = GameObservation({"state_type": "map", "run": {"floor": 4}})
    transition = Transition(
        state=observation,
        action=GameAction("proceed"),
        reward=1.0,
        next_state=observation,
        done=False,
        info={"action_error": action_error},
    )
    return EpisodeResult(
        initial_state=observation.raw_state,
        final_state=observation.raw_state,
        transitions=tuple(transition for _ in range(steps)),
        total_reward=2.5,
        terminated=True,
        truncated=False,
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
