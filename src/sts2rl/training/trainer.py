"""Multi-episode orchestration around the single-episode runner."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
import threading
from time import monotonic

from sts2rl.agents import CandidatePPOAgent, EpisodeResult, EpisodeRunner
from sts2rl.training.checkpoint import CheckpointManager
from sts2rl.env import ResetSpec
from sts2rl.env.mcp_client import STS2ClientError
from sts2rl.training.config import (
    MAX_EPISODE_FAILURES,
    TrainingPlan,
    TrainingState,
)
from sts2rl.training.metrics import EpisodeMetrics, TrainingMetricsWriter


class _EpisodeGate:
    """Let episodes run freely, and drain them when a checkpoint is due.

    A checkpoint needs an empty rollout and no unobserved action, and with
    several clients playing there is no moment when that is true by luck.  The
    gate closes the door on new episodes and waits for the ones in flight, so
    the boundary is created rather than hoped for.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._open = True
        self._active = 0

    def enter(self) -> None:
        with self._condition:
            while not self._open:
                self._condition.wait()
            self._active += 1

    def leave(self) -> None:
        with self._condition:
            self._active -= 1
            self._condition.notify_all()

    def close_and_drain(self) -> None:
        with self._condition:
            self._open = False
            while self._active:
                self._condition.wait()

    def open(self) -> None:
        with self._condition:
            self._open = True
            self._condition.notify_all()


class Trainer:
    """Run complete episodes, publish metrics, and save resumable boundaries."""

    def __init__(
        self,
        runner: EpisodeRunner | Sequence[EpisodeRunner],
        agent: CandidatePPOAgent,
        checkpoint_manager: CheckpointManager,
        metrics_writer: TrainingMetricsWriter,
        plan: TrainingPlan,
        state: TrainingState | None = None,
        reporter: Callable[[EpisodeMetrics], None] | None = None,
        tensorboard_log_dir: str = "tensorboard",
        max_episode_failures: int = MAX_EPISODE_FAILURES,
    ) -> None:
        self.runners: tuple[EpisodeRunner, ...] = (
            tuple(runner) if isinstance(runner, Sequence) else (runner,)
        )
        if not self.runners:
            raise ValueError("at least one runner is required")
        self.runner = self.runners[0]
        self.agent = agent
        self.checkpoint_manager = checkpoint_manager
        self.metrics_writer = metrics_writer
        self.plan = plan
        self.state = state or TrainingState()
        self.reporter = reporter
        self.tensorboard_log_dir = tensorboard_log_dir
        if max_episode_failures < 1:
            raise ValueError("max_episode_failures must be at least 1")
        self.max_episode_failures = max_episode_failures

    def train(self) -> TrainingState:
        """Train until the configured cumulative episode target is reached."""
        target = self.plan.training.total_episodes
        if target < self.state.completed_episodes:
            raise ValueError(
                "total_episodes is below the restored completed episode count"
            )
        if target == self.state.completed_episodes:
            return self.state

        if len(self.runners) > 1:
            return self._train_parallel(target)

        last_saved_update = self.state.optimizer_updates
        last_saved_episode = self.state.completed_episodes
        consecutive_failures = 0
        try:
            while self.state.completed_episodes < target:
                started_at = monotonic()
                reset_spec = self._episode_reset_spec()
                try:
                    result = self.runner.run(reset_spec)
                except STS2ClientError as exc:
                    # The game is external and does crash.  Losing the episode
                    # it died in is the price; losing the job is not.
                    consecutive_failures += 1
                    self.agent.abort_lane(0)
                    self.metrics_writer.log_event(
                        "episode_failed",
                        {
                            "lane": 0,
                            "seed": reset_spec.run_seed,
                            "consecutive_failures": consecutive_failures,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                        self.state.environment_steps,
                    )
                    if consecutive_failures >= self.max_episode_failures:
                        raise
                    continue
                consecutive_failures = 0
                self._adopt_agent_counters()
                self.state.completed_episodes += 1
                self._log_pending_updates()
                episode_metrics = self._episode_metrics(
                    result,
                    duration_seconds=monotonic() - started_at,
                    seed=None if result.reused_run else reset_spec.run_seed,
                )
                self.metrics_writer.log_episode(episode_metrics)
                self._log_action_errors(result)
                if self.reporter is not None:
                    self.reporter(episode_metrics)

                if (
                    self.state.optimizer_updates - last_saved_update
                    >= self.plan.training.checkpoint_every
                ):
                    self._save_checkpoint()
                    last_saved_update = self.state.optimizer_updates
                    last_saved_episode = self.state.completed_episodes

            # Episodes can finish without ever filling a rollout, and a run
            # that ends with nothing on disk has lost all of its work.
            if self.state.completed_episodes != last_saved_episode:
                self._save_checkpoint()
            return self.state
        except KeyboardInterrupt as exc:
            self._recover("interrupted", exc)
            raise
        except Exception as exc:
            self._recover("recovery", exc)
            raise

    def _train_parallel(self, target: int) -> TrainingState:
        """Play every client at once, sharing one agent.

        The game is the bottleneck -- a worker spends its time waiting on HTTP,
        outside the agent's lock -- so threads buy close to linear throughput
        while the tensor work stays serialized and therefore correct.
        """
        gate = _EpisodeGate()
        bookkeeping = threading.Lock()
        checkpointing = threading.Lock()
        in_flight = 0
        last_saved_update = self.state.optimizer_updates
        last_saved_episode = self.state.completed_episodes
        failure: list[BaseException] = []
        retired: dict[int, BaseException] = {}

        def worker(runner: EpisodeRunner, lane: int) -> None:
            nonlocal in_flight, last_saved_update, last_saved_episode
            consecutive_failures = 0
            while True:
                with bookkeeping:
                    if failure or self.state.completed_episodes + in_flight >= target:
                        return
                    # The seed is claimed at dispatch, not on completion, so
                    # two clients never play the same seed at the same time.
                    reset_spec = self._episode_reset_spec(
                        self.state.completed_episodes + in_flight
                    )
                    in_flight += 1

                gate.enter()
                error: STS2ClientError | None = None
                try:
                    started_at = monotonic()
                    result = runner.run(reset_spec)
                except STS2ClientError as exc:
                    # The game is external and does crash.  Its client dying is
                    # a reason to start another run, not to end a job that has
                    # cost hours.  Anything else is our own bug and stays fatal.
                    error = exc
                finally:
                    gate.leave()

                if error is not None:
                    self.agent.abort_lane(lane)
                    with bookkeeping:
                        in_flight -= 1
                        consecutive_failures += 1
                        self.metrics_writer.log_event(
                            "episode_failed",
                            {
                                "lane": lane,
                                "seed": reset_spec.run_seed,
                                "consecutive_failures": consecutive_failures,
                                "error_type": type(error).__name__,
                                "error": str(error),
                            },
                            self.state.environment_steps,
                        )
                        exhausted = (
                            consecutive_failures >= self.max_episode_failures
                        )
                        if exhausted:
                            retired[lane] = error
                    if exhausted:
                        return
                    continue

                consecutive_failures = 0
                with bookkeeping:
                    in_flight -= 1
                    self._adopt_agent_counters()
                    self.state.completed_episodes += 1
                    self._log_pending_updates()
                    episode_metrics = self._episode_metrics(
                        result,
                        duration_seconds=monotonic() - started_at,
                        seed=None if result.reused_run else reset_spec.run_seed,
                    )
                    self.metrics_writer.log_episode(episode_metrics)
                    self._log_action_errors(result)
                    if self.reporter is not None:
                        self.reporter(episode_metrics)
                    due = (
                        self.state.optimizer_updates - last_saved_update
                        >= self.plan.training.checkpoint_every
                        or self.state.completed_episodes >= target
                    )

                if due:
                    with checkpointing:
                        if last_saved_episode != self.state.completed_episodes:
                            gate.close_and_drain()
                            try:
                                self._save_checkpoint()
                                last_saved_update = self.state.optimizer_updates
                                last_saved_episode = self.state.completed_episodes
                            finally:
                                gate.open()

        def guarded(runner: EpisodeRunner, lane: int) -> None:
            try:
                worker(runner, lane)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
                with bookkeeping:
                    failure.append(exc)
                # A worker that dies holding the door shut would hang the rest.
                gate.open()

        threads = [
            threading.Thread(
                target=guarded,
                args=(runner, getattr(getattr(runner, "agent", None), "lane", index)),
                daemon=True,
            )
            for index, runner in enumerate(self.runners)
        ]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        except KeyboardInterrupt as exc:
            failure.append(exc)
            gate.open()
            for thread in threads:
                thread.join()

        if not failure and retired and self.state.completed_episodes < target:
            # Every client gave up before the target; that is a real failure.
            failure.append(next(iter(retired.values())))

        if failure:
            error = failure[0]
            kind = (
                "interrupted" if isinstance(error, KeyboardInterrupt) else "recovery"
            )
            self._recover(kind, error)
            raise error

        if self.state.completed_episodes != last_saved_episode:
            self._save_checkpoint()
        return self.state

    def _save_checkpoint(self) -> None:
        """Flush the rollout, then save; a checkpoint needs a clean boundary.

        The agent accumulates across episodes, so this is the one place that
        forces a possibly-short update.  It fires every ``checkpoint_every``
        optimizer updates, which is a fixed amount of training, rather than
        every N episodes, whose length quadruples over a run.
        """
        self.agent.update()
        self._adopt_agent_counters()
        self._log_pending_updates()
        self.metrics_writer.flush()
        self.checkpoint_manager.save_progress(
            self.agent,
            self.plan,
            self.state,
            self.tensorboard_log_dir,
        )

    def _recover(self, kind: str, original: BaseException) -> None:
        try:
            self.agent.abort_episode()
            self._adopt_agent_counters()
            self._log_pending_updates()
            self.metrics_writer.log_event(
                "training_interrupted" if kind == "interrupted" else "training_failed",
                {
                    "episode": self.state.completed_episodes + 1,
                    "error_type": type(original).__name__,
                    "error": str(original),
                },
                self.state.environment_steps,
            )
            self.metrics_writer.flush()
            self.checkpoint_manager.save_recovery(
                kind,
                self.agent,
                self.plan,
                self.state,
                self.tensorboard_log_dir,
            )
        except Exception as recovery_error:
            original.add_note(f"failed to save recovery state: {recovery_error}")

    def _log_pending_updates(self) -> None:
        for metrics in self.agent.drain_update_metrics():
            self.metrics_writer.log_ppo_update(metrics)

    def _log_action_errors(self, result: EpisodeResult) -> None:
        """Record what the API rejected, not just how many times.

        Episode metrics only carry a count, which is not enough to diagnose a
        rejection after the fact. Distinct messages are logged, capped so a
        pathological episode cannot flood the log.
        """
        rejected: dict[str, dict[str, object]] = {}
        for transition in result.transitions:
            if not transition.info.get("action_error"):
                continue
            message = str(transition.info.get("error"))
            rejected.setdefault(
                message,
                {
                    "count": 0,
                    "action": transition.action.to_dict(),
                    "state_type": transition.state.raw_state.get("state_type"),
                },
            )
            rejected[message]["count"] += 1  # type: ignore[operator]
        if not rejected:
            return
        self.metrics_writer.log_event(
            "action_errors",
            {
                "episode": self.state.completed_episodes,
                "rejections": [
                    {"error": message, **detail}
                    for message, detail in list(rejected.items())[:10]
                ],
            },
            self.state.environment_steps,
        )

    def _episode_reset_spec(self, episode_index: int | None = None) -> ResetSpec:
        """Return the reset for the next episode, seeded from the pool.

        Without a pool this is the configured reset unchanged, so an unseeded
        run behaves exactly as before.
        """
        if episode_index is None:
            episode_index = self.state.completed_episodes
        seed = self.plan.training.seed_for_episode(episode_index)
        if seed is None:
            return self.plan.reset
        return replace(self.plan.reset, run_seed=seed)

    def _adopt_agent_counters(self) -> None:
        """Copy the agent's counters, which are the authority while training."""
        self.state.environment_steps = self.agent.environment_steps
        self.state.optimizer_updates = self.agent.optimizer_updates

    def _episode_metrics(
        self,
        result: EpisodeResult,
        *,
        duration_seconds: float,
        seed: str | None,
    ) -> EpisodeMetrics:
        return EpisodeMetrics(
            episode=self.state.completed_episodes,
            environment_steps=self.state.environment_steps,
            reward=result.total_reward,
            steps=result.steps,
            floor=_floor(result.final_state),
            terminated=result.terminated,
            truncated=result.truncated,
            action_errors=sum(
                transition.info.get("action_error") is True
                for transition in result.transitions
            ),
            optimizer_updates=self.state.optimizer_updates,
            duration_seconds=duration_seconds,
            seed=seed,
            reused_run=result.reused_run,
        )


def _floor(raw_state: dict[str, object]) -> int | None:
    run = raw_state.get("run")
    if not isinstance(run, dict):
        return None
    value = run.get("floor")
    if isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
