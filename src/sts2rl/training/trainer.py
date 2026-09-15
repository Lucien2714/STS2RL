"""Multi-episode orchestration around the single-episode runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
import json
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


# How many raw states one run may write for diagnosis.  A state is tens of
# kilobytes and a livelock produces one per episode, so this is a budget rather
# than a switch: a handful of examples explains a failure, and thousands only
# fill the disk.
MAX_STATE_SNAPSHOTS = 50

# How many times one accepted-but-inert action must repeat within an episode
# before its state is worth keeping.  A single inert step is ordinary -- a
# screen that had nothing to change -- while a hundred is a policy stuck.
INERT_SNAPSHOT_THRESHOLD = 10


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
        max_state_snapshots: int = MAX_STATE_SNAPSHOTS,
        inert_snapshot_threshold: int = INERT_SNAPSHOT_THRESHOLD,
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
        if max_state_snapshots < 0 or inert_snapshot_threshold < 1:
            raise ValueError(
                "max_state_snapshots must not be negative and "
                "inert_snapshot_threshold must be at least 1"
            )
        self.max_state_snapshots = max_state_snapshots
        self.inert_snapshot_threshold = inert_snapshot_threshold
        self._snapshots_written = 0
        self._checkpointing = threading.Lock()
        self._last_saved_update = self.state.optimizer_updates
        self._last_saved_episode = self.state.completed_episodes

    def _checkpoint_after_update(self) -> None:
        """Save right where the rollout is empty, if enough training has passed.

        This runs from inside the agent, immediately after an update, which is
        the only moment the rollout is empty by construction.  Checkpointing at
        an episode boundary instead meant manufacturing that moment by force,
        updating on whatever had been collected since -- 13 transitions in one
        measured case -- and then saving the weights that update had moved.
        """
        with self._checkpointing:
            if (
                self.state.optimizer_updates - self._last_saved_update
                < self.plan.training.checkpoint_every
            ):
                return
            self._adopt_agent_counters()
            self._save_checkpoint()

    def train(self) -> TrainingState:
        """Train until the configured cumulative episode target is reached."""
        target = self.plan.training.total_episodes
        if target < self.state.completed_episodes:
            raise ValueError(
                "total_episodes is below the restored completed episode count"
            )
        if target == self.state.completed_episodes:
            return self.state

        self.agent.on_update = self._checkpoint_after_update
        try:
            if len(self.runners) > 1:
                return self._train_parallel(target)
            return self._train_sequential(target)
        finally:
            self.agent.on_update = None

    def _train_sequential(self, target: int) -> TrainingState:
        """Play one client, without threads."""

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
                self._log_inert_actions(result)
                self._log_truncation(result)
                if self.reporter is not None:
                    self.reporter(episode_metrics)

            # Episodes can finish without ever filling a rollout, and a run
            # that ends with nothing on disk has lost all of its work.
            if self.state.completed_episodes != self._last_saved_episode:
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
        bookkeeping = threading.Lock()
        in_flight = 0
        failure: list[BaseException] = []
        retired: dict[int, BaseException] = {}

        def worker(runner: EpisodeRunner, lane: int) -> None:
            nonlocal in_flight
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

                error: STS2ClientError | None = None
                try:
                    started_at = monotonic()
                    result = runner.run(reset_spec)
                except STS2ClientError as exc:
                    # The game is external and does crash.  Its client dying is
                    # a reason to start another run, not to end a job that has
                    # cost hours.  Anything else is our own bug and stays fatal.
                    error = exc

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
                    self._log_inert_actions(result)
                    self._log_truncation(result)
                    if self.reporter is not None:
                        self.reporter(episode_metrics)

        def guarded(runner: EpisodeRunner, lane: int) -> None:
            try:
                worker(runner, lane)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
                with bookkeeping:
                    failure.append(exc)

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

        if self.state.completed_episodes != self._last_saved_episode:
            self._save_checkpoint()
        return self.state

    def _save_checkpoint(self) -> None:
        """Save a snapshot without disturbing the rollout.

        Nothing is flushed and nothing is discarded: whatever has been
        collected since the last update stays in the rollout and is learned
        from normally.  That matters most for the transitions a checkpoint used
        to land on -- the end of an episode, where the deaths are, and where
        the 29 boss wins in 1549 episodes are.
        """
        self._adopt_agent_counters()
        self._log_pending_updates()
        self.metrics_writer.flush()
        self.checkpoint_manager.save_progress(
            self.agent,
            self.plan,
            self.state,
            self.tensorboard_log_dir,
        )
        self._last_saved_update = self.state.optimizer_updates
        self._last_saved_episode = self.state.completed_episodes

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

    def _log_inert_actions(self, result: EpisodeResult) -> None:
        """Record actions the game accepted that changed nothing.

        A rejection is loud -- it carries a message and lands in
        ``action_errors``.  An accepted action that moves no part of the state
        is silent, and it is the more dangerous of the two: nothing fails, so
        the stall handling never sees it, and a deterministic policy repeats it
        until the step budget is gone.  Grouped by action type and screen,
        because the same inert action repeated ten thousand times is one fact.
        """
        inert: dict[tuple[str, str], dict[str, object]] = {}
        for transition in result.transitions:
            if not transition.info.get("inert"):
                continue
            state_type = str(transition.state.raw_state.get("state_type"))
            key = (transition.action.action_type, state_type)
            inert.setdefault(
                key,
                {
                    "count": 0,
                    "action": transition.action.to_dict(),
                    "state_type": state_type,
                },
            )
            inert[key]["count"] += 1  # type: ignore[operator]
        if not inert:
            return
        self.metrics_writer.log_event(
            "inert_actions",
            {
                "episode": self.state.completed_episodes,
                "actions": list(inert.values())[:10],
            },
            self.state.environment_steps,
        )
        worst = max(inert.values(), key=lambda entry: entry["count"])  # type: ignore[arg-type]
        if int(worst["count"]) >= self.inert_snapshot_threshold:
            self._save_state_snapshot(
                "inert",
                {
                    "episode": self.state.completed_episodes,
                    "action": worst["action"],
                    "repeated": worst["count"],
                    "state": self._last_state_for(result, worst["state_type"]),
                },
            )

    def _log_truncation(self, result: EpisodeResult) -> None:
        """Record why an episode stopped short, and what it was trying.

        ``truncated`` on its own says only that the run did not end in death,
        which covers a combat that never reached its play phase, a screen the
        policy could not leave, and a step budget spent -- three different
        problems that need three different fixes.
        """
        if not result.truncated:
            return
        self.metrics_writer.log_event(
            "truncation",
            {
                "episode": self.state.completed_episodes,
                "reason": result.truncation_reason,
                "state_type": result.final_state.get("state_type"),
                "steps": result.steps,
                "attempted_actions": list(result.attempted_actions),
            },
            self.state.environment_steps,
        )
        self._save_state_snapshot(
            "truncation",
            {
                "episode": self.state.completed_episodes,
                "reason": result.truncation_reason,
                "attempted_actions": list(result.attempted_actions),
                "state": result.final_state,
            },
        )

    def _last_state_for(
        self, result: EpisodeResult, state_type: object
    ) -> dict[str, object]:
        """Return the last raw state of the given screen, for a snapshot."""
        for transition in reversed(result.transitions):
            if transition.state.raw_state.get("state_type") == state_type:
                return transition.state.raw_state
        return result.final_state

    def _save_state_snapshot(self, kind: str, payload: Mapping[str, object]) -> None:
        """Write one raw state to disk, under a budget.

        ``state_type`` alone cannot reproduce a livelock: the shop that trapped
        one evaluation needed its item list and their ``can_afford`` flags to
        explain.  A whole state is tens of kilobytes, though, so this is capped
        per run -- diagnosis needs a handful of examples, not every one.
        """
        if self._snapshots_written >= self.max_state_snapshots:
            return
        directory = self.plan.training.run_dir / "diagnostics"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (
                f"{kind}_ep{self.state.completed_episodes:06d}"
                f"_{self._snapshots_written:03d}.json"
            )
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
        except OSError:
            # Diagnostics must never take the run down with them.
            return
        self._snapshots_written += 1

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
