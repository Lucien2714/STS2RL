"""Multi-episode orchestration around the single-episode runner."""

from __future__ import annotations

from collections.abc import Callable
from time import monotonic

from sts2rl.agents import CandidatePPOAgent, EpisodeResult, EpisodeRunner
from sts2rl.training.checkpoint import CheckpointManager
from sts2rl.training.config import TrainingPlan, TrainingState
from sts2rl.training.metrics import EpisodeMetrics, TrainingMetricsWriter


class Trainer:
    """Run complete episodes, publish metrics, and save resumable boundaries."""

    def __init__(
        self,
        runner: EpisodeRunner,
        agent: CandidatePPOAgent,
        checkpoint_manager: CheckpointManager,
        metrics_writer: TrainingMetricsWriter,
        plan: TrainingPlan,
        state: TrainingState | None = None,
        reporter: Callable[[EpisodeMetrics], None] | None = None,
        tensorboard_log_dir: str = "tensorboard",
    ) -> None:
        self.runner = runner
        self.agent = agent
        self.checkpoint_manager = checkpoint_manager
        self.metrics_writer = metrics_writer
        self.plan = plan
        self.state = state or TrainingState()
        self.reporter = reporter
        self.tensorboard_log_dir = tensorboard_log_dir

    def train(self) -> TrainingState:
        """Train until the configured cumulative episode target is reached."""
        target = self.plan.training.total_episodes
        if target < self.state.completed_episodes:
            raise ValueError(
                "total_episodes is below the restored completed episode count"
            )
        if target == self.state.completed_episodes:
            return self.state

        last_saved_episode: int | None = None
        try:
            while self.state.completed_episodes < target:
                started_at = monotonic()
                result = self.runner.run(self.plan.reset)
                self._adopt_agent_counters()
                self.state.completed_episodes += 1
                self._log_pending_updates()
                episode_metrics = self._episode_metrics(
                    result,
                    duration_seconds=monotonic() - started_at,
                )
                self.metrics_writer.log_episode(episode_metrics)
                self._log_action_errors(result)
                if self.reporter is not None:
                    self.reporter(episode_metrics)

                if (
                    self.state.completed_episodes % self.plan.training.checkpoint_every
                    == 0
                ):
                    self._save_episode_checkpoint()
                    last_saved_episode = self.state.completed_episodes

            if last_saved_episode != self.state.completed_episodes:
                self._save_episode_checkpoint()
            return self.state
        except KeyboardInterrupt as exc:
            self._recover("interrupted", exc)
            raise
        except Exception as exc:
            self._recover("recovery", exc)
            raise

    def _save_episode_checkpoint(self) -> None:
        """Flush the rollout, then save; a checkpoint needs a clean boundary.

        The agent accumulates across episodes, so this is the one place that
        forces a possibly-short update — every ``checkpoint_every`` episodes
        rather than every episode.
        """
        self.agent.update()
        self._adopt_agent_counters()
        self._log_pending_updates()
        self.metrics_writer.flush()
        self.checkpoint_manager.save_episode(
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

    def _adopt_agent_counters(self) -> None:
        """Copy the agent's counters, which are the authority while training."""
        self.state.environment_steps = self.agent.environment_steps
        self.state.optimizer_updates = self.agent.optimizer_updates

    def _episode_metrics(
        self,
        result: EpisodeResult,
        *,
        duration_seconds: float,
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
