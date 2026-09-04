"""Auditable JSONL metrics mirrored to TensorBoard scalars."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Protocol


class MetricsError(RuntimeError):
    """Raised when training metrics cannot be safely written or resumed."""


class ScalarWriter(Protocol):
    """Small SummaryWriter surface used by the training runtime and tests."""

    def add_scalar(self, tag: str, scalar_value: object, global_step: int) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class EpisodeMetrics:
    """One completed or truncated episode summary."""

    episode: int
    environment_steps: int
    reward: float
    steps: int
    floor: int | None
    terminated: bool
    truncated: bool
    action_errors: int
    optimizer_updates: int
    duration_seconds: float

    def __post_init__(self) -> None:
        for name in (
            "episode",
            "environment_steps",
            "steps",
            "action_errors",
            "optimizer_updates",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")


class TrainingMetricsWriter:
    """Append recoverable JSONL records and matching TensorBoard scalars."""

    EPISODE_TAGS: Mapping[str, str] = {
        "reward": "episode/reward",
        "steps": "episode/length",
        "floor": "episode/floor",
        "action_errors": "episode/action_errors",
        "terminated": "episode/terminated",
        "truncated": "episode/truncated",
        "duration_seconds": "episode/duration_seconds",
        "episode": "training/completed_episodes",
        "optimizer_updates": "training/optimizer_updates",
        "environment_steps": "training/environment_steps",
    }
    PPO_TAGS: Mapping[str, str] = {
        "loss": "ppo/loss",
        "policy_loss": "ppo/policy_loss",
        "value_loss": "ppo/value_loss",
        "entropy": "ppo/entropy",
        "gradient_norm": "ppo/gradient_norm",
        "rollout_steps": "ppo/rollout_steps",
    }

    def __init__(
        self,
        run_dir: str | Path,
        *,
        tensorboard_enabled: bool = True,
        tensorboard_log_dir: str = "tensorboard",
        tensorboard_flush_secs: int = 30,
        resume_step: int | None = None,
        writer_factory: Callable[..., ScalarWriter] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        tensorboard_relative = Path(tensorboard_log_dir)
        if (
            not tensorboard_log_dir
            or tensorboard_relative.is_absolute()
            or ".." in tensorboard_relative.parts
            or tensorboard_relative == Path(".")
        ):
            raise ValueError("tensorboard_log_dir must be a non-empty relative path")
        self.tensorboard_path = self.run_dir / tensorboard_relative
        self._closed = False
        if resume_step is not None:
            if resume_step < 0:
                raise ValueError("resume_step must be non-negative")
            self._truncate_jsonl(resume_step)
        self._jsonl = self.jsonl_path.open("a", encoding="utf-8")
        self._tensorboard: ScalarWriter | None = None
        if tensorboard_enabled:
            if writer_factory is None:
                from torch.utils.tensorboard import SummaryWriter

                writer_factory = SummaryWriter
            options: dict[str, object] = {
                "log_dir": str(self.tensorboard_path),
                "flush_secs": tensorboard_flush_secs,
            }
            if resume_step is not None:
                options["purge_step"] = resume_step + 1
            try:
                self._tensorboard = writer_factory(**options)
            except Exception as exc:
                self._jsonl.close()
                raise MetricsError(
                    f"failed to create TensorBoard writer: {exc}"
                ) from exc

    def log_episode(self, metrics: EpisodeMetrics) -> None:
        """Write one episode record and its TensorBoard scalar mirror."""
        payload = asdict(metrics)
        self._write_record("episode", metrics.environment_steps, payload)
        if self._tensorboard is not None:
            for field, tag in self.EPISODE_TAGS.items():
                value = payload[field]
                if value is not None:
                    self._tensorboard.add_scalar(
                        tag,
                        value,
                        metrics.environment_steps,
                    )

    def log_ppo_update(self, metrics: Mapping[str, float]) -> None:
        """Write one completed PPO optimizer update."""
        raw_step = metrics.get("environment_steps")
        if raw_step is None or float(raw_step) < 0 or not float(raw_step).is_integer():
            raise MetricsError("PPO metrics require an integer environment_steps")
        global_step = int(raw_step)
        payload = {key: float(value) for key, value in metrics.items()}
        self._write_record("ppo_update", global_step, payload)
        if self._tensorboard is not None:
            for field, tag in self.PPO_TAGS.items():
                if field in payload:
                    self._tensorboard.add_scalar(tag, payload[field], global_step)

    def log_event(
        self,
        event_type: str,
        payload: Mapping[str, object],
        global_step: int,
    ) -> None:
        """Append a non-scalar lifecycle event to JSONL."""
        if not event_type:
            raise ValueError("event_type must not be empty")
        self._write_record(event_type, global_step, dict(payload))

    def flush(self) -> None:
        """Synchronously flush JSONL and TensorBoard queues."""
        if self._closed:
            return
        self._jsonl.flush()
        os.fsync(self._jsonl.fileno())
        if self._tensorboard is not None:
            self._tensorboard.flush()

    def close(self) -> None:
        """Flush and close both outputs; repeated calls are harmless."""
        if self._closed:
            return
        try:
            self.flush()
        finally:
            try:
                if self._tensorboard is not None:
                    self._tensorboard.close()
            finally:
                self._jsonl.close()
                self._closed = True

    def __enter__(self) -> TrainingMetricsWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _write_record(
        self,
        record_type: str,
        global_step: int,
        payload: Mapping[str, object],
    ) -> None:
        if self._closed:
            raise MetricsError("metrics writer is closed")
        record = {**payload, "type": record_type, "global_step": global_step}
        try:
            self._jsonl.write(
                json.dumps(
                    record,
                    allow_nan=False,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            self._jsonl.flush()
        except (OSError, TypeError, ValueError) as exc:
            raise MetricsError(f"failed to write training metrics: {exc}") from exc

    def _truncate_jsonl(self, resume_step: int) -> None:
        if not self.jsonl_path.exists():
            return
        temporary_path = self.jsonl_path.with_suffix(".jsonl.tmp")
        try:
            with self.jsonl_path.open("r", encoding="utf-8") as source:
                records = []
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError(f"line {line_number} is not a JSON object")
                    step = record.get("global_step")
                    if isinstance(step, bool) or not isinstance(step, int):
                        raise ValueError(
                            f"line {line_number} has no integer global_step"
                        )
                    if step <= resume_step:
                        records.append(record)
            with temporary_path.open("w", encoding="utf-8") as target:
                for record in records:
                    target.write(
                        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary_path, self.jsonl_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            temporary_path.unlink(missing_ok=True)
            raise MetricsError(
                f"failed to resume metrics from {self.jsonl_path}: {exc}"
            ) from exc
