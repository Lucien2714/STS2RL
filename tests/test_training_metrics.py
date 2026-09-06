"""JSONL and TensorBoard metric writer tests."""

from __future__ import annotations

import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import pytest

from sts2rl.training import EpisodeMetrics, TrainingMetricsWriter
from sts2rl.training import metrics as metrics_module
from sts2rl.training.metrics import MetricsError


class FakeSummaryWriter:
    def __init__(self, **options: object) -> None:
        self.options = options
        self.scalars: list[tuple[str, object, int]] = []
        self.flush_count = 0
        self.close_count = 0

    def add_scalar(self, tag: str, value: object, step: int) -> None:
        self.scalars.append((tag, value, step))

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.close_count += 1


def _episode(step: int = 12) -> EpisodeMetrics:
    return EpisodeMetrics(
        episode=2,
        environment_steps=step,
        reward=7.5,
        steps=5,
        floor=3,
        terminated=True,
        truncated=False,
        action_errors=1,
        optimizer_updates=4,
        duration_seconds=0.25,
    )


def test_metrics_mirror_jsonl_to_writer_and_close_once(tmp_path: Path):
    fake = FakeSummaryWriter()
    writer = TrainingMetricsWriter(
        tmp_path,
        writer_factory=lambda **options: _capture_options(fake, options),
        tensorboard_flush_secs=17,
    )

    writer.log_episode(_episode())
    writer.log_ppo_update(
        {
            "environment_steps": 12.0,
            "optimizer_update": 4.0,
            "loss": 1.25,
            "policy_loss": -0.1,
            "value_loss": 2.5,
            "entropy": 0.8,
            "gradient_norm": 0.4,
            "rollout_steps": 6.0,
        }
    )
    writer.close()
    writer.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["type"] for record in records] == ["episode", "ppo_update"]
    assert all(record["global_step"] == 12 for record in records)
    assert ("episode/reward", 7.5, 12) in fake.scalars
    assert ("ppo/loss", 1.25, 12) in fake.scalars
    assert fake.options["flush_secs"] == 17
    assert fake.close_count == 1
    assert fake.flush_count >= 1


def test_resume_truncates_future_jsonl_and_sets_tensorboard_purge_step(
    tmp_path: Path,
):
    with TrainingMetricsWriter(tmp_path, tensorboard_enabled=False) as writer:
        writer.log_episode(_episode(10))
        writer.log_event("future", {}, 20)
    fake = FakeSummaryWriter()

    with TrainingMetricsWriter(
        tmp_path,
        resume_step=10,
        writer_factory=lambda **options: _capture_options(fake, options),
    ):
        pass

    records = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["global_step"] for record in records] == [10]
    assert fake.options["purge_step"] == 11


def test_real_tensorboard_event_contains_episode_and_ppo_scalars(tmp_path: Path):
    with TrainingMetricsWriter(tmp_path) as writer:
        writer.log_episode(_episode())
        writer.log_ppo_update(
            {
                "environment_steps": 12.0,
                "optimizer_update": 4.0,
                "loss": 1.5,
                "rollout_steps": 8.0,
            }
        )

    accumulator = EventAccumulator(str(tmp_path / "tensorboard"))
    accumulator.Reload()

    assert "episode/reward" in accumulator.Tags()["scalars"]
    assert "ppo/loss" in accumulator.Tags()["scalars"]
    reward_event = accumulator.Scalars("episode/reward")[0]
    assert reward_event.step == 12
    assert reward_event.value == 7.5


def test_disabling_tensorboard_only_writes_jsonl(tmp_path: Path):
    with TrainingMetricsWriter(tmp_path, tensorboard_enabled=False) as writer:
        writer.log_episode(_episode())

    assert (tmp_path / "metrics.jsonl").is_file()
    assert not (tmp_path / "tensorboard").exists()


def _capture_options(
    writer: FakeSummaryWriter,
    options: dict[str, object],
) -> FakeSummaryWriter:
    writer.options = options
    return writer


def test_a_locked_metrics_file_says_what_to_do(tmp_path, monkeypatch):
    """Windows reports only "access is denied" when another process holds it."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(
        '{"type": "episode", "global_step": 1}\n', encoding="utf-8"
    )
    monkeypatch.setattr(
        metrics_module.os,
        "replace",
        lambda *_: (_ for _ in ()).throw(PermissionError("access is denied")),
    )

    with pytest.raises(MetricsError, match="holding that file open"):
        TrainingMetricsWriter(run_dir, tensorboard_enabled=False, resume_step=1)
