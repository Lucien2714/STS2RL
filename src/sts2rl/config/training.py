"""Typed training-loop configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration values that control checkpointing and loop timing."""

    episode_log: Path = Path("logs") / "training_episodes.jsonl"
    save_interval: int = 10
    backup_interval: int = 20000
    step_sleep_seconds: float = 0.3
    episode_sleep_seconds: float = 0.5
    reconnect_poll_seconds: float = 2.0
