"""Typed evaluation-run configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration values for evaluating saved battle checkpoints."""

    episodes: int = 3
    checkpoint_dir: Path = Path("checkpoints")
    game_mode: str = "custom"
    timeout: float = 20.0
    max_steps: int = 0
    sleep_seconds: float = 0.3
    reset_environment: bool = True
    auto_pause: bool = False
