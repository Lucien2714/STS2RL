"""Typed environment configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    """Connection and run-start configuration for STS2MCP environments."""

    character: int = 0
    base_url: str = "http://localhost:15526/api/v1"
    timeout: float = 20.0
    game_mode: str = "standard"
    start_run_option: str = "confirm"
