"""Reset configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResetSpec:
    """Describes how an environment reset should start a run."""

    game_mode: str = "standard"
    run_seed: str | None = None
    start_run_option: str = "confirm"

    @property
    def uses_seed(self) -> bool:
        """Return whether the reset should pass a seed to STS2MCP."""
        return self.game_mode in {"custom", "daily"} and self.run_seed is not None
