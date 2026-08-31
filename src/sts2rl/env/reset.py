"""Reset configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResetSpec:
    """Describes how an environment reset should start a run."""

    character: int = 0
    game_mode: str = "standard"
    run_seed: str | None = None
    start_run_option: str = "confirm"

    def __post_init__(self) -> None:
        if self.game_mode not in {"standard", "custom", "daily"}:
            raise ValueError(f"Unsupported game mode: {self.game_mode!r}")
        if self.start_run_option not in {"confirm", "embark"}:
            raise ValueError(
                f"Unsupported start-run option: {self.start_run_option!r}"
            )

    @property
    def uses_seed(self) -> bool:
        """Return whether the reset should pass a seed to STS2MCP."""
        return self.game_mode in {"custom", "daily"} and self.run_seed is not None
