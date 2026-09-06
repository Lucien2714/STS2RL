"""Serializable configuration and counters for repeatable training runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from sts2rl.agents import PPOConfig
from sts2rl.encoder import EncoderConfig
from sts2rl.env import DEFAULT_ACTION_DELAY_SECONDS, ResetSpec


# Fixed run seeds, cycled one per episode.
#
# A fresh random run every episode puts map layout, card rewards, shops, and
# the enemy sequence into the return, where the critic can only ever predict
# their average -- everything seed-specific lands in the advantage as noise,
# and "mean return over the last N episodes" mixes policy improvement with
# draw luck.  A fixed pool makes that number comparable across checkpoints.
#
# A pool, not one seed: a single seed is memorized as an action sequence.  The
# holdout seeds are never trained on, so evaluating on them is what separates
# "learned to climb" from "learned these twelve maps".
DEFAULT_SEED_POOL = (
    "7NKRVDBV", "TJWVA3B8", "GJ677ZKE", "DDY7BHHQ",
    "7PFC7NZR", "CSJ92XBT", "SQFNH36F", "ZU9GBB22",
    "7AU6U593", "PST6BSQ9", "KFGBG4ZP", "6T76XVK2",
)
DEFAULT_HOLDOUT_SEEDS = ("QXVE762C", "YDZERTWD", "ZTTJDGJF")


@dataclass(frozen=True)
class TrainingConfig:
    """Operational settings that do not define model tensor shapes."""

    total_episodes: int = 100
    checkpoint_every: int = 10
    max_steps_per_episode: int = 10_000
    max_state_refreshes: int = 3
    base_url: str = "http://localhost:15526/api/v1"
    timeout: float = 20.0
    action_delay_seconds: float = DEFAULT_ACTION_DELAY_SECONDS
    training_seeds: tuple[str, ...] = ()
    holdout_seeds: tuple[str, ...] = ()
    device: str = "cpu"
    torch_seed: int = 0
    run_dir: Path = Path("runs/default")
    tensorboard_enabled: bool = True
    tensorboard_flush_secs: int = 30

    def __post_init__(self) -> None:
        for name in (
            "total_episodes",
            "checkpoint_every",
            "max_steps_per_episode",
            "tensorboard_flush_secs",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.max_state_refreshes, bool)
            or not isinstance(self.max_state_refreshes, int)
            or self.max_state_refreshes < 0
        ):
            raise ValueError("max_state_refreshes must be a non-negative integer")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)):
            raise TypeError("timeout must be a number")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if isinstance(self.action_delay_seconds, bool) or not isinstance(
            self.action_delay_seconds, (int, float)
        ):
            raise TypeError("action_delay_seconds must be a number")
        if self.action_delay_seconds < 0:
            raise ValueError("action_delay_seconds must not be negative")
        if isinstance(self.torch_seed, bool) or not isinstance(self.torch_seed, int):
            raise ValueError("torch_seed must be an integer")
        if not isinstance(self.base_url, str):
            raise TypeError("base_url must be a string")
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if not isinstance(self.device, str):
            raise TypeError("device must be a string")
        if not isinstance(self.tensorboard_enabled, bool):
            raise TypeError("tensorboard_enabled must be a boolean")
        try:
            torch.device(self.device)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"invalid torch device: {self.device!r}") from exc
        for name in ("training_seeds", "holdout_seeds"):
            seeds = tuple(getattr(self, name))
            object.__setattr__(self, name, seeds)
            if any(not isinstance(seed, str) or not seed for seed in seeds):
                raise ValueError(f"{name} must be non-empty strings")
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"{name} must not repeat a seed")
        overlap = set(self.training_seeds) & set(self.holdout_seeds)
        if overlap:
            raise ValueError(
                "holdout seeds must never be trained on; both pools list "
                f"{sorted(overlap)}"
            )
        object.__setattr__(self, "run_dir", Path(self.run_dir))

    def validate_runtime_device(self) -> torch.device:
        """Resolve the configured device and reject unavailable CUDA devices."""
        device = torch.device(self.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"CUDA device requested but CUDA is unavailable: {device}")
        return device

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        result = asdict(self)
        result["run_dir"] = str(self.run_dir)
        result["training_seeds"] = list(self.training_seeds)
        result["holdout_seeds"] = list(self.holdout_seeds)
        return result

    def seed_for_episode(self, completed_episodes: int) -> str | None:
        """Return the seed the next episode should run, cycling the pool.

        The cursor is derived from the episode counter rather than stored, so
        a resumed run continues the cycle instead of restarting it -- which
        would otherwise re-train the same few seeds and starve the rest.
        """
        if not self.training_seeds:
            return None
        return self.training_seeds[completed_episodes % len(self.training_seeds)]

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> TrainingConfig:
        """Restore configuration from its validated JSON representation."""
        data = dict(values)
        if "run_dir" in data:
            data["run_dir"] = Path(str(data["run_dir"]))
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class TrainingPlan:
    """Complete immutable configuration of one training experiment."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reset: ResetSpec = field(default_factory=ResetSpec)

    def __post_init__(self) -> None:
        if self.reset.character not in range(5):
            raise ValueError("reset character must be between 0 and 4")
        if self.reset.game_mode == "standard" and self.reset.run_seed is not None:
            raise ValueError("run_seed is only supported for custom or daily runs")
        if self.training.training_seeds and self.reset.game_mode != "custom":
            raise ValueError(
                "a seed pool needs the custom-run screen, which is the only one "
                "that accepts a seed; pass --game-mode custom"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the complete plan as JSON-compatible primitives."""
        return {
            "training": self.training.to_dict(),
            "encoder": asdict(self.encoder),
            "ppo": asdict(self.ppo),
            "reset": asdict(self.reset),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> TrainingPlan:
        """Restore a complete plan and rerun every component's validation."""
        training = _mapping(values.get("training"), "training")
        encoder = _mapping(values.get("encoder"), "encoder")
        ppo = _mapping(values.get("ppo"), "ppo")
        reset = _mapping(values.get("reset"), "reset")
        return cls(
            training=TrainingConfig.from_dict(training),
            encoder=EncoderConfig(**encoder),  # type: ignore[arg-type]
            ppo=PPOConfig(**ppo),  # type: ignore[arg-type]
            reset=ResetSpec(**reset),  # type: ignore[arg-type]
        )


@dataclass
class TrainingState:
    """Mutable counters persisted across process restarts."""

    completed_episodes: int = 0
    environment_steps: int = 0
    optimizer_updates: int = 0

    def __post_init__(self) -> None:
        for name in (
            "completed_episodes",
            "environment_steps",
            "optimizer_updates",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, int]:
        """Return counters as checkpoint-safe primitives."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> TrainingState:
        """Restore and validate persisted counters."""
        return cls(**dict(values))  # type: ignore[arg-type]


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value
