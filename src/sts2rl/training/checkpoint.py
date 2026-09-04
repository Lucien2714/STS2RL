"""Versioned, atomic checkpoints for resumable candidate-PPO training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import torch

from sts2rl.agents import CandidatePPOAgent
from sts2rl.encoder import GameVocabulary
from sts2rl.training.config import TrainingPlan, TrainingState


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be safely saved or restored."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when a valid checkpoint belongs to an incompatible model."""


@dataclass(frozen=True)
class LoggingState:
    """Position needed to continue JSONL and TensorBoard without rewinding."""

    tensorboard_log_dir: str = "tensorboard"
    tensorboard_global_step: int = 0
    last_logged_episode: int = 0
    last_logged_optimizer_update: int = 0

    def __post_init__(self) -> None:
        log_path = Path(self.tensorboard_log_dir)
        if (
            not self.tensorboard_log_dir
            or log_path.is_absolute()
            or ".." in log_path.parts
            or log_path == Path(".")
        ):
            raise ValueError("tensorboard_log_dir must be a non-empty relative path")
        for name in (
            "tensorboard_global_step",
            "last_logged_episode",
            "last_logged_optimizer_update",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "tensorboard_log_dir": self.tensorboard_log_dir,
            "tensorboard_global_step": self.tensorboard_global_step,
            "last_logged_episode": self.last_logged_episode,
            "last_logged_optimizer_update": self.last_logged_optimizer_update,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> LoggingState:
        return cls(**dict(values))  # type: ignore[arg-type]


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Validated checkpoint payload awaiting restoration into an agent."""

    path: Path
    plan: TrainingPlan
    training_state: TrainingState
    logging_state: LoggingState
    agent_state: Mapping[str, object]
    torch_rng_state: torch.Tensor
    cuda_rng_states: tuple[torch.Tensor, ...]


class CheckpointManager:
    """Save and load checkpoints relative to one experiment directory."""

    FORMAT_VERSION = 1

    def __init__(self, run_dir: str | Path, vocabulary: GameVocabulary) -> None:
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.latest_path = self.checkpoint_dir / "latest.json"
        self.vocabulary = vocabulary

    def initialize_run(self, plan: TrainingPlan, *, resume: bool) -> None:
        """Create a new run layout or validate that a resumed run exists."""
        config_path = self.run_dir / "config.json"
        if resume:
            if not self.run_dir.is_dir():
                raise CheckpointError(
                    f"resume run directory does not exist: {self.run_dir}"
                )
            if not config_path.is_file():
                raise CheckpointError(f"resume run has no config.json: {self.run_dir}")
            try:
                initial_plan = TrainingPlan.from_dict(self._read_json(config_path))
            except (OSError, TypeError, ValueError) as exc:
                raise CheckpointError(
                    f"invalid run config {config_path}: {exc}"
                ) from exc
            if (
                initial_plan.encoder != plan.encoder
                or initial_plan.ppo != plan.ppo
                or initial_plan.reset != plan.reset
                or initial_plan.training.torch_seed != plan.training.torch_seed
                or initial_plan.training.tensorboard_enabled
                != plan.training.tensorboard_enabled
            ):
                raise CheckpointCompatibilityError(
                    "checkpoint configuration does not belong to this run directory"
                )
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            return

        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise CheckpointError(
                f"run directory is not empty; pass --resume explicitly: {self.run_dir}"
            )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write_json(config_path, plan.to_dict())

    def resolve(self, value: str | Path) -> Path:
        """Resolve ``latest`` or an explicit checkpoint path."""
        if str(value).casefold() != "latest":
            path = Path(value)
            if not path.is_absolute():
                path = self.checkpoint_dir / path
            if not path.is_file():
                raise CheckpointError(f"checkpoint does not exist: {path}")
            return path

        try:
            latest = self._read_json(self.latest_path)
            filename = latest["checkpoint"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise CheckpointError(
                f"cannot resolve latest checkpoint from {self.latest_path}: {exc}"
            ) from exc
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise CheckpointError("latest checkpoint entry must be a plain filename")
        path = self.checkpoint_dir / filename
        if not path.is_file():
            raise CheckpointError(f"latest checkpoint does not exist: {path}")
        return path

    def save_episode(
        self,
        agent: CandidatePPOAgent,
        plan: TrainingPlan,
        training_state: TrainingState,
        logging_state: LoggingState,
    ) -> Path:
        """Save a numbered checkpoint after a completed episode."""
        return self.save(
            f"episode_{training_state.completed_episodes:06d}.pt",
            agent,
            plan,
            training_state,
            logging_state,
        )

    def save_recovery(
        self,
        kind: str,
        agent: CandidatePPOAgent,
        plan: TrainingPlan,
        training_state: TrainingState,
        logging_state: LoggingState,
    ) -> Path:
        """Save an interrupted or failed run after its partial rollout is dropped."""
        if kind not in {"interrupted", "recovery"}:
            raise ValueError("recovery checkpoint kind must be interrupted or recovery")
        return self.save(
            f"{kind}_step_{training_state.environment_steps:09d}.pt",
            agent,
            plan,
            training_state,
            logging_state,
        )

    def save(
        self,
        filename: str,
        agent: CandidatePPOAgent,
        plan: TrainingPlan,
        training_state: TrainingState,
        logging_state: LoggingState,
    ) -> Path:
        """Atomically publish a complete checkpoint and update ``latest``."""
        if Path(filename).name != filename or not filename.endswith(".pt"):
            raise ValueError("checkpoint filename must be a plain .pt filename")
        self._validate_counter_alignment(agent, training_state, logging_state)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.checkpoint_dir / filename
        temporary_path = final_path.with_suffix(final_path.suffix + ".tmp")
        payload = {
            "format_version": self.FORMAT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "agent_state": agent.checkpoint_state(),
            "training_state": training_state.to_dict(),
            "logging_state": logging_state.to_dict(),
            "training_plan": plan.to_dict(),
            "vocabulary_fingerprint": self.vocabulary.fingerprint(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                tuple(torch.cuda.get_rng_state_all())
                if torch.cuda.is_available()
                else ()
            ),
        }
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, final_path)
            self._atomic_write_json(
                self.latest_path,
                {"checkpoint": final_path.name},
            )
        except Exception as exc:
            temporary_path.unlink(missing_ok=True)
            raise CheckpointError(
                f"failed to save checkpoint {final_path}: {exc}"
            ) from exc
        return final_path

    def load(
        self,
        checkpoint: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> LoadedCheckpoint:
        """Read, validate, and return one trusted local checkpoint."""
        path = self.resolve(checkpoint)
        try:
            payload = torch.load(path, map_location=map_location, weights_only=True)
        except Exception as exc:
            raise CheckpointError(f"failed to load checkpoint {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise CheckpointError("checkpoint payload must be a mapping")
        if payload.get("format_version") != self.FORMAT_VERSION:
            raise CheckpointCompatibilityError(
                "unsupported checkpoint format_version: "
                f"{payload.get('format_version')!r}"
            )
        fingerprint = payload.get("vocabulary_fingerprint")
        if fingerprint != self.vocabulary.fingerprint():
            raise CheckpointCompatibilityError(
                "checkpoint vocabulary fingerprint does not match bundled data"
            )

        try:
            agent_state = self._require_mapping(payload, "agent_state")
            training_state = TrainingState.from_dict(
                self._require_mapping(payload, "training_state")
            )
            logging_state = LoggingState.from_dict(
                self._require_mapping(payload, "logging_state")
            )
            plan = TrainingPlan.from_dict(
                self._require_mapping(payload, "training_plan")
            )
        except (TypeError, ValueError) as exc:
            raise CheckpointError(f"invalid checkpoint metadata: {exc}") from exc
        torch_rng_state = payload.get("torch_rng_state")
        if not isinstance(torch_rng_state, torch.Tensor):
            raise CheckpointError("checkpoint torch_rng_state must be a tensor")
        raw_cuda_states = payload.get("cuda_rng_states", ())
        if not isinstance(raw_cuda_states, (list, tuple)) or not all(
            isinstance(item, torch.Tensor) for item in raw_cuda_states
        ):
            raise CheckpointError("checkpoint cuda_rng_states must be tensors")

        loaded = LoadedCheckpoint(
            path=path,
            plan=plan,
            training_state=training_state,
            logging_state=logging_state,
            agent_state=dict(agent_state),
            torch_rng_state=torch_rng_state.cpu(),
            cuda_rng_states=tuple(item.cpu() for item in raw_cuda_states),
        )
        self._validate_loaded_alignment(loaded)
        return loaded

    def restore_agent(
        self,
        loaded: LoadedCheckpoint,
        agent: CandidatePPOAgent,
        plan: TrainingPlan,
    ) -> None:
        """Validate model-defining configuration, then restore agent and RNG."""
        if loaded.plan.encoder != plan.encoder:
            raise CheckpointCompatibilityError("encoder configuration does not match")
        if loaded.plan.ppo != plan.ppo:
            raise CheckpointCompatibilityError("PPO configuration does not match")
        if loaded.plan.reset != plan.reset:
            raise CheckpointCompatibilityError("reset configuration does not match")
        agent.load_checkpoint_state(loaded.agent_state)
        torch.set_rng_state(loaded.torch_rng_state)
        if loaded.cuda_rng_states and agent.device.type == "cuda":
            torch.cuda.set_rng_state_all(list(loaded.cuda_rng_states))

    @staticmethod
    def _require_mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, object]:
        value = payload.get(name)
        if not isinstance(value, Mapping):
            raise CheckpointError(f"checkpoint {name} must be a mapping")
        return value

    @staticmethod
    def _validate_counter_alignment(
        agent: CandidatePPOAgent,
        training_state: TrainingState,
        logging_state: LoggingState,
    ) -> None:
        if agent.environment_steps != training_state.environment_steps:
            raise CheckpointError("agent and training environment_steps do not match")
        if agent.optimizer_updates != training_state.optimizer_updates:
            raise CheckpointError("agent and training optimizer_updates do not match")
        if logging_state.tensorboard_global_step != training_state.environment_steps:
            raise CheckpointError("logging global step does not match training state")
        if logging_state.last_logged_episode != training_state.completed_episodes:
            raise CheckpointError("logged episode does not match training state")
        if (
            logging_state.last_logged_optimizer_update
            != training_state.optimizer_updates
        ):
            raise CheckpointError(
                "logged optimizer update does not match training state"
            )

    def _validate_loaded_alignment(self, loaded: LoadedCheckpoint) -> None:
        agent_environment_steps = loaded.agent_state.get("environment_steps")
        agent_optimizer_updates = loaded.agent_state.get("optimizer_updates")
        if agent_environment_steps != loaded.training_state.environment_steps:
            raise CheckpointError("checkpoint environment step counters disagree")
        if agent_optimizer_updates != loaded.training_state.optimizer_updates:
            raise CheckpointError("checkpoint optimizer update counters disagree")
        if (
            loaded.logging_state.tensorboard_global_step
            != loaded.training_state.environment_steps
        ):
            raise CheckpointError("checkpoint logging step counter disagrees")
        if (
            loaded.logging_state.last_logged_episode
            != loaded.training_state.completed_episodes
        ):
            raise CheckpointError("checkpoint logged episode counter disagrees")
        if (
            loaded.logging_state.last_logged_optimizer_update
            != loaded.training_state.optimizer_updates
        ):
            raise CheckpointError("checkpoint logged update counter disagrees")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
        if not isinstance(value, dict):
            raise ValueError("JSON root must be an object")
        return value

    @staticmethod
    def _atomic_write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(value, file, indent=2, sort_keys=True)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
