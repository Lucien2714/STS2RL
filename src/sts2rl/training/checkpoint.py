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
class LoadedCheckpoint:
    """Validated checkpoint payload awaiting restoration into an agent.

    ``TrainingState`` is the single record of every counter.  Metrics and
    TensorBoard resume from ``training_state.environment_steps`` rather than
    from a parallel copy that then has to be proven equal to it.
    """

    path: Path
    plan: TrainingPlan
    training_state: TrainingState
    tensorboard_log_dir: str
    agent_state: Mapping[str, object]
    torch_rng_state: torch.Tensor
    cuda_rng_states: tuple[torch.Tensor, ...]


class CheckpointManager:
    """Save and load checkpoints relative to one experiment directory."""

    FORMAT_VERSION = 2

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

    def save_progress(
        self,
        agent: CandidatePPOAgent,
        plan: TrainingPlan,
        training_state: TrainingState,
        tensorboard_log_dir: str,
    ) -> Path:
        """Save a checkpoint named by how much training it contains.

        Optimizer updates are the unit, not episodes: episode length here grows
        from about 35 steps to about 140 as the policy improves, so "every 50
        episodes" quietly means four times as much training late in a run as
        early.  Updates are a fixed 256 transitions each, so the number in the
        filename is comparable between checkpoints and between runs.
        """
        return self.save(
            f"update_{training_state.optimizer_updates:06d}.pt",
            agent,
            plan,
            training_state,
            tensorboard_log_dir,
        )

    def save_recovery(
        self,
        kind: str,
        agent: CandidatePPOAgent,
        plan: TrainingPlan,
        training_state: TrainingState,
        tensorboard_log_dir: str,
    ) -> Path:
        """Save an interrupted or failed run after its partial rollout is dropped."""
        if kind not in {"interrupted", "recovery"}:
            raise ValueError("recovery checkpoint kind must be interrupted or recovery")
        return self.save(
            f"{kind}_step_{training_state.environment_steps:09d}.pt",
            agent,
            plan,
            training_state,
            tensorboard_log_dir,
        )

    def save(
        self,
        filename: str,
        agent: CandidatePPOAgent,
        plan: TrainingPlan,
        training_state: TrainingState,
        tensorboard_log_dir: str,
    ) -> Path:
        """Atomically publish a complete checkpoint and update ``latest``."""
        if Path(filename).name != filename or not filename.endswith(".pt"):
            raise ValueError("checkpoint filename must be a plain .pt filename")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.checkpoint_dir / filename
        temporary_path = final_path.with_suffix(final_path.suffix + ".tmp")
        payload = {
            "format_version": self.FORMAT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "agent_state": agent.checkpoint_state(),
            "training_state": training_state.to_dict(),
            "tensorboard_log_dir": tensorboard_log_dir,
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
            plan = TrainingPlan.from_dict(
                self._require_mapping(payload, "training_plan")
            )
        except (TypeError, ValueError) as exc:
            raise CheckpointError(f"invalid checkpoint metadata: {exc}") from exc
        tensorboard_log_dir = payload.get("tensorboard_log_dir")
        if not isinstance(tensorboard_log_dir, str) or not tensorboard_log_dir:
            raise CheckpointError("checkpoint tensorboard_log_dir must be a string")
        torch_rng_state = payload.get("torch_rng_state")
        if not isinstance(torch_rng_state, torch.Tensor):
            raise CheckpointError("checkpoint torch_rng_state must be a tensor")
        raw_cuda_states = payload.get("cuda_rng_states", ())
        if not isinstance(raw_cuda_states, (list, tuple)) or not all(
            isinstance(item, torch.Tensor) for item in raw_cuda_states
        ):
            raise CheckpointError("checkpoint cuda_rng_states must be tensors")

        return LoadedCheckpoint(
            path=path,
            plan=plan,
            training_state=training_state,
            tensorboard_log_dir=tensorboard_log_dir,
            agent_state=dict(agent_state),
            torch_rng_state=torch_rng_state.cpu(),
            cuda_rng_states=tuple(item.cpu() for item in raw_cuda_states),
        )

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
        agent.environment_steps = loaded.training_state.environment_steps
        agent.optimizer_updates = loaded.training_state.optimizer_updates
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
