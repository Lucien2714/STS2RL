"""Atomic checkpoint and compatibility tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from sts2rl.agents import CandidatePPOAgent, PPOConfig
from sts2rl.encoder import EncoderConfig, GameEncoder, GameTokenizer, GameVocabulary
from sts2rl.training import (
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointManager,
    TrainingConfig,
    TrainingPlan,
    TrainingState,
)


def _plan(run_dir: Path) -> TrainingPlan:
    return TrainingPlan(
        training=TrainingConfig(
            total_episodes=4,
            checkpoint_every=2,
            run_dir=run_dir,
            tensorboard_enabled=False,
        ),
        encoder=EncoderConfig(
            hidden_dim=16,
            entity_heads=4,
            entity_ff_dim=32,
        ),
        ppo=PPOConfig(update_epochs=1),
    )


def _agent(vocabulary: GameVocabulary, plan: TrainingPlan) -> CandidatePPOAgent:
    return CandidatePPOAgent(
        GameTokenizer(vocabulary),
        GameEncoder(vocabulary, plan.encoder),
        config=plan.ppo,
    )


def _training_state() -> TrainingState:
    return TrainingState(
        completed_episodes=2,
        environment_steps=25,
        optimizer_updates=4,
    )


def test_checkpoint_round_trip_restores_agent_rng_and_latest(tmp_path: Path):
    run_dir = tmp_path / "run"
    vocabulary = GameVocabulary.from_bundled_data()
    plan = _plan(run_dir)
    manager = CheckpointManager(run_dir, vocabulary)
    manager.initialize_run(plan, resume=False)
    agent = _agent(vocabulary, plan)
    loss = sum(
        parameter.square().mean() for parameter in agent.game_encoder.parameters()
    )
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()
    expected_parameter = next(agent.game_encoder.parameters()).detach().clone()
    agent.environment_steps = 25
    agent.optimizer_updates = 4
    training = _training_state()
    torch.manual_seed(123)

    path = manager.save_episode(agent, plan, training, "tensorboard")
    expected_random = torch.rand(4)
    loaded = manager.load("latest")
    restored = _agent(vocabulary, plan)
    torch.manual_seed(999)
    manager.restore_agent(loaded, restored, plan)

    assert path.name == "episode_000002.pt"
    assert manager.resolve("latest") == path
    assert torch.equal(torch.rand(4), expected_random)
    assert restored.environment_steps == 25
    assert restored.optimizer_updates == 4
    assert restored.optimizer.state
    assert torch.equal(next(restored.game_encoder.parameters()), expected_parameter)
    assert loaded.training_state.to_dict() == training.to_dict()
    assert not list(manager.checkpoint_dir.glob("*.tmp"))
    latest = json.loads(manager.latest_path.read_text(encoding="utf-8"))
    assert latest == {"checkpoint": path.name}


def test_checkpoint_rejects_vocabulary_and_model_config_mismatch(tmp_path: Path):
    run_dir = tmp_path / "run"
    vocabulary = GameVocabulary.from_bundled_data()
    plan = _plan(run_dir)
    manager = CheckpointManager(run_dir, vocabulary)
    manager.initialize_run(plan, resume=False)
    agent = _agent(vocabulary, plan)
    agent.environment_steps = 25
    agent.optimizer_updates = 4
    training = _training_state()
    manager.save_episode(agent, plan, training, "tensorboard")

    other_data = tmp_path / "other-data"
    other_data.mkdir()
    (other_data / "cards.json").write_text(
        '[{"id": "ONLY_OTHER_CARD"}]',
        encoding="utf-8",
    )
    incompatible_manager = CheckpointManager(
        run_dir,
        GameVocabulary.from_bundled_data(other_data),
    )
    with pytest.raises(CheckpointCompatibilityError, match="vocabulary"):
        incompatible_manager.load("latest")

    loaded = manager.load("latest")
    incompatible_plan = replace(
        plan,
        encoder=EncoderConfig(hidden_dim=32, entity_heads=4),
    )
    with pytest.raises(CheckpointCompatibilityError, match="encoder"):
        manager.restore_agent(loaded, agent, incompatible_plan)


def test_counters_are_stored_once_and_restored_onto_the_agent(tmp_path: Path):
    run_dir = tmp_path / "run"
    vocabulary = GameVocabulary.from_bundled_data()
    plan = _plan(run_dir)
    manager = CheckpointManager(run_dir, vocabulary)
    manager.initialize_run(plan, resume=False)
    agent = _agent(vocabulary, plan)
    agent.environment_steps = 25
    agent.optimizer_updates = 4
    manager.save_episode(agent, plan, _training_state(), "tensorboard")

    loaded = manager.load("latest")
    restored = _agent(vocabulary, plan)
    manager.restore_agent(loaded, restored, plan)

    assert "environment_steps" not in loaded.agent_state
    assert "optimizer_updates" not in loaded.agent_state
    assert loaded.tensorboard_log_dir == "tensorboard"
    assert restored.environment_steps == 25
    assert restored.optimizer_updates == 4


def test_new_run_refuses_nonempty_directory_and_resume_requires_config(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "existing.txt").write_text("keep", encoding="utf-8")
    vocabulary = GameVocabulary.from_bundled_data()
    manager = CheckpointManager(run_dir, vocabulary)

    with pytest.raises(CheckpointError, match="not empty"):
        manager.initialize_run(_plan(run_dir), resume=False)
    with pytest.raises(CheckpointError, match="config.json"):
        manager.initialize_run(_plan(run_dir), resume=True)


def test_invalid_or_missing_latest_checkpoint_is_explicit(tmp_path: Path):
    manager = CheckpointManager(tmp_path / "run", GameVocabulary.from_bundled_data())

    with pytest.raises(CheckpointError, match="latest"):
        manager.resolve("latest")


def test_unsupported_checkpoint_version_is_rejected(tmp_path: Path):
    manager = CheckpointManager(tmp_path / "run", GameVocabulary.from_bundled_data())
    manager.checkpoint_dir.mkdir(parents=True)
    torch.save({"format_version": 999}, manager.checkpoint_dir / "future.pt")

    with pytest.raises(CheckpointCompatibilityError, match="format_version"):
        manager.load("future.pt")


def test_failed_save_keeps_previous_latest_and_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_dir = tmp_path / "run"
    vocabulary = GameVocabulary.from_bundled_data()
    plan = _plan(run_dir)
    manager = CheckpointManager(run_dir, vocabulary)
    manager.initialize_run(plan, resume=False)
    agent = _agent(vocabulary, plan)
    agent.environment_steps = 25
    agent.optimizer_updates = 4
    training = _training_state()
    original = manager.save_episode(agent, plan, training, "tensorboard")
    monkeypatch.setattr(torch, "save", lambda *args, **kwargs: _raise_disk_error())

    with pytest.raises(CheckpointError, match="failed to save"):
        manager.save("episode_000003.pt", agent, plan, training, "tensorboard")

    assert manager.resolve("latest") == original
    assert not list(manager.checkpoint_dir.glob("*.tmp"))


def _raise_disk_error() -> None:
    raise OSError("disk full")
