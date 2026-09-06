"""Training configuration and serialization tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from sts2rl.encoder import EncoderConfig
from sts2rl.env import ResetSpec
from sts2rl.training import TrainingConfig, TrainingPlan, TrainingState


def test_training_plan_json_round_trip_preserves_nested_types(tmp_path: Path):
    plan = TrainingPlan(
        training=TrainingConfig(
            total_episodes=12,
            checkpoint_every=3,
            run_dir=tmp_path / "run",
            tensorboard_enabled=False,
        ),
        encoder=EncoderConfig(hidden_dim=32, entity_heads=4),
        reset=ResetSpec(character=1, game_mode="custom", run_seed="ABC"),
    )

    restored = TrainingPlan.from_dict(plan.to_dict())

    assert restored == plan
    assert restored.training.run_dir == tmp_path / "run"


@pytest.mark.parametrize(
    "changes",
    [
        {"total_episodes": 0},
        {"checkpoint_every": 0},
        {"max_steps_per_episode": 0},
        {"max_state_refreshes": -1},
        {"timeout": 0.0},
        {"tensorboard_flush_secs": 0},
        {"device": "not a device"},
    ],
)
def test_invalid_training_config_is_rejected(changes: dict[str, object]):
    with pytest.raises(ValueError):
        TrainingConfig(**changes)  # type: ignore[arg-type]


def test_standard_run_rejects_unused_seed_and_character_range():
    with pytest.raises(ValueError, match="run_seed"):
        TrainingPlan(reset=ResetSpec(run_seed="IGNORED"))
    with pytest.raises(ValueError, match="character"):
        TrainingPlan(reset=ResetSpec(character=5))


def test_training_state_validates_non_negative_counters():
    state = TrainingState.from_dict(
        {
            "completed_episodes": 2,
            "environment_steps": 20,
            "optimizer_updates": 4,
        }
    )

    assert state.to_dict()["environment_steps"] == 20
    with pytest.raises(ValueError):
        TrainingState(environment_steps=-1)


def test_runtime_device_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA is unavailable"):
        TrainingConfig(device="cuda").validate_runtime_device()


def test_the_seed_pool_cycles_one_seed_per_episode():
    config = TrainingConfig(training_seeds=("A", "B", "C"))

    picked = [config.seed_for_episode(episode) for episode in range(7)]

    assert picked == ["A", "B", "C", "A", "B", "C", "A"]


def test_a_resumed_run_continues_the_cycle_rather_than_restarting_it():
    """The cursor is derived from the episode counter, so resume keeps place."""
    config = TrainingConfig(training_seeds=("A", "B", "C"))

    assert config.seed_for_episode(100) == config.training_seeds[100 % 3]


def test_no_pool_leaves_the_configured_reset_seed_alone():
    assert TrainingConfig().seed_for_episode(0) is None


def test_holdout_seeds_may_never_appear_in_the_training_pool():
    with pytest.raises(ValueError, match="never be trained on"):
        TrainingConfig(training_seeds=("A", "B"), holdout_seeds=("B", "C"))


def test_a_repeated_training_seed_is_rejected():
    with pytest.raises(ValueError, match="must not repeat"):
        TrainingConfig(training_seeds=("A", "A"))


def test_a_seed_pool_requires_the_custom_run_screen():
    with pytest.raises(ValueError, match="custom-run screen"):
        TrainingPlan(training=TrainingConfig(training_seeds=("A",)))


def test_seed_pools_survive_a_serialization_round_trip():
    plan = TrainingPlan(
        training=TrainingConfig(
            training_seeds=("A", "B"), holdout_seeds=("C",)
        ),
        reset=ResetSpec(game_mode="custom", modifiers=("MIDAS",)),
    )

    restored = TrainingPlan.from_dict(json.loads(json.dumps(plan.to_dict())))

    assert restored.training.training_seeds == ("A", "B")
    assert restored.training.holdout_seeds == ("C",)
    assert restored.reset.modifiers == ("MIDAS",)
