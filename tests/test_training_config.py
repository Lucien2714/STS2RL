"""Training configuration and serialization tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from sts2rl.agents import PPOConfig
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


def test_one_client_per_port_keeps_the_configured_url():
    assert TrainingConfig().client_base_urls() == (
        "http://localhost:15526/api/v1",
    )


def test_ports_replace_only_the_port_component():
    config = TrainingConfig(
        base_url="http://127.0.0.1:15526/api/v1", ports=(15527, 15528)
    )

    assert config.client_base_urls() == (
        "http://127.0.0.1:15527/api/v1",
        "http://127.0.0.1:15528/api/v1",
    )


def test_a_repeated_port_is_rejected():
    """Two clients on one port would be one game played by two lanes."""
    with pytest.raises(ValueError, match="must not repeat"):
        TrainingConfig(ports=(15526, 15526))


def test_an_out_of_range_port_is_rejected():
    with pytest.raises(ValueError, match="between 1 and 65535"):
        TrainingConfig(ports=(0,))


def test_ports_survive_a_serialization_round_trip():
    plan = TrainingPlan(training=TrainingConfig(ports=(15526, 15527)))

    restored = TrainingPlan.from_dict(json.loads(json.dumps(plan.to_dict())))

    assert restored.training.ports == (15526, 15527)


def test_searching_fights_needs_the_simulator(tmp_path: Path):
    with pytest.raises(ValueError, match="backend sim"):
        TrainingConfig(run_dir=tmp_path, search_combat=True)
    config = TrainingConfig(run_dir=tmp_path, backend="sim", search_combat=True, search_rooms=["elite", "boss"])
    assert config.search_rooms == ("elite", "boss")
    with pytest.raises(ValueError, match="search_rooms"):
        TrainingConfig(run_dir=tmp_path, backend="sim", search_combat=True, search_rooms=("shop",))


def test_search_settings_survive_a_round_trip(tmp_path: Path):
    config = TrainingConfig(
        run_dir=tmp_path, backend="sim", search_combat=True, search_simulations=200,
        search_turn_depth=3, search_weights="w.json",
    )
    restored = TrainingConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config


def test_elite_and_boss_budgets_default_to_the_search_budget(tmp_path: Path):
    config = TrainingConfig(run_dir=tmp_path, backend="sim", search_combat=True, search_boss_simulations=200)
    assert config.search_room_simulations() == {"boss": 200}
    assert TrainingConfig(run_dir=tmp_path).search_room_simulations() == {}
    with pytest.raises(ValueError, match="search_elite_simulations"):
        TrainingConfig(run_dir=tmp_path, search_elite_simulations=0)
    values = config.to_dict()
    assert TrainingConfig.from_dict(json.loads(json.dumps(values))) == config
    del values["search_elite_simulations"], values["search_boss_simulations"]
    assert TrainingConfig.from_dict(values).search_room_simulations() == {}  # older plans


def test_a_plan_saved_before_search_existed_loads_without_it(tmp_path: Path):
    values = TrainingConfig(run_dir=tmp_path).to_dict()
    for name in ("search_combat", "search_rooms", "search_simulations", "search_turn_depth", "search_weights"):
        values.pop(name)
    assert TrainingConfig.from_dict(values).search_combat is False


def test_a_plan_saved_before_fights_could_leave_the_rollout_keeps_them_in(tmp_path: Path):
    values = TrainingConfig(run_dir=tmp_path).to_dict()
    values.pop("search_fights_in_rollout")
    assert TrainingConfig.from_dict(values).search_fights_in_rollout is True


def test_exploration_rates_survive_a_serialization_round_trip():
    plan = TrainingPlan(ppo=PPOConfig(exploration={"rest_site": 0.3, "map": 0.15}))

    restored = TrainingPlan.from_dict(json.loads(json.dumps(plan.to_dict())))

    assert restored == plan
    assert restored.ppo.exploration == (("map", 0.15), ("rest_site", 0.3))


def test_a_plan_saved_before_exploration_existed_loads_without_it():
    values = TrainingPlan().to_dict()
    del values["ppo"]["exploration"]

    assert TrainingPlan.from_dict(values).ppo.exploration == ()
