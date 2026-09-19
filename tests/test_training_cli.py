"""Training CLI parsing and resume compatibility tests."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sts2rl.env import ResetSpec
from sts2rl.training import TrainingConfig, TrainingPlan
from sts2rl.training.config import DEFAULT_HOLDOUT_SEEDS, DEFAULT_SEED_POOL
from sts2rl.training import cli
from sts2rl.training import eval_cli


def test_help_exits_without_starting_training(monkeypatch: pytest.MonkeyPatch):
    called = False

    def fail_if_called(args: argparse.Namespace) -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(cli, "run_training", fail_if_called)

    with pytest.raises(SystemExit) as raised:
        cli.main(["--help"])

    assert raised.value.code == 0
    assert not called


def test_new_plan_maps_cli_options_and_can_disable_tensorboard(tmp_path: Path):
    args = cli.create_parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--total-episodes",
            "12",
            "--checkpoint-every",
            "3",
            "--hidden-dim",
            "32",
            "--entity-heads",
            "4",
            "--gamma",
            "0.9",
            "--no-tensorboard",
        ]
    )

    plan = cli._new_plan(args)

    assert plan.training.total_episodes == 12
    assert plan.training.checkpoint_every == 3
    assert not plan.training.tensorboard_enabled
    assert plan.encoder.hidden_dim == 32
    assert plan.ppo.gamma == 0.9


def test_resume_uses_checkpoint_plan_and_allows_runtime_overrides(tmp_path: Path):
    saved = TrainingPlan(
        training=TrainingConfig(
            total_episodes=10,
            checkpoint_every=2,
            run_dir=tmp_path / "run",
        )
    )
    args = cli.create_parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--resume",
            "latest",
            "--total-episodes",
            "25",
            "--checkpoint-every",
            "5",
            "--timeout",
            "40",
        ]
    )

    resumed = cli._resumed_plan(args, SimpleNamespace(plan=saved))  # type: ignore[arg-type]

    assert resumed.training.total_episodes == 25
    assert resumed.training.checkpoint_every == 5
    assert resumed.training.timeout == 40
    assert resumed.encoder == saved.encoder
    assert resumed.ppo == saved.ppo


def test_resume_rejects_model_defining_override(tmp_path: Path):
    saved = TrainingPlan(training=TrainingConfig(run_dir=tmp_path / "run"))
    args = cli.create_parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--resume",
            "latest",
            "--gamma",
            "0.5",
        ]
    )

    with pytest.raises(ValueError, match="--gamma cannot change"):
        cli._resumed_plan(args, SimpleNamespace(plan=saved))  # type: ignore[arg-type]


def test_main_returns_training_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(cli, "run_training", lambda args: 7)

    result = cli.main(["--run-dir", str(tmp_path / "run")])

    assert result == 7


def test_main_maps_keyboard_interrupt_to_exit_130(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    def interrupt(args: argparse.Namespace) -> int:
        del args
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_training", interrupt)

    assert cli.main(["--run-dir", str(tmp_path / "run")]) == 130


def test_action_delay_is_configurable_on_a_new_plan(tmp_path: Path):
    args = cli.create_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--action-delay", "0.25"]
    )

    plan = cli._new_plan(args)

    assert plan.training.action_delay_seconds == 0.25


def test_action_delay_can_be_disabled_on_resume(tmp_path: Path):
    """Zero is a real setting, so it must survive _or_default."""
    saved = TrainingPlan(training=TrainingConfig(run_dir=tmp_path / "run"))
    args = cli.create_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--resume", "latest", "--action-delay", "0"]
    )

    resumed = cli._resumed_plan(args, SimpleNamespace(plan=saved))  # type: ignore[arg-type]

    assert resumed.training.action_delay_seconds == 0.0


def test_seed_pool_and_modifiers_come_from_the_command_line(tmp_path: Path):
    args = cli.create_parser().parse_args(
        [
            "--run-dir", str(tmp_path / "run"),
            "--game-mode", "custom",
            "--seed-pool", "AAA,BBB",
            "--holdout-seeds", "ZZZ",
            "--modifiers", "MIDAS",
        ]
    )

    plan = cli._new_plan(args)

    assert plan.training.training_seeds == ("AAA", "BBB")
    assert plan.training.holdout_seeds == ("ZZZ",)
    assert plan.reset.modifiers == ("MIDAS",)


def test_default_installs_the_bundled_pools(tmp_path: Path):
    args = cli.create_parser().parse_args(
        [
            "--run-dir", str(tmp_path / "run"),
            "--game-mode", "custom",
            "--seed-pool", "default",
            "--holdout-seeds", "default",
        ]
    )

    plan = cli._new_plan(args)

    assert plan.training.training_seeds == DEFAULT_SEED_POOL
    assert plan.training.holdout_seeds == DEFAULT_HOLDOUT_SEEDS
    assert not set(plan.training.training_seeds) & set(plan.training.holdout_seeds)
    # Size is the dial between memorizing a map and spending the variance
    # budget on draw luck, so a pool that silently shrank back to a dozen is
    # worth failing over.
    assert len(plan.training.training_seeds) >= 100
    assert len(plan.training.holdout_seeds) >= 20


def test_resume_refuses_to_reshuffle_the_seed_pool(tmp_path: Path):
    """Changing the pool would remap every seed onto a different episode."""
    saved = TrainingPlan(
        training=TrainingConfig(
            run_dir=tmp_path / "run", training_seeds=("AAA", "BBB")
        ),
        reset=ResetSpec(game_mode="custom"),
    )
    args = cli.create_parser().parse_args(
        [
            "--run-dir", str(tmp_path / "run"),
            "--resume", "latest",
            "--seed-pool", "AAA,CCC",
        ]
    )

    with pytest.raises(ValueError, match="--seed-pool cannot change"):
        cli._resumed_plan(args, SimpleNamespace(plan=saved))  # type: ignore[arg-type]


def test_resume_accepts_the_pool_it_was_saved_with(tmp_path: Path):
    saved = TrainingPlan(
        training=TrainingConfig(
            run_dir=tmp_path / "run", training_seeds=("AAA", "BBB")
        ),
        reset=ResetSpec(game_mode="custom"),
    )
    args = cli.create_parser().parse_args(
        [
            "--run-dir", str(tmp_path / "run"),
            "--resume", "latest",
            "--seed-pool", "AAA,BBB",
        ]
    )

    resumed = cli._resumed_plan(args, SimpleNamespace(plan=saved))  # type: ignore[arg-type]

    assert resumed.training.training_seeds == ("AAA", "BBB")


def test_ports_come_from_the_command_line(tmp_path: Path):
    args = cli.create_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--ports", "15526, 15527"]
    )

    plan = cli._new_plan(args)

    assert plan.training.ports == (15526, 15527)


def test_resume_refuses_to_change_the_client_set(tmp_path: Path):
    saved = TrainingPlan(
        training=TrainingConfig(run_dir=tmp_path / "run", ports=(15526, 15527))
    )
    args = cli.create_parser().parse_args(
        [
            "--run-dir", str(tmp_path / "run"),
            "--resume", "latest",
            "--ports", "15526",
        ]
    )

    with pytest.raises(ValueError, match="--ports cannot change"):
        cli._resumed_plan(args, SimpleNamespace(plan=saved))  # type: ignore[arg-type]


def test_the_evaluation_cli_defaults_to_both_pools(tmp_path: Path):
    args = eval_cli.create_parser().parse_args(["--run-dir", str(tmp_path / "run")])

    assert args.pools == "both"
    assert args.checkpoint == "latest"


def test_the_evaluation_cli_takes_ports_and_a_checkpoint(tmp_path: Path):
    args = eval_cli.create_parser().parse_args(
        [
            "--run-dir", str(tmp_path / "run"),
            "--checkpoint", "episode_000500.pt",
            "--ports", "15526,15527",
            "--episodes-per-seed", "5",
        ]
    )

    assert args.checkpoint == "episode_000500.pt"
    assert args.episodes_per_seed == 5
    assert args.ports == "15526,15527"


def test_the_evaluation_cli_builds_one_client_per_port(tmp_path: Path):
    args = eval_cli.create_parser().parse_args(
        ["--run-dir", str(tmp_path / "run"), "--ports", "15526,15527,15528"]
    )
    plan = TrainingPlan(training=TrainingConfig(run_dir=tmp_path / "run"))

    urls = eval_cli._client_base_urls(args, plan)

    assert urls == (
        "http://localhost:15526/api/v1",
        "http://localhost:15527/api/v1",
        "http://localhost:15528/api/v1",
    )


def test_init_encoder_and_resume_are_mutually_exclusive(tmp_path: Path):
    """One starts from cloned weights, the other continues its own."""
    args = cli.create_parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--resume",
            "--init-encoder",
            str(tmp_path / "bc_best.pt"),
        ]
    )

    with pytest.raises(ValueError, match="Pass one or the other"):
        cli.run_training(args)


def test_init_encoder_defaults_to_absent(tmp_path: Path):
    args = cli.create_parser().parse_args(["--run-dir", str(tmp_path / "run")])

    assert args.init_encoder is None


def test_init_encoder_loads_only_the_encoder_weights(tmp_path: Path):
    """The artifact carries no optimizer state and no counters to restore."""
    from sts2rl.encoder import EncoderConfig, GameEncoder, GameVocabulary
    from sts2rl.training.bc import BCConfig, EpochMetrics, save_bc_checkpoint

    vocabulary = GameVocabulary.from_bundled_data()
    cloned = GameEncoder(vocabulary, EncoderConfig())
    with torch.no_grad():
        for parameter in cloned.parameters():
            parameter.add_(0.25)
    artifact = save_bc_checkpoint(
        tmp_path / "bc_best.pt",
        encoder=cloned,
        vocabulary_fingerprint=vocabulary.fingerprint(),
        encoder_config=EncoderConfig(),
        bc_config=BCConfig(),
        metrics=EpochMetrics(
            epoch=1,
            train_loss=0.5,
            train_accuracy=0.9,
            holdout_loss=0.6,
            holdout_accuracy=0.4,
            holdout_chance=0.2,
            holdout_first_candidate=0.3,
        ),
    )

    fresh = GameEncoder(vocabulary, EncoderConfig())
    fresh.load_state_dict(
        cli.load_bc_encoder_state(
            artifact, vocabulary=vocabulary, encoder_config=EncoderConfig()
        )
    )

    for name, tensor in cloned.state_dict().items():
        assert torch.allclose(fresh.state_dict()[name], tensor)


def test_init_encoder_is_applied_before_any_client_is_contacted(tmp_path: Path):
    """The call site is wired, and it runs before the network.

    Loading weights after a client is contacted would leave a half-built run
    behind when the artifact turns out to be incompatible.
    """
    from sts2rl.encoder import EncoderConfig

    seen: dict[str, object] = {}

    class Stop(RuntimeError):
        pass

    def capture(path, *, vocabulary, encoder_config):
        seen["path"] = path
        seen["fingerprint"] = vocabulary.fingerprint()
        seen["encoder_config"] = encoder_config
        raise Stop

    args = cli.create_parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--init-encoder",
            str(tmp_path / "bc_best.pt"),
            "--hidden-dim",
            "32",
            "--entity-heads",
            "2",
        ]
    )
    original = cli.load_bc_encoder_state
    cli.load_bc_encoder_state = capture
    try:
        with pytest.raises(Stop):
            cli.run_training(args)
    finally:
        cli.load_bc_encoder_state = original

    assert seen["path"] == tmp_path / "bc_best.pt"
    assert seen["encoder_config"] == EncoderConfig(hidden_dim=32, entity_heads=2)


def test_a_run_started_from_cloned_weights_says_so_in_its_config(tmp_path: Path):
    """Two runs identical in every other field are different experiments."""
    args = cli.create_parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--init-encoder",
            str(tmp_path / "bc_best.pt"),
        ]
    )

    plan = cli._new_plan(args)

    assert plan.training.init_encoder == str(tmp_path / "bc_best.pt")
    assert TrainingPlan.from_dict(plan.to_dict()).training.init_encoder == str(
        tmp_path / "bc_best.pt"
    )


def test_a_run_from_random_weights_records_none(tmp_path: Path):
    args = cli.create_parser().parse_args(["--run-dir", str(tmp_path / "run")])

    assert cli._new_plan(args).training.init_encoder is None


def test_a_config_saved_before_the_field_existed_still_loads(tmp_path: Path):
    """Every run directory written so far lacks init_encoder."""
    values = TrainingConfig(run_dir=tmp_path).to_dict()
    del values["init_encoder"]

    assert TrainingConfig.from_dict(values).init_encoder is None


def test_an_empty_init_encoder_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="init_encoder"):
        TrainingConfig(run_dir=tmp_path, init_encoder="")
