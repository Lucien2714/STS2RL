"""Training CLI parsing and resume compatibility tests."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from sts2rl.env import ResetSpec
from sts2rl.training import TrainingConfig, TrainingPlan
from sts2rl.training import cli


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

    assert len(plan.training.training_seeds) == 12
    assert len(plan.training.holdout_seeds) == 3
    assert not set(plan.training.training_seeds) & set(plan.training.holdout_seeds)


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
