"""Tests for checkpoint ordering."""

from pathlib import Path

from sts2rl.checkpoints.manager import battle_agent_checkpoint_dir, battle_backup_path, battle_latest_path
from sts2rl.evaluation.checkpoints import checkpoint_sort_key


def test_checkpoint_sort_key_orders_steps_before_latest():
    """Step checkpoints should sort before latest and unrelated files."""
    paths = [
        Path("battle_agent_latest.pt"),
        Path("battle_agent_step_10000.pt"),
        Path("battle_agent_step_5000.pt"),
        Path("other.pt"),
    ]

    assert [path.name for path in sorted(paths, key=checkpoint_sort_key)] == [
        "battle_agent_step_5000.pt",
        "battle_agent_step_10000.pt",
        "battle_agent_latest.pt",
        "other.pt",
    ]


def test_battle_agent_checkpoint_paths_are_partitioned_by_agent_type():
    """Battle agent checkpoints should live under agent-specific directories."""
    assert battle_agent_checkpoint_dir("DQN") == Path("checkpoints") / "battleAgent" / "DQN"
    assert battle_agent_checkpoint_dir("ppo") == Path("checkpoints") / "battleAgent" / "PPO"
    assert battle_latest_path("DQN") == (
        Path("checkpoints") / "battleAgent" / "DQN" / "battleagent_latest.pt"
    )
    assert battle_backup_path(20000, "PPO") == (
        Path("checkpoints") / "battleAgent" / "PPO" / "battleagent_step_20000.pt"
    )


def test_checkpoint_sort_key_orders_new_step_names_before_latest():
    """New battleagent_step names should sort chronologically."""
    paths = [
        Path("battleagent_latest.pt"),
        Path("battleagent_step_10000.pt"),
        Path("battleagent_step_5000.pt"),
    ]

    assert [path.name for path in sorted(paths, key=checkpoint_sort_key)] == [
        "battleagent_step_5000.pt",
        "battleagent_step_10000.pt",
        "battleagent_latest.pt",
    ]
