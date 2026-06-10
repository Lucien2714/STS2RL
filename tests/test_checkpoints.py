"""Tests for checkpoint ordering."""

from pathlib import Path

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
