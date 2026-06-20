"""Tests for optional TensorBoard logging helpers."""

from sts2rl.training.tensorboard import TensorBoardLogger, safe_scalar, sanitize_run_name


def test_tensorboard_logger_noop_when_logdir_is_none(tmp_path):
    """A disabled logger should not import TensorBoard or create files."""
    logger = TensorBoardLogger(None, "test run")

    logger.add_scalar("train/loss", 1.0, 1)
    logger.add_scalars("episode", {"reward": 2.0}, 2)
    logger.flush()
    logger.close()

    assert logger.enabled is False
    assert list(tmp_path.iterdir()) == []


def test_safe_scalar_filters_non_numeric_values():
    """Only finite numeric values should be written as TensorBoard scalars."""
    assert safe_scalar(1) == 1.0
    assert safe_scalar("2.5") == 2.5
    assert safe_scalar(None) is None
    assert safe_scalar(True) is None
    assert safe_scalar(float("nan")) is None


def test_sanitize_run_name_keeps_tensorboard_paths_readable():
    """Run names should be stable and filesystem-friendly."""
    assert sanitize_run_name("training DQN: run/1") == "training_DQN_run_1"
