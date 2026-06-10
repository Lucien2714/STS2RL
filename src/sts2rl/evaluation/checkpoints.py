"""Checkpoint discovery and sorting helpers."""

import re
from pathlib import Path


def checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    """Sort step checkpoints chronologically before the latest checkpoint."""
    match = re.search(r"_step_(\d+)", path.stem)
    if match:
        return (0, int(match.group(1)), path.name)
    if path.stem.endswith("_latest"):
        return (1, 0, path.name)
    return (2, 0, path.name)


def find_checkpoints(checkpoint_dir: Path) -> list[Path]:
    """Return checkpoint files sorted in evaluation order."""
    return sorted(checkpoint_dir.glob("*.pt"), key=checkpoint_sort_key)
