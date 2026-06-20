"""Optional TensorBoard logging helpers for training and evaluation."""

from __future__ import annotations

import math
import re
import threading
import time
from pathlib import Path
from typing import Any


class TensorBoardLogger:
    """Small no-op-capable wrapper around torch's SummaryWriter."""

    def __init__(self, log_dir: Path | None, run_name: str) -> None:
        self.enabled = log_dir is not None
        self.log_dir = None if log_dir is None else Path(log_dir) / sanitize_run_name(run_name)
        self._writer = None
        self._lock = threading.Lock()
        if not self.enabled:
            return

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError(
                "TensorBoard logging requires the tensorboard package. "
                "Install project dependencies or run without --tensorboard-logdir."
            ) from exc

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))

    def add_scalar(self, tag: str, value: Any, step: int) -> None:
        """Write one scalar value when TensorBoard is enabled."""
        if self._writer is None:
            return
        scalar = safe_scalar(value)
        if scalar is None:
            return
        with self._lock:
            self._writer.add_scalar(tag, scalar, step)

    def add_scalars(self, prefix: str, values: dict[str, Any], step: int) -> None:
        """Write all finite scalar values under a common tag prefix."""
        for key, value in values.items():
            self.add_scalar(f"{prefix}/{key}", value, step)

    def flush(self) -> None:
        """Flush pending events."""
        if self._writer is None:
            return
        with self._lock:
            self._writer.flush()

    def close(self) -> None:
        """Close the underlying SummaryWriter."""
        if self._writer is None:
            return
        with self._lock:
            self._writer.close()
            self._writer = None


def safe_scalar(value: Any) -> float | None:
    """Return a finite float value or None when TensorBoard should skip it."""
    if value is None or isinstance(value, bool):
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(scalar):
        return None
    return scalar


def sanitize_run_name(name: str) -> str:
    """Return a filesystem-friendly TensorBoard run name."""
    normalized = re.sub(r"[^A-Za-z0-9_.=-]+", "_", name.strip())
    return normalized.strip("_") or time.strftime("%Y%m%d-%H%M%S")
