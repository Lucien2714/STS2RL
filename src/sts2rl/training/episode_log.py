"""Thread-safe JSONL writer for completed training episodes."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class EpisodeLogWriter:
    """Append one structured JSON record per completed training episode."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, episode_result: dict[str, Any]) -> None:
        """Append an episode result with run id and timestamp metadata."""
        row = {
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **episode_result,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as episode_log:
                episode_log.write(json.dumps(row, sort_keys=True) + "\n")
