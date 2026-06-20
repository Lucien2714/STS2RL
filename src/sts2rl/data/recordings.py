"""Reader for human-play recordings produced by the STS2MCP mod.

The mod writes one JSON object per line (JSONL). Each line pairs the pre-action
decision state with the action the human took:

    {"ts": 1.23, "state_type": "monster", "state": {...}, "action": {"type": "play_card", ...}}

This module is deliberately algorithm-independent: it yields the raw
``(state, action)`` dicts exactly as recorded so any agent's own encoders and
candidate logic can consume them. Encoding and loss live in the trainer, not here.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


RecordingSample = tuple[dict, dict]


def resolve_recording_files(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    """Expand files, directories, and globs into a sorted list of JSONL files.

    A directory expands to its ``*.jsonl`` files (recursively). A path containing
    glob characters is expanded relative to the current directory. A plain file
    path is used as-is.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    files: list[Path] = []
    for entry in paths:
        path = Path(entry)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.jsonl")))
        elif any(char in str(entry) for char in "*?["):
            files.extend(sorted(Path().glob(str(entry))))
        elif path.is_file():
            files.append(path)
        else:
            logger.warning("Recording path does not exist, skipping: %s", path)

    # De-duplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def iter_recordings(
    paths: str | Path | Iterable[str | Path],
    *,
    action_types: Iterable[str] | None = None,
) -> Iterator[RecordingSample]:
    """Yield raw ``(state, action)`` pairs from one or more JSONL recordings.

    ``action_types`` optionally restricts the stream to actions whose ``type`` is
    in the given set (e.g. the battle agent's ``ACTION_TYPES``). Malformed lines
    and records missing a state or action are skipped with a warning.
    """
    allowed = set(action_types) if action_types is not None else None
    files = resolve_recording_files(paths)
    if not files:
        logger.warning("No recording files found for %s", paths)
        return

    for file in files:
        with file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed line %s:%d (%s)", file, line_number, exc)
                    continue

                state = record.get("state")
                action = record.get("action")
                if not isinstance(state, dict) or not isinstance(action, dict):
                    logger.warning(
                        "Skipping record without state/action at %s:%d", file, line_number
                    )
                    continue
                if allowed is not None and action.get("type") not in allowed:
                    continue
                yield state, action


def load_recordings(
    paths: str | Path | Iterable[str | Path],
    *,
    action_types: Iterable[str] | None = None,
) -> list[RecordingSample]:
    """Eagerly load all ``(state, action)`` pairs into a list."""
    return list(iter_recordings(paths, action_types=action_types))
