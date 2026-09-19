"""Parse and validate one recorded human run from ``gameplay_records``.

A record is external input, so everything is checked where it enters and the
typed objects below are trusted afterwards.  The recorder writes exactly what
an agent needs for one decision: the screen it acted on, the ``/player``
payload that carries the run-level master deck, and the action it sent.  Those
three are ``GameObservation.raw_state``, ``GameObservation.player_detail``, and
a ``GameAction``, so a record replays into the live pipeline unchanged.

Like ``clean``, this depends on the action layer and never on the encoder.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from sts2rl.actions import GameAction
from sts2rl.env.types import GameObservation, RawState


# Recorder schema the cleaner understands.  A record written by a newer
# recorder is refused rather than read optimistically: the fields this reads
# are load-bearing, and a silently renamed one would produce a dataset that
# trains on the wrong thing.
SUPPORTED_SCHEMA_VERSIONS = frozenset({2})


class RecordError(ValueError):
    """Raised when a gameplay record cannot be trusted as written."""


@dataclass(frozen=True)
class RecordedAction:
    """One action the human sent, as the recorder wrote it."""

    action_type: str
    args: Mapping[str, Any]
    label: str | None = None

    def to_game_action(self) -> GameAction:
        """Return the action in the form the rest of the codebase speaks."""
        return GameAction(self.action_type, **dict(self.args))

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.action_type, **dict(self.args)}


@dataclass(frozen=True)
class RecordedStep:
    """One recorded decision point: the screen, the deck, and the action.

    ``action`` is None on the final step, which is the state the run ended in
    and carries no decision.  ``resumed`` marks a step the recorder captured
    after the game was reloaded, so the step before it is *not* what produced
    this state -- the reload is.
    """

    index: int
    raw_state: RawState
    player_detail: RawState | None
    action: RecordedAction | None
    resumed: bool = False

    def observation(self) -> GameObservation:
        """Return the agent input this step represents."""
        return GameObservation(
            raw_state=self.raw_state, player_detail=self.player_detail
        )

    @property
    def state_type(self) -> str | None:
        value = self.raw_state.get("state_type")
        return value if isinstance(value, str) else None

    @property
    def act(self) -> int | None:
        return _optional_integer(_mapping(self.raw_state.get("run")).get("act"))

    @property
    def floor(self) -> int | None:
        return _optional_integer(_mapping(self.raw_state.get("run")).get("floor"))


@dataclass(frozen=True)
class RecordedRun:
    """One complete recorded run and its outcome."""

    path: Path
    run_id: str
    schema_version: int
    recorder_version: str | None
    game_version: str | None
    seed: str | None
    character: str | None
    game_mode: str | None
    ascension: int | None
    num_reloads: int | None
    victory: bool | None
    abandoned: bool | None
    floor_reached: int | None
    steps: tuple[RecordedStep, ...]

    @classmethod
    def load(cls, path: str | Path) -> RecordedRun:
        """Read, validate, and return one record.

        The files carry a UTF-8 BOM, so ``utf-8-sig`` is not optional.
        """
        source = Path(path)
        try:
            with source.open("r", encoding="utf-8-sig") as handle:
                payload = json.load(handle)
        except OSError as exc:
            raise RecordError(f"cannot read gameplay record {source}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RecordError(f"gameplay record {source} is not valid JSON: {exc}") from exc
        return cls.from_payload(payload, path=source)

    @classmethod
    def from_payload(cls, payload: object, *, path: str | Path) -> RecordedRun:
        """Validate one already-parsed record."""
        source = Path(path)
        if not isinstance(payload, Mapping):
            raise RecordError(f"gameplay record {source} must be a JSON object")

        schema_version = payload.get("schema_version")
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise RecordError(
                f"gameplay record {source} has unsupported schema_version "
                f"{schema_version!r}; this cleaner reads "
                f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )

        run_id = payload.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise RecordError(
                f"gameplay record {source} must carry a non-empty string run_id"
            )

        run = _mapping(payload.get("run"))
        outcome = _mapping(payload.get("outcome"))
        steps = cls._parse_steps(payload.get("steps"), source)
        return cls(
            path=source,
            run_id=run_id,
            schema_version=int(schema_version),
            recorder_version=_optional_text(payload.get("recorder_version")),
            game_version=_optional_text(payload.get("game_version")),
            seed=_optional_text(run.get("seed")),
            character=_optional_text(run.get("character")),
            game_mode=_optional_text(run.get("game_mode")),
            ascension=_optional_integer(run.get("ascension")),
            num_reloads=_optional_integer(run.get("num_reloads")),
            victory=_optional_bool(outcome.get("victory")),
            abandoned=_optional_bool(outcome.get("abandoned")),
            floor_reached=_optional_integer(outcome.get("floor_reached")),
            steps=steps,
        )

    @staticmethod
    def _parse_steps(value: object, source: Path) -> tuple[RecordedStep, ...]:
        if not isinstance(value, list) or not value:
            raise RecordError(
                f"gameplay record {source} must carry a non-empty steps array"
            )

        steps: list[RecordedStep] = []
        for position, raw in enumerate(value):
            if not isinstance(raw, Mapping):
                raise RecordError(
                    f"{source}: step at position {position} must be an object"
                )
            index = raw.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise RecordError(
                    f"{source}: step at position {position} has a non-integer "
                    f"index {index!r}"
                )
            if steps and index <= steps[-1].index:
                raise RecordError(
                    f"{source}: step indices must increase; {index} follows "
                    f"{steps[-1].index}"
                )
            state = raw.get("state")
            if not isinstance(state, Mapping) or not state:
                raise RecordError(
                    f"{source}: step {index} must carry a non-empty state object"
                )
            player_detail = raw.get("player_detail")
            if player_detail is not None and not isinstance(player_detail, Mapping):
                raise RecordError(
                    f"{source}: step {index} player_detail must be an object or null"
                )
            steps.append(
                RecordedStep(
                    index=index,
                    raw_state=dict(state),
                    player_detail=(
                        None if player_detail is None else dict(player_detail)
                    ),
                    action=RecordedRun._parse_action(raw.get("action"), index, source),
                    resumed=raw.get("resumed") is True,
                )
            )
        return tuple(steps)

    @staticmethod
    def _parse_action(
        value: object, index: int, source: Path
    ) -> RecordedAction | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise RecordError(
                f"{source}: step {index} action must be an object or null"
            )
        action_type = value.get("action")
        if not isinstance(action_type, str) or not action_type:
            raise RecordError(
                f"{source}: step {index} action needs a non-empty 'action' name"
            )
        args = value.get("args")
        if args is None:
            args = {}
        if not isinstance(args, Mapping):
            raise RecordError(
                f"{source}: step {index} action args must be an object or null"
            )
        if any(not isinstance(key, str) for key in args):
            raise RecordError(
                f"{source}: step {index} action args must be keyed by strings"
            )
        return RecordedAction(
            action_type=action_type,
            args=dict(args),
            label=_optional_text(value.get("label")),
        )


def find_records(source: str | Path) -> tuple[Path, ...]:
    """Return the record files under ``source``, or ``source`` itself.

    Sorted by name so a dataset built twice from the same directory lists its
    runs in the same order.
    """
    path = Path(source)
    if path.is_file():
        return (path,)
    if not path.is_dir():
        raise RecordError(f"no gameplay records at {path}")
    found = tuple(sorted(item for item in path.glob("*.json") if item.is_file()))
    if not found:
        raise RecordError(f"no *.json gameplay records in {path}")
    return found


def load_records(paths: Iterable[str | Path]) -> tuple[RecordedRun, ...]:
    """Load several records, refusing a run id that appears twice."""
    runs: list[RecordedRun] = []
    seen: dict[str, Path] = {}
    for path in paths:
        run = RecordedRun.load(path)
        if run.run_id in seen:
            raise RecordError(
                f"run_id {run.run_id!r} appears in both {seen[run.run_id]} and "
                f"{run.path}; the same run must not be cleaned twice"
            )
        seen[run.run_id] = run.path
        runs.append(run)
    return tuple(runs)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)
