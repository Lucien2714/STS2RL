"""Turn recorded human runs into decisions a candidate policy can train on.

The product is deliberately *pre-tokenization*: one JSON line per decision
holding the raw screen, the ``/player`` payload, the candidate set, and which
candidate the human picked.  Two reasons, one hard and one soft.

The hard one: ``TokenizedDecision`` cannot be pickled at all, because
``TokenizedState.entities`` is a mappingproxy.  There is no tensor cache to
write even if we wanted one.

The soft one: ``encoder/schema.py`` is the single home of every model-visible
column, and a dataset written before tokenization survives a column being
added there.  Tokenizing costs about 1.1 ms per decision against about 2.3 ms
for one encoder forward pass, so paying it per epoch is not the bottleneck.

Cleaning depends on the action layer and never on the encoder.  That is what
lets a dataset outlive a column added to ``encoder/schema.py``, and it is why
``--verify`` is a separate opt-in pass rather than part of cleaning.  Note that
this is a *source* boundary, not a runtime one: ``sts2rl.agents`` eagerly
imports the PPO agent from its ``__init__``, so importing anything under it
loads torch regardless.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
from typing import Any

from sts2rl.actions import GameAction
from sts2rl.agents.action_space import LegalActionProvider
from sts2rl.offline.record import RecordedRun, RecordedStep, find_records, load_records


# Bump when the shape of a decision line changes.  A reader that finds a
# version it does not know refuses the dataset instead of misreading it, the
# same bargain ``CheckpointManager.FORMAT_VERSION`` makes.
DATASET_FORMAT_VERSION = 1

DECISIONS_FILENAME = "decisions.jsonl.gz"
MANIFEST_FILENAME = "manifest.json"

# Why a recorded step did not become a trainable decision.  **The order is
# part of the meaning**, because a step is attributed to the first reason that
# fires and several of them overlap.
#
# ``no_observable_effect`` in particular has to precede ``unmatched_action``.
# The two describe the same twelve game rejections from opposite sides -- a
# potion reward claimed against a full belt is both "refused by the game" and
# "not in the candidate set, because the provider skips it" -- and attributing
# them to the action space would overstate what the action space costs.  With
# the refusals removed first, every remaining mismatch is an action type the
# provider withholds on purpose, and ``params_differ`` drops to zero.
REJECTION_REASONS = (
    "terminal_no_action",
    "no_observable_effect",
    "no_candidates",
    "unmatched_action",
    "forced_single_candidate",
)


@dataclass(frozen=True)
class Decision:
    """One trainable decision: a state, its candidates, and the human's pick.

    ``candidates`` is a snapshot, not the authority.  A reader re-derives the
    candidate set from ``raw_state`` with the current ``LegalActionProvider``
    and compares, so a change to the action space is caught as a stale dataset
    rather than silently shifting ``expert_index`` onto a different action.

    ``next_step_index`` is the successor a future critic would bootstrap from,
    and it is None both at the end of a run *and* where the next step is one
    the recorder marked ``resumed``: after a reload the following state is not
    what this action produced, and a reward read as a difference between those
    two states would invent progress that never happened.
    """

    run_id: str
    step_index: int
    state_type: str | None
    act: int | None
    floor: int | None
    raw_state: Mapping[str, Any]
    player_detail: Mapping[str, Any] | None
    candidates: tuple[Mapping[str, Any], ...]
    expert_index: int
    next_step_index: int | None
    steps_to_run_end: int
    is_run_final_decision: bool

    def to_json(self) -> dict[str, Any]:
        """Return the decision as one JSON-compatible object."""
        return {
            "run_id": self.run_id,
            "step_index": self.step_index,
            "state_type": self.state_type,
            "act": self.act,
            "floor": self.floor,
            "raw_state": dict(self.raw_state),
            "player_detail": (
                None if self.player_detail is None else dict(self.player_detail)
            ),
            "candidates": [dict(candidate) for candidate in self.candidates],
            "expert_index": self.expert_index,
            "next_step_index": self.next_step_index,
            "steps_to_run_end": self.steps_to_run_end,
            "is_run_final_decision": self.is_run_final_decision,
        }

    @classmethod
    def from_json(cls, value: object) -> Decision:
        """Validate and restore one decision line."""
        if not isinstance(value, Mapping):
            raise ValueError("a decision line must be a JSON object")
        raw_state = value.get("raw_state")
        if not isinstance(raw_state, Mapping) or not raw_state:
            raise ValueError("decision raw_state must be a non-empty object")
        player_detail = value.get("player_detail")
        if player_detail is not None and not isinstance(player_detail, Mapping):
            raise ValueError("decision player_detail must be an object or null")
        candidates = value.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("decision candidates must be a non-empty array")
        if any(not isinstance(item, Mapping) for item in candidates):
            raise ValueError("every decision candidate must be an object")
        expert_index = value.get("expert_index")
        if (
            isinstance(expert_index, bool)
            or not isinstance(expert_index, int)
            or not 0 <= expert_index < len(candidates)
        ):
            raise ValueError(
                f"decision expert_index {expert_index!r} is not a position in its "
                f"{len(candidates)} candidates"
            )
        run_id = value.get("run_id")
        step_index = value.get("step_index")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("decision run_id must be a non-empty string")
        if isinstance(step_index, bool) or not isinstance(step_index, int):
            raise ValueError("decision step_index must be an integer")
        return cls(
            run_id=run_id,
            step_index=step_index,
            state_type=_optional_text(value.get("state_type")),
            act=_optional_integer(value.get("act")),
            floor=_optional_integer(value.get("floor")),
            raw_state=dict(raw_state),
            player_detail=None if player_detail is None else dict(player_detail),
            candidates=tuple(dict(item) for item in candidates),
            expert_index=expert_index,
            next_step_index=_optional_integer(value.get("next_step_index")),
            steps_to_run_end=_optional_integer(value.get("steps_to_run_end")) or 0,
            is_run_final_decision=value.get("is_run_final_decision") is True,
        )

    @property
    def expert_action(self) -> Mapping[str, Any]:
        """Return the candidate the human picked."""
        return self.candidates[self.expert_index]


@dataclass
class RunReport:
    """What one record contributed, and what it lost and why."""

    run_id: str
    path: str
    seed: str | None
    character: str | None
    game_mode: str | None
    victory: bool | None
    abandoned: bool | None
    floor_reached: int | None
    num_reloads: int | None
    steps: int
    kept: int
    rejected: dict[str, int] = field(
        default_factory=lambda: {reason: 0 for reason in REJECTION_REASONS}
    )
    unmatched_examples: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "path": self.path,
            "seed": self.seed,
            "character": self.character,
            "game_mode": self.game_mode,
            "victory": self.victory,
            "abandoned": self.abandoned,
            "floor_reached": self.floor_reached,
            "num_reloads": self.num_reloads,
            "steps": self.steps,
            "kept": self.kept,
            "rejected": dict(self.rejected),
            "unmatched_examples": list(self.unmatched_examples),
        }


@dataclass(frozen=True)
class ExcludedRun:
    """A record that was not cleaned at all, and why.

    Excluding happens per *run*, before its steps are read, so an excluded run
    contributes nothing to the ledger.  It is listed separately rather than
    dropped silently: "563 decisions" means something different depending on
    whether it came from two runs or from twenty with eighteen filtered out.
    """

    run_id: str
    path: str
    victory: bool | None
    floor_reached: int | None
    reason: str

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "path": self.path,
            "victory": self.victory,
            "floor_reached": self.floor_reached,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CleanResult:
    """Everything one cleaning pass produced."""

    decisions_path: Path
    manifest_path: Path
    decisions: int
    runs: tuple[RunReport, ...]
    excluded: tuple[ExcludedRun, ...] = ()

    @property
    def steps(self) -> int:
        return sum(report.steps for report in self.runs)

    @property
    def rejected(self) -> dict[str, int]:
        """Return the pooled rejection ledger over every run."""
        totals = {reason: 0 for reason in REJECTION_REASONS}
        for report in self.runs:
            for reason, count in report.rejected.items():
                totals[reason] += count
        return totals


def clean_run(
    run: RecordedRun,
    provider: LegalActionProvider | None = None,
    *,
    max_unmatched_examples: int = 20,
) -> tuple[tuple[Decision, ...], RunReport]:
    """Return the trainable decisions in one record, and the ledger for the rest.

    Every step lands in exactly one bucket, so ``kept`` plus the ledger always
    equals the number of recorded steps.  A cleaning pass that cannot say where
    a step went is not a cleaning pass.
    """
    action_provider = provider or LegalActionProvider()
    report = RunReport(
        run_id=run.run_id,
        path=str(run.path),
        seed=run.seed,
        character=run.character,
        game_mode=run.game_mode,
        victory=run.victory,
        abandoned=run.abandoned,
        floor_reached=run.floor_reached,
        num_reloads=run.num_reloads,
        steps=len(run.steps),
        kept=0,
    )

    decisions: list[Decision] = []
    for position, step in enumerate(run.steps):
        successor = run.steps[position + 1] if position + 1 < len(run.steps) else None
        outcome = _classify(step, successor, action_provider)
        if isinstance(outcome, str):
            report.rejected[outcome] += 1
            if outcome == "unmatched_action":
                _record_unmatched_example(
                    report, step, action_provider, max_unmatched_examples
                )
            continue

        candidates, expert_index = outcome
        decisions.append(
            Decision(
                run_id=run.run_id,
                step_index=step.index,
                state_type=step.state_type,
                act=step.act,
                floor=step.floor,
                raw_state=step.raw_state,
                player_detail=step.player_detail,
                candidates=tuple(action.to_dict() for action in candidates),
                expert_index=expert_index,
                # A reload means the next state is not what this action
                # produced, so it is not a successor a reward can be read off.
                next_step_index=(
                    None
                    if successor is None or successor.resumed
                    else successor.index
                ),
                steps_to_run_end=len(run.steps) - 1 - position,
                is_run_final_decision=False,
            )
        )

    if decisions:
        decisions[-1] = replace(decisions[-1], is_run_final_decision=True)
    report.kept = len(decisions)
    return tuple(decisions), report


def excluded_reason(
    run: RecordedRun,
    *,
    only_victories: bool,
    min_floor: int | None,
) -> str | None:
    """Return why this run should not be cloned at all, or None to keep it.

    Behavior cloning copies the demonstrator, so a run is a quality decision
    before it is a quantity one: the early floors of a run that died on floor 6
    are ordinary play, but everything after the mistake that killed it is a
    demonstration of losing.  Filtering is per run because that is the unit the
    outcome is known for.
    """
    if only_victories and run.victory is not True:
        return "not a victory"
    if min_floor is not None:
        if run.floor_reached is None:
            return f"floor_reached unknown, below --min-floor {min_floor}"
        if run.floor_reached < min_floor:
            return f"floor_reached {run.floor_reached} below --min-floor {min_floor}"
    return None


def clean_records(
    source: str | Path,
    out_dir: str | Path,
    *,
    provider: LegalActionProvider | None = None,
    max_unmatched_examples: int = 20,
    only_victories: bool = False,
    min_floor: int | None = None,
) -> CleanResult:
    """Clean every record under ``source`` into one dataset under ``out_dir``."""
    if isinstance(max_unmatched_examples, bool) or not isinstance(
        max_unmatched_examples, int
    ):
        raise TypeError("max_unmatched_examples must be an integer")
    if max_unmatched_examples < 0:
        raise ValueError("max_unmatched_examples must not be negative")
    if min_floor is not None and (
        isinstance(min_floor, bool) or not isinstance(min_floor, int) or min_floor < 1
    ):
        raise ValueError("min_floor must be a positive integer")

    paths = find_records(source)
    runs = load_records(paths)
    action_provider = provider or LegalActionProvider()

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    decisions_path = destination / DECISIONS_FILENAME
    manifest_path = destination / MANIFEST_FILENAME

    reports: list[RunReport] = []
    excluded: list[ExcludedRun] = []
    total = 0
    temporary = decisions_path.with_suffix(decisions_path.suffix + ".tmp")
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as handle:
            for run in runs:
                reason = excluded_reason(
                    run, only_victories=only_victories, min_floor=min_floor
                )
                if reason is not None:
                    excluded.append(
                        ExcludedRun(
                            run_id=run.run_id,
                            path=str(run.path),
                            victory=run.victory,
                            floor_reached=run.floor_reached,
                            reason=reason,
                        )
                    )
                    continue
                decisions, report = clean_run(
                    run,
                    action_provider,
                    max_unmatched_examples=max_unmatched_examples,
                )
                for decision in decisions:
                    handle.write(json.dumps(decision.to_json(), sort_keys=True))
                    handle.write("\n")
                total += len(decisions)
                reports.append(report)
        os.replace(temporary, decisions_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    result = CleanResult(
        decisions_path=decisions_path,
        manifest_path=manifest_path,
        decisions=total,
        runs=tuple(reports),
        excluded=tuple(excluded),
    )
    _write_json(manifest_path, _manifest(result, source))
    return result


def _classify(
    step: RecordedStep,
    successor: RecordedStep | None,
    provider: LegalActionProvider,
) -> str | tuple[tuple[GameAction, ...], int]:
    """Return the rejection reason, or the candidates and the human's index."""
    if step.action is None:
        return "terminal_no_action"

    # Refused and unchanged: the game rejected the action and left the screen
    # exactly as it was.  ``EpisodeRunner`` treats that same signature as a
    # stall rather than a transition, and for the same reason -- the action was
    # legal, the screen was simply not ready, and learning from it would teach
    # that resting at a rest site does nothing.
    if successor is not None and successor.raw_state == step.raw_state:
        return "no_observable_effect"

    candidates = provider.candidates(step.raw_state)
    if not candidates:
        return "no_candidates"

    expert = _action_key(step.action.to_game_action())
    keys = [_action_key(candidate) for candidate in candidates]
    if expert not in keys:
        return "unmatched_action"

    # One candidate carries no gradient: the cross entropy of a single-choice
    # distribution is exactly zero.  ``CandidatePPOAgent.choose_action`` drops
    # these from the rollout for the same reason.
    if len(candidates) == 1:
        return "forced_single_candidate"

    return candidates, keys.index(expert)


def _record_unmatched_example(
    report: RunReport,
    step: RecordedStep,
    provider: LegalActionProvider,
    limit: int,
) -> None:
    """Keep a few mismatches by name, so the action space's cost stays visible.

    A pooled count says "17 demonstrated actions were unreachable"; it does not
    say that seven of them were ``proceed`` on a reward screen, which is a
    deliberate exclusion, versus something nobody has accounted for.
    """
    if len(report.unmatched_examples) >= limit or step.action is None:
        return
    candidates = provider.candidates(step.raw_state)
    offered = sorted({candidate.action_type for candidate in candidates})
    report.unmatched_examples.append(
        {
            "step_index": step.index,
            "state_type": step.state_type,
            "action": step.action.to_dict(),
            "reason": (
                "type_absent"
                if step.action.action_type not in offered
                else "params_differ"
            ),
            "offered_types": offered,
        }
    )


def _action_key(action: GameAction) -> str:
    """Return a stable identity for one complete parameterized action."""
    return repr(sorted(action.to_dict().items()))


def _manifest(result: CleanResult, source: str | Path) -> dict[str, Any]:
    return {
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "decisions": result.decisions,
        "steps": result.steps,
        "rejected": result.rejected,
        "runs": [report.to_json() for report in result.runs],
        "excluded_runs": [run.to_json() for run in result.excluded],
    }


def _write_json(path: Path, value: object) -> None:
    """Publish a JSON file atomically, as the checkpoint manager does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_decisions(path: str | Path) -> Iterable[Decision]:
    """Yield the decisions in one ``decisions.jsonl.gz``, validating each line."""
    source = Path(path)
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield Decision.from_json(json.loads(line))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{source}: line {number}: {exc}") from exc


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)
