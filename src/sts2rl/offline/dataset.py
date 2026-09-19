"""Read a cleaned dataset back and tokenize it for behavior cloning.

This is the one module in ``offline/`` that knows about the encoder.  It is
also where the dataset's validity is decided, and it is decided against real
data rather than against a fingerprint: the candidate set is re-derived from
every stored ``raw_state`` with the *current* ``LegalActionProvider`` and
compared to the snapshot the cleaner wrote.

That check is the whole reason the snapshot is stored.  ``expert_index`` is a
position in an ordered candidate list, so an action space that gained, lost, or
reordered a candidate would silently move the label onto a different action --
no exception, no metric, just a policy cloned onto the wrong choices.  A
re-derivation catches it exactly, and re-cleaning takes seconds.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import warnings

from sts2rl.actions import GameAction
from sts2rl.agents.action_space import LegalActionProvider
from sts2rl.encoder import GameTokenizer, GameVocabulary, TokenizedDecision
from sts2rl.env.types import GameObservation
from sts2rl.offline.clean import (
    DATASET_FORMAT_VERSION,
    DECISIONS_FILENAME,
    MANIFEST_FILENAME,
    Decision,
    read_decisions,
)


class DatasetError(RuntimeError):
    """Raised when a cleaned dataset cannot be read."""


class StaleDatasetError(DatasetError):
    """Raised when the action space no longer agrees with the stored snapshot."""


@dataclass(frozen=True)
class BCExample:
    """One tokenized decision and the index of the candidate the human picked."""

    decision: TokenizedDecision
    expert_index: int
    run_id: str
    step_index: int
    state_type: str | None


@dataclass(frozen=True)
class TokenizationFailure:
    """One decision the tokenizer refused, named so it can be looked up."""

    run_id: str
    step_index: int
    state_type: str | None
    error: str


class BCDataset:
    """The decisions of one cleaned dataset, tokenized on demand.

    Tokenizing is not cached.  It costs about 1.1 ms per decision against about
    2.3 ms for one encoder forward pass, so caching would trade a large amount
    of memory for a fraction of an epoch -- and ``TokenizedDecision`` cannot be
    serialized anyway, so the cache could never outlive the process.
    """

    def __init__(
        self,
        decisions: Sequence[Decision],
        manifest: dict[str, object],
        *,
        tokenizer: GameTokenizer | None = None,
        provider: LegalActionProvider | None = None,
    ) -> None:
        self.decisions = tuple(decisions)
        self.manifest = manifest
        self.tokenizer = tokenizer or GameTokenizer(GameVocabulary.from_bundled_data())
        self.provider = provider or LegalActionProvider()

    @classmethod
    def load(
        cls,
        dataset_dir: str | Path,
        *,
        tokenizer: GameTokenizer | None = None,
        provider: LegalActionProvider | None = None,
    ) -> BCDataset:
        """Read one dataset directory, validating its version and every label."""
        directory = Path(dataset_dir)
        manifest = cls._read_manifest(directory / MANIFEST_FILENAME)
        decisions_path = directory / DECISIONS_FILENAME
        if not decisions_path.is_file():
            raise DatasetError(f"dataset has no {DECISIONS_FILENAME}: {directory}")
        try:
            decisions = tuple(read_decisions(decisions_path))
        except (OSError, ValueError) as exc:
            raise DatasetError(f"cannot read {decisions_path}: {exc}") from exc
        if not decisions:
            raise DatasetError(f"dataset is empty: {decisions_path}")

        dataset = cls(
            decisions, manifest, tokenizer=tokenizer, provider=provider
        )
        dataset._require_current_action_space()
        return dataset

    def __len__(self) -> int:
        return len(self.decisions)

    def __iter__(self) -> Iterator[BCExample]:
        for index in range(len(self.decisions)):
            yield self[index]

    def __getitem__(self, index: int) -> BCExample:
        decision = self.decisions[index]
        return BCExample(
            decision=self.tokenizer.tokenize_decision(
                decision_observation(decision), candidate_actions(decision)
            ),
            expert_index=decision.expert_index,
            run_id=decision.run_id,
            step_index=decision.step_index,
            state_type=decision.state_type,
        )

    def run_ids(self) -> tuple[str, ...]:
        """Return the run ids in the order they first appear."""
        seen: dict[str, None] = {}
        for decision in self.decisions:
            seen.setdefault(decision.run_id, None)
        return tuple(seen)

    def verify(self) -> tuple[TokenizationFailure, ...]:
        """Tokenize everything and return the decisions the encoder refused."""
        failures: list[TokenizationFailure] = []
        for index, decision in enumerate(self.decisions):
            try:
                self[index]
            except Exception as exc:  # the tokenizer raises TokenizationError
                failures.append(
                    TokenizationFailure(
                        run_id=decision.run_id,
                        step_index=decision.step_index,
                        state_type=decision.state_type,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return tuple(failures)

    def split_by_run(
        self, holdout_fraction: float = 0.2
    ) -> tuple[BCDataset, BCDataset]:
        """Split into train and holdout **by run**, never by decision.

        Decisions inside one run are strongly correlated -- the same deck, the
        same map, often the same fight -- so a decision-level split puts near
        duplicates of the training data in the holdout and reports a number
        that measures nothing.  Splitting by run is what makes the holdout a
        comparison rather than a score.

        With too few runs to split there is nothing honest to return, so the
        decisions are cut at the tail and the caller is warned that the result
        is a sanity check and not a generalization measure.
        """
        if not 0 < holdout_fraction < 1:
            raise ValueError("holdout_fraction must be between 0 and 1")

        runs = self.run_ids()
        holdout_runs = max(1, round(len(runs) * holdout_fraction))
        if len(runs) - holdout_runs < 1:
            warnings.warn(
                f"{len(runs)} run(s) cannot be split by run; falling back to a "
                "tail split of the decisions. The holdout then shares its run "
                "with training, so its accuracy is a sanity check and NOT a "
                "measure of generalization. Record more runs.",
                stacklevel=2,
            )
            cut = max(1, int(len(self.decisions) * (1.0 - holdout_fraction)))
            return self._subset(self.decisions[:cut]), self._subset(
                self.decisions[cut:]
            )

        holdout = set(runs[-holdout_runs:])
        return (
            self._subset(
                [d for d in self.decisions if d.run_id not in holdout]
            ),
            self._subset([d for d in self.decisions if d.run_id in holdout]),
        )

    def _subset(self, decisions: Sequence[Decision]) -> BCDataset:
        return BCDataset(
            decisions,
            self.manifest,
            tokenizer=self.tokenizer,
            provider=self.provider,
        )

    def _require_current_action_space(self) -> None:
        """Refuse a dataset whose stored candidates the provider no longer gives."""
        for decision in self.decisions:
            stored = [dict(candidate) for candidate in decision.candidates]
            derived = [
                action.to_dict()
                for action in self.provider.candidates(dict(decision.raw_state))
            ]
            if stored != derived:
                raise StaleDatasetError(
                    "the action space no longer produces the candidates this "
                    f"dataset was cleaned against: run {decision.run_id} step "
                    f"{decision.step_index} ({decision.state_type}) stored "
                    f"{stored} but LegalActionProvider now gives {derived}. "
                    "expert_index is a position in that list, so the labels "
                    "cannot be trusted -- re-run sts2rl-bc-clean."
                )

    @staticmethod
    def _read_manifest(path: Path) -> dict[str, object]:
        if not path.is_file():
            raise DatasetError(f"dataset has no {MANIFEST_FILENAME}: {path.parent}")
        try:
            with path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, ValueError) as exc:
            raise DatasetError(f"cannot read {path}: {exc}") from exc
        if not isinstance(manifest, dict):
            raise DatasetError(f"{path}: manifest must be a JSON object")
        version = manifest.get("dataset_format_version")
        if version != DATASET_FORMAT_VERSION:
            raise DatasetError(
                f"{path}: unsupported dataset_format_version {version!r}; this "
                f"build reads {DATASET_FORMAT_VERSION}. Re-run sts2rl-bc-clean."
            )
        return manifest


def decision_observation(decision: Decision) -> GameObservation:
    """Return the agent input one stored decision represents."""
    return GameObservation(
        raw_state=dict(decision.raw_state),
        player_detail=(
            None if decision.player_detail is None else dict(decision.player_detail)
        ),
    )


def candidate_actions(decision: Decision) -> tuple[GameAction, ...]:
    """Return the stored candidate set as typed actions, in stored order."""
    return tuple(
        GameAction.from_dict(dict(candidate)) for candidate in decision.candidates
    )
