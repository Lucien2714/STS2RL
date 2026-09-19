"""Behavior cloning: copy a human's choice out of the same candidate set.

The actor head already scores a variable candidate set, so cloning is one
cross entropy over those scores against the index the human picked.  Nothing
about the model changes; only where the gradient comes from.

**Only the actor is trained.**  The critic is left alone on purpose.  The
critic predicts in ``_ReturnScale``'s normalized space, and an expert's returns
live at a scale roughly twenty times an early policy's, so handing PPO a critic
calibrated to the expert's divisor and letting the divisor be relearned is the
failure CLAUDE.md names -- a resumed critic reading its own predictions at the
wrong scale is worse than one that was never saved.  The value target is also
orders of magnitude larger than a cross entropy, and ``GameEncoder`` is shared,
so a value term here would leave the encoder mostly trained by the critic when
the actor is the whole point.  The cleaned dataset keeps ``next_step_index``
and ``steps_to_run_end`` so this stays cheap to revisit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sts2rl.encoder import EncoderConfig, GameEncoder, GameVocabulary
from sts2rl.offline.dataset import BCDataset


# Bump when the artifact's payload changes.  ``sts2rl-train --init-encoder``
# refuses a version it does not know rather than loading tensors whose meaning
# has moved.
BC_FORMAT_VERSION = 1

BEST_CHECKPOINT_FILENAME = "bc_best.pt"
METRICS_FILENAME = "metrics.jsonl"


@dataclass(frozen=True)
class BCConfig:
    """Hyperparameters for one cloning run."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    epochs: int = 50
    minibatch_size: int = 32
    max_grad_norm: float = 1.0
    holdout_fraction: float = 0.2
    # Epochs without a lower holdout cross entropy before stopping.  Training
    # accuracy keeps climbing long after the holdout has turned, so the holdout
    # is the only thing worth stopping on -- and its *loss*, not its accuracy.
    # See ``BCTrainer.run``.
    patience: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("epochs", "minibatch_size", "patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must not be negative")
        if not 0 < self.holdout_fraction < 1:
            raise ValueError("holdout_fraction must be between 0 and 1")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


@dataclass
class EpochMetrics:
    """One epoch's numbers, as written to ``metrics.jsonl``."""

    epoch: int
    train_loss: float
    train_accuracy: float
    holdout_loss: float
    holdout_accuracy: float
    # Uniform choice over each decision's candidates, recomputed per split.  A
    # top-1 number means nothing without it: candidate sets here average about
    # eight actions, so 20% is chance and 33% is "always take the first one".
    holdout_chance: float
    holdout_first_candidate: float
    accuracy_by_state_type: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"type": "bc_epoch", **asdict(self)}


@dataclass(frozen=True)
class BCResult:
    """What one cloning run produced."""

    best_epoch: int
    # Accuracy *at* the saved epoch.  The epoch is chosen by holdout loss, so
    # this is not necessarily the highest accuracy any epoch reached.
    best_holdout_accuracy: float
    final_train_accuracy: float
    epochs_run: int
    checkpoint_path: Path | None
    history: tuple[EpochMetrics, ...]


def baseline_scores(dataset: BCDataset) -> tuple[float, float]:
    """Return the chance and always-first top-1 accuracies for one split.

    Cheap, exact, and computed from the stored candidate sets rather than the
    tokenizer, so it costs nothing to report beside every epoch.
    """
    if not len(dataset):
        return 0.0, 0.0
    chance = sum(1 / len(d.candidates) for d in dataset.decisions) / len(dataset)
    first = sum(1 for d in dataset.decisions if d.expert_index == 0) / len(dataset)
    return chance, first


class BCTrainer:
    """Train ``GameEncoder``'s actor head to reproduce demonstrated choices."""

    def __init__(
        self,
        encoder: GameEncoder,
        train: BCDataset,
        holdout: BCDataset,
        config: BCConfig | None = None,
        device: str | torch.device | None = None,
        out_dir: str | Path | None = None,
    ) -> None:
        self.encoder = encoder
        self.train_set = train
        self.holdout_set = holdout
        self.config = config or BCConfig()
        self.device = torch.device(device or "cpu")
        self.encoder.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self._random = random.Random(self.config.seed)

    def run(self) -> BCResult:
        """Train to ``epochs`` or until the holdout stops improving.

        The saved epoch is the one with the **lowest holdout cross entropy**,
        not the highest accuracy, for two reasons measured on six recorded
        runs.  Accuracy is a coarse count that wandered within about 1.5 points
        from epoch 1 onward, so choosing its maximum was choosing noise.  And
        the loss had already turned: it was lowest at epoch 1 and 8% higher at
        the accuracy-chosen epoch, which is a policy growing more confident
        without growing more right.  PPO *samples* from this policy, so an
        overconfident start is a start with too little entropy to explore --
        calibration is the property that transfers, and cross entropy is what
        measures it.  It is also the objective being trained.
        """
        if not len(self.train_set):
            raise ValueError("behavior cloning needs at least one training decision")

        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / METRICS_FILENAME).unlink(missing_ok=True)

        train_chance, train_first = baseline_scores(self.train_set)
        holdout_chance, holdout_first = baseline_scores(self.holdout_set)
        del train_chance, train_first

        history: list[EpochMetrics] = []
        best_loss = float("inf")
        best_accuracy = 0.0
        best_epoch = 0
        since_best = 0
        checkpoint_path: Path | None = None

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_accuracy = self._train_one_epoch()
            holdout_loss, holdout_accuracy, by_state = self._evaluate(self.holdout_set)
            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                holdout_loss=holdout_loss,
                holdout_accuracy=holdout_accuracy,
                holdout_chance=holdout_chance,
                holdout_first_candidate=holdout_first,
                accuracy_by_state_type=by_state,
            )
            history.append(metrics)
            self._append_metrics(metrics)

            if holdout_loss < best_loss:
                best_loss = holdout_loss
                best_accuracy = holdout_accuracy
                best_epoch = epoch
                since_best = 0
                checkpoint_path = self._save_best(metrics)
            else:
                since_best += 1
                if since_best >= self.config.patience:
                    break

        return BCResult(
            best_epoch=best_epoch,
            best_holdout_accuracy=best_accuracy,
            final_train_accuracy=history[-1].train_accuracy if history else 0.0,
            epochs_run=len(history),
            checkpoint_path=checkpoint_path,
            history=tuple(history),
        )

    def _train_one_epoch(self) -> tuple[float, float]:
        self.encoder.train(True)
        order = list(range(len(self.train_set)))
        self._random.shuffle(order)

        total_loss = 0.0
        # Accumulated on device and read once at the end: ``.item()`` per
        # example would force a synchronization on every decision.
        correct = torch.zeros((), dtype=torch.long, device=self.device)
        for start in range(0, len(order), self.config.minibatch_size):
            batch = order[start : start + self.config.minibatch_size]
            losses: list[Tensor] = []
            # One forward pass per decision, stacked before a single backward.
            # ``GameEncoder`` already batches a decision's whole candidate set;
            # padding several decisions together is cross-step batching, which
            # the codebase deliberately leaves as open work rather than
            # inventing here.
            for index in batch:
                example = self.train_set[index]
                logits = self._logits(example.decision)
                target = torch.tensor(example.expert_index, device=self.device)
                losses.append(F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0)))
                correct += (torch.argmax(logits) == target).long()
            loss = torch.stack(losses).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.encoder.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch)

        count = len(order)
        return total_loss / count, int(correct) / count

    def _evaluate(
        self, dataset: BCDataset
    ) -> tuple[float, float, dict[str, float]]:
        if not len(dataset):
            return 0.0, 0.0, {}
        self.encoder.train(False)
        losses: list[Tensor] = []
        hit_flags: list[Tensor] = []
        keys: list[str] = []
        with torch.no_grad():
            for index in range(len(dataset)):
                example = dataset[index]
                logits = self._logits(example.decision)
                target = torch.tensor(example.expert_index, device=self.device)
                losses.append(
                    F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
                )
                hit_flags.append((torch.argmax(logits) == target).long())
                keys.append(example.state_type or "<unknown>")
        # One synchronization for the whole split rather than two per decision.
        hits_per_example = torch.stack(hit_flags).cpu().tolist()
        mean_loss = float(torch.stack(losses).mean().cpu())

        hits: dict[str, int] = {}
        counts: dict[str, int] = {}
        for key, hit in zip(keys, hits_per_example):
            counts[key] = counts.get(key, 0) + 1
            hits[key] = hits.get(key, 0) + hit
        by_state = {key: hits[key] / counts[key] for key in sorted(counts)}
        return mean_loss, sum(hits_per_example) / len(dataset), by_state

    def _logits(self, decision) -> Tensor:
        return self.encoder.policy_value(decision.to(self.device)).logits

    def _append_metrics(self, metrics: EpochMetrics) -> None:
        if self.out_dir is None:
            return
        path = self.out_dir / METRICS_FILENAME
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics.to_json(), sort_keys=True))
            handle.write("\n")

    def _save_best(self, metrics: EpochMetrics) -> Path | None:
        if self.out_dir is None:
            return None
        return save_bc_checkpoint(
            self.out_dir / BEST_CHECKPOINT_FILENAME,
            encoder=self.encoder,
            vocabulary_fingerprint=_fingerprint_of(self.train_set),
            encoder_config=self.encoder.config,
            bc_config=self.config,
            metrics=metrics,
            dataset_manifest=self.train_set.manifest,
        )


def save_bc_checkpoint(
    path: str | Path,
    *,
    encoder: GameEncoder,
    vocabulary_fingerprint: str,
    encoder_config: EncoderConfig,
    bc_config: BCConfig,
    metrics: EpochMetrics,
    dataset_manifest: dict[str, object] | None = None,
) -> Path:
    """Publish an encoder-weights-only artifact, atomically.

    Deliberately *not* a PPO checkpoint.  Adam's state here belongs to a
    different objective and the counters belong to a run that never happened,
    so the artifact carries only what transfers: the weights, and enough
    identity to refuse them when they would mean something else.
    """
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_suffix(final_path.suffix + ".tmp")
    payload = {
        "format_version": BC_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "encoder": {
            name: tensor.detach().cpu().clone()
            for name, tensor in encoder.state_dict().items()
        },
        "vocabulary_fingerprint": vocabulary_fingerprint,
        "encoder_config": asdict(encoder_config),
        "bc_config": asdict(bc_config),
        "metrics": metrics.to_json(),
        "dataset": _dataset_provenance(dataset_manifest),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, final_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return final_path


def split_dataset(
    dataset: BCDataset, config: BCConfig
) -> tuple[BCDataset, BCDataset]:
    """Split by run, which is the only split that measures generalization."""
    return dataset.split_by_run(config.holdout_fraction)


def _fingerprint_of(dataset: BCDataset) -> str:
    return dataset.tokenizer.vocabulary.fingerprint()


def _dataset_provenance(manifest: dict[str, object] | None) -> dict[str, object]:
    """Keep enough of the manifest to say what these weights were cloned from."""
    if not manifest:
        return {}
    runs = manifest.get("runs")
    return {
        "decisions": manifest.get("decisions"),
        "dataset_format_version": manifest.get("dataset_format_version"),
        "run_ids": (
            [entry.get("run_id") for entry in runs if isinstance(entry, dict)]
            if isinstance(runs, list)
            else []
        ),
    }


def load_bc_encoder_state(
    path: str | Path,
    *,
    vocabulary: GameVocabulary,
    encoder_config: EncoderConfig,
) -> dict[str, Tensor]:
    """Read one BC artifact, refusing weights that would mean something else.

    The two checks are the same ones a PPO checkpoint makes, for the same
    reason: a vocabulary whose rows moved and an encoder whose tensors changed
    shape both produce weights that load and then behave as something they are
    not.
    """
    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"failed to load BC checkpoint {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: BC checkpoint payload must be a mapping")
    if payload.get("format_version") != BC_FORMAT_VERSION:
        raise ValueError(
            f"{source}: unsupported BC format_version "
            f"{payload.get('format_version')!r}; this build reads "
            f"{BC_FORMAT_VERSION}"
        )
    if payload.get("vocabulary_fingerprint") != vocabulary.fingerprint():
        raise ValueError(
            f"{source}: BC checkpoint vocabulary fingerprint does not match the "
            "bundled data; the embedding rows would mean something else"
        )
    stored_config = payload.get("encoder_config")
    if stored_config != asdict(encoder_config):
        raise ValueError(
            f"{source}: BC checkpoint encoder config {stored_config!r} does not "
            f"match this run's {asdict(encoder_config)!r}"
        )
    state = payload.get("encoder")
    if not isinstance(state, dict):
        raise ValueError(f"{source}: BC checkpoint encoder state must be a mapping")
    return dict(state)


def summarize(result: BCResult, history: Sequence[EpochMetrics] | None = None) -> str:
    """Render a short human-readable verdict for the CLI.

    Every number comes from the **best** epoch -- the one whose weights were
    saved.  Mixing it with the last epoch's per-screen accuracy would describe
    two different models in one table.
    """
    epochs = history if history is not None else result.history
    if not epochs:
        return "no epochs ran"
    best = next(
        (epoch for epoch in epochs if epoch.epoch == result.best_epoch), epochs[-1]
    )
    lines = [
        f"epochs run           {result.epochs_run}",
        f"best epoch           {best.epoch} (saved)",
        f"train accuracy       {best.train_accuracy:.1%}",
        f"holdout accuracy     {best.holdout_accuracy:.1%}",
        f"  chance             {best.holdout_chance:.1%}",
        f"  always-first       {best.holdout_first_candidate:.1%}",
    ]
    if best.accuracy_by_state_type:
        lines.append("holdout accuracy by screen:")
        for state_type, accuracy in best.accuracy_by_state_type.items():
            lines.append(f"  {state_type:<14} {accuracy:.1%}")
    return "\n".join(lines)
