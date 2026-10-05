"""Carry trained weights across a change to the model's inputs.

Every vocabulary refresh and every column added to ``encoder/schema.py`` used to
mean training from random weights again: a checkpoint is refused by fingerprint,
or fails to load on a changed tensor width, and both refusals are right, because
the old tensors would load and then mean something else. OpenAI Five met the same
problem across ten months of game patches and answered it with "surgery": rewrite
the old parameters into the new shape so that the new model computes exactly what
the old one did, then continue training. This module is that rewrite for
``GameEncoder``.

Two things change between models, and each has an exact remedy:

* **Vocabulary rows.** Tables are sorted, so a new token lands in the middle and
  moves every row after it. Rows are moved by token identity. A token the old
  vocabulary lacked gets the old ``<unknown>`` row, because that is what the old
  model saw for it.
* **Schema columns.** Categorical columns are embeddings *summed* into the row,
  so a new one starts as an all-zero table and adds nothing. Numeric columns
  enter one ``Linear`` per projection as ``[values..., masks...]``, so the input
  weights are moved by field name and a new field's two columns start at zero.

Both preserve the old model's output exactly on every state the old model could
encode -- except where the change was the point, such as an id that used to read
as ``<unknown>`` and now resolves. A new entity *kind* cannot be preserved: its
rows join the transformer's sequence and change the attention whatever their
weights. It is reported, not refused, and :func:`compare_outputs` measures it.

The old vocabulary and schema exist only in the old code, so every checkpoint now
carries a :func:`model_spec` of its own. A checkpoint written before that needs
one dumped from a checkout of the code it was trained with (``sts2rl-surgery spec``).

The optimizer state is never carried: Adam's moments are per row and per column,
and are meaningless after either moves.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch import Tensor

from sts2rl.actions.game_action import GameAction
from sts2rl.agents.ppo import CandidatePPOAgent
from sts2rl.encoder import EncoderConfig, GameEncoder, GameTokenizer, GameVocabulary
from sts2rl.encoder.entity_encoder import EntityTransformer
from sts2rl.encoder.spec import EVENT_OPTIONS, model_spec, schema_spec
from sts2rl.encoder.vocabulary import normalize_data_type, normalize_token
from sts2rl.env.types import GameObservation
from sts2rl.training.checkpoint import CheckpointManager, LoadedCheckpoint
from sts2rl.training.config import TrainingPlan, TrainingState

UNKNOWN_ROW = 1


class SurgeryError(ValueError):
    """The weights cannot be carried over as asked."""


# --------------------------------------------------------------------------- spec


def _spec_schema(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    # A vocabulary-only dump (scripts/migrate_vocabulary.py, before this module)
    # carries no schema; it was only ever used where the schema had not changed.
    # Shapes are still checked, so a wrong guess fails rather than misloads.
    return spec.get("schema") or schema_spec()


def embedding_tables(schema: Mapping[str, Any]) -> dict[str, str]:
    """Each embedding parameter the schema implies, and the table it indexes."""
    names = {
        f"entity_encoder.global_embeddings.{field}.weight": table
        for field, table in schema["global_categorical"]
    }
    for kind, columns in schema["entity_categorical"].items():
        for field, table in columns:
            if field != "entity_zone":
                key = EntityTransformer._embedding_key(kind, field)
                names[f"entity_encoder.entity_embeddings.{key}.weight"] = table
    names["entity_encoder.entity_type_embedding.weight"] = "entity_types"
    names["entity_encoder.zone_embedding.weight"] = "entity_zones"
    names["action_type_embedding.weight"] = "action_types"
    names["map_encoder.node_type_embedding.weight"] = "map_node_types"
    return names


def numeric_inputs(schema: Mapping[str, Any]) -> dict[str, list[str]]:
    """Each numeric projection's weight and the name of each of its input columns."""

    def values_and_masks(fields: list[str]) -> list[str]:
        return [f"value:{field}" for field in fields] + [f"mask:{field}" for field in fields]

    inputs = {
        "entity_encoder.global_numeric_projection.weight": values_and_masks(schema["global_numeric"]),
        "action_numeric_projection.weight": values_and_masks(schema["action_numeric"]),
        "map_encoder.node_numeric_projection.weight": values_and_masks(schema["map_numeric"]),
    }
    for kind, fields in schema["entity_numeric"].items():
        inputs[f"entity_encoder.entity_numeric_projections.{kind}.weight"] = (
            values_and_masks(fields) + ["active", "active_mask"]
        )
    return inputs


def _row_keys(spec: Mapping[str, Any], table: str) -> list[Any]:
    """The identity of each row of one table, comparable across vocabularies."""
    if table == EVENT_OPTIONS:
        return [(normalize_token(e), normalize_token(o)) for e, o in spec[EVENT_OPTIONS]]
    return [normalize_token(token) for token in spec["tables"][normalize_data_type(table)]]


# ---------------------------------------------------------------------- migration


@dataclass(frozen=True)
class Migration:
    """A rewritten encoder state and what the rewrite did."""

    state: dict[str, Tensor]
    report: dict[str, Any]

    @property
    def output_preserving(self) -> bool:
        """Whether every state the old model could encode is encoded as before.

        Not counting ids that used to miss and now resolve -- those change by
        design, and only :func:`compare_outputs` can count them.
        """
        return self.report["output_preserving"]


def migrate_encoder_state(
    old_state: Mapping[str, Tensor],
    old_spec: Mapping[str, Any],
    encoder: GameEncoder,
    vocabulary: GameVocabulary,
) -> Migration:
    """Rewrite ``old_state`` for ``encoder``, the model of this checkout."""
    old_schema, new_schema = _spec_schema(old_spec), schema_spec()
    new_spec = model_spec(vocabulary)
    new_state = encoder.state_dict()
    old_tables, new_tables = embedding_tables(old_schema), embedding_tables(new_schema)
    old_inputs, new_inputs = numeric_inputs(old_schema), numeric_inputs(new_schema)
    embeddings = {
        f"{name}.weight"
        for name, module in encoder.named_modules()
        if isinstance(module, torch.nn.Embedding)
    }
    if embeddings != set(new_tables):
        raise SurgeryError(
            "the embedding map is out of date with GameEncoder: unmapped "
            f"{sorted(embeddings - set(new_tables))}, stale {sorted(set(new_tables) - embeddings)}"
        )
    old_kinds = set(old_schema["entity_categorical"])
    new_kinds = sorted(set(new_schema["entity_categorical"]) - old_kinds)

    report: dict[str, Any] = {
        "tables": {},
        "columns": {},
        "zeroed": [],
        "fresh": [],
        "dropped": sorted(set(old_state) - set(new_state)),
        "new_entity_kinds": new_kinds,
    }
    result: dict[str, Tensor] = {}
    for name, fresh in new_state.items():
        old = old_state.get(name)
        if name in new_tables:
            if old is None:
                # A new categorical column: summed into its row, so zero adds nothing.
                result[name] = torch.zeros_like(fresh)
                report["zeroed"].append(name)
                continue
            result[name], report["tables"][name] = _move_rows(
                name, old, _row_keys(old_spec, old_tables[name]), _row_keys(new_spec, new_tables[name])
            )
        elif name in new_inputs:
            if old is None:
                result[name] = fresh
                report["fresh"].append(name)
                continue
            if name not in old_inputs:
                raise SurgeryError(f"{name}: the old spec does not describe its input columns")
            result[name], report["columns"][name] = _move_columns(
                name, old, old_inputs[name], new_inputs[name]
            )
        elif old is None:
            result[name] = fresh
            report["fresh"].append(name)
        elif old.shape != fresh.shape:
            raise SurgeryError(
                f"{name} changed shape outside the embeddings and numeric inputs: "
                f"{tuple(old.shape)} -> {tuple(fresh.shape)}"
            )
        else:
            result[name] = old.clone()

    removed_columns = any(change["removed"] for change in report["columns"].values())
    # A zeroed table of a new kind adds nothing to that kind's rows -- but the
    # kind's rows are new to the sequence, which is already counted below.
    report["output_preserving"] = not (
        report["fresh"] or report["dropped"] or removed_columns or new_kinds
    )
    encoder.load_state_dict(result)
    return Migration(result, report)


def _move_rows(name: str, old: Tensor, before: list[Any], after: list[Any]) -> tuple[Tensor, dict[str, int]]:
    if len(before) != old.shape[0]:
        raise SurgeryError(f"{name}: the old spec has {len(before)} rows, the weights {old.shape[0]}")
    index = {key: row for row, key in enumerate(before)}
    rows = old[UNKNOWN_ROW].repeat(len(after), 1)  # unseen -> what the old model saw
    moved = 0
    for row, key in enumerate(after):
        if key in index:
            rows[row] = old[index[key]]
            moved += index[key] != row
    kept = set(after)
    return rows, {
        "rows": len(after),
        "new": sum(key not in index for key in after),
        "dropped": sum(key not in kept for key in before),
        "moved": moved,
    }


def _move_columns(
    name: str, old: Tensor, before: list[str], after: list[str]
) -> tuple[Tensor, dict[str, list[str]]]:
    if len(before) != old.shape[1]:
        raise SurgeryError(
            f"{name}: the old spec has {len(before)} input columns, the weights {old.shape[1]}"
            " (a vocabulary-only dump cannot describe a schema change; dump the old"
            " spec with `sts2rl-surgery spec` in the old checkout)"
        )
    index = {column: position for position, column in enumerate(before)}
    weight = old.new_zeros((old.shape[0], len(after)))
    for position, column in enumerate(after):
        if column in index:
            weight[:, position] = old[:, index[column]]
    fields = lambda columns: sorted({c.split(":", 1)[-1] for c in columns})  # noqa: E731
    return weight, {
        "added": fields([c for c in after if c not in index]),
        "removed": fields([c for c in before if c not in set(after)]),
    }


# --------------------------------------------------------------------- checkpoints


def resolve_source(source: str | Path, vocabulary: GameVocabulary) -> tuple[CheckpointManager, Path]:
    """A run directory (its latest checkpoint) or a checkpoint file."""
    path = Path(source)
    if path.is_dir():
        manager = CheckpointManager(path, vocabulary)
        # Absolute: ``CheckpointManager.load`` reads a relative path as relative to
        # its own checkpoint directory, which ``resolve`` has already prefixed.
        return manager, manager.resolve("latest").resolve()
    if not path.is_file():
        raise SurgeryError(f"no run directory or checkpoint at {path}")
    return CheckpointManager(path.parent, vocabulary), path.resolve()


@dataclass(frozen=True)
class InitialWeights:
    """What a new run takes from an old one: the model, not the optimizer or counters."""

    source: Path
    encoder: dict[str, Tensor]
    return_scale: Mapping[str, Any]


def load_initial_weights(
    source: str | Path, *, vocabulary: GameVocabulary, encoder_config: EncoderConfig
) -> InitialWeights:
    """Read a checkpoint for ``sts2rl-train --init-from``, refusing what would misload."""
    manager, path = resolve_source(source, vocabulary)
    try:
        loaded: LoadedCheckpoint = manager.load(path, map_location="cpu")
    except Exception as exc:
        raise SurgeryError(
            f"{path}: {exc}. A checkpoint from an older vocabulary or schema can be "
            "carried over with `sts2rl-surgery migrate` first."
        ) from exc
    if loaded.plan.encoder != encoder_config:
        raise SurgeryError(
            f"{path}: encoder config {loaded.plan.encoder} does not match this run's "
            f"{encoder_config}; layer widths cannot be carried over"
        )
    encoder = loaded.agent_state.get("encoder")
    return_scale = loaded.agent_state.get("return_scale")
    if not isinstance(encoder, Mapping) or not isinstance(return_scale, Mapping):
        raise SurgeryError(f"{path}: the checkpoint has no encoder or return scale")
    return InitialWeights(path, dict(encoder), dict(return_scale))


def migrate_checkpoint(
    source: str | Path,
    to_run: str | Path,
    *,
    old_spec: Mapping[str, Any] | None = None,
) -> tuple[Path, Migration]:
    """Write ``source`` into a new run directory for this checkout's model.

    The result is an ordinary checkpoint: ``--resume`` continues it (with a fresh
    optimizer) and ``--init-from`` starts a new plan from it. Counters are kept,
    because the weights did take that many updates to train.
    """
    vocabulary = GameVocabulary.from_bundled_data()
    _, path = resolve_source(source, vocabulary)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != CheckpointManager.FORMAT_VERSION:
        raise SurgeryError(
            f"{path}: checkpoint format {payload.get('format_version')!r}, this build "
            f"reads {CheckpointManager.FORMAT_VERSION}"
        )
    spec = payload.get("model_spec") or old_spec
    if spec is None:
        raise SurgeryError(
            f"{path} predates model specs in checkpoints; pass --old-spec, dumped with "
            "`sts2rl-surgery spec` in a checkout of the code it was trained with"
        )
    if spec.get("fingerprint") != payload.get("vocabulary_fingerprint"):
        raise SurgeryError(f"{path} was not trained with the vocabulary in this spec")

    plan = TrainingPlan.from_dict(payload["training_plan"])
    plan = replace(plan, training=replace(plan.training, run_dir=Path(to_run)))
    encoder = GameEncoder(vocabulary, plan.encoder)
    migration = migrate_encoder_state(payload["agent_state"]["encoder"], spec, encoder, vocabulary)

    agent = CandidatePPOAgent(tokenizer=GameTokenizer(vocabulary), game_encoder=encoder, config=plan.ppo)
    agent.initialize_from(migration.state, payload["agent_state"]["return_scale"])
    target = CheckpointManager(to_run, vocabulary)
    target.initialize_run(plan, resume=False)
    saved = target.save(
        path.name,
        agent,
        plan,
        TrainingState.from_dict(payload["training_state"]),
        str(Path(to_run) / "tensorboard"),
    )
    (Path(to_run) / "migration.json").write_text(
        json.dumps(
            {
                "from": str(path),
                "old_fingerprint": spec["fingerprint"],
                "new_fingerprint": vocabulary.fingerprint(),
                "optimizer": "dropped (per-row and per-column moments do not survive a move)",
                **migration.report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return saved, migration


# ------------------------------------------------------------------ output check


def _decisions(dataset: Path, per_type: int) -> Iterator[dict[str, Any]]:
    """Up to ``per_type`` recorded decisions of each state type, in file order."""
    taken: Counter[str] = Counter()
    with gzip.open(dataset / "decisions.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            decision = json.loads(line)
            state_type = str(decision.get("state_type"))
            if taken[state_type] < per_type:
                taken[state_type] += 1
                yield decision


def decision_outputs(source: str | Path, dataset: str | Path, *, per_type: int = 50) -> dict[str, Any]:
    """A checkpoint's actor logits and critic value on recorded decisions.

    The candidates are the recorded ones, not re-derived, so two checkouts whose
    action spaces differ still score the same candidate lists. Run it on the old
    checkpoint in the old code and on the migrated one here, then
    :func:`compare_outputs`.
    """
    vocabulary = GameVocabulary.from_bundled_data()
    manager, path = resolve_source(source, vocabulary)
    loaded = manager.load(path, map_location="cpu")
    encoder = GameEncoder(vocabulary, loaded.plan.encoder)
    encoder.load_state_dict(dict(loaded.agent_state["encoder"]))
    encoder.eval()
    tokenizer = GameTokenizer(vocabulary)
    rows = []
    for decision in _decisions(Path(dataset), per_type):
        row: dict[str, Any] = {
            "key": f"{decision['run_id']}:{decision['step_index']}",
            "state_type": decision.get("state_type"),
        }
        try:
            tokenized = tokenizer.tokenize_decision(
                GameObservation(decision["raw_state"], decision.get("player_detail")),
                [GameAction.from_dict(candidate) for candidate in decision["candidates"]],
            )
            with torch.no_grad():
                output = encoder.policy_value(tokenized)
            row["logits"] = output.logits.tolist()
            row["value"] = float(output.value)
        except (ValueError, KeyError, TypeError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return {"source": str(path), "fingerprint": vocabulary.fingerprint(), "rows": rows}


def compare_outputs(
    before: Mapping[str, Any], after: Mapping[str, Any], *, tolerance: float = 1e-5
) -> dict[str, Any]:
    """How many recorded decisions two models score the same."""
    old_rows = {row["key"]: row for row in before["rows"]}
    by_type: dict[str, Counter[str]] = {}
    worst: list[tuple[float, str, str]] = []
    max_logit = max_value = 0.0
    for row in after["rows"]:
        old = old_rows.get(row["key"])
        counts = by_type.setdefault(str(row.get("state_type")), Counter())
        counts["decisions"] += 1
        if old is None or "error" in old or "error" in row:
            counts["unscored"] += 1
            continue
        if len(old["logits"]) != len(row["logits"]):
            counts["candidates_differ"] += 1
            continue
        logit = max(abs(a - b) for a, b in zip(old["logits"], row["logits"]))
        value = abs(old["value"] - row["value"])
        max_logit, max_value = max(max_logit, logit), max(max_value, value)
        same_choice = _argmax(old["logits"]) == _argmax(row["logits"])
        counts["same_argmax"] += same_choice
        if logit <= tolerance and value <= tolerance:
            counts["identical"] += 1
        else:
            worst.append((max(logit, value), row["key"], str(row.get("state_type"))))
    totals: Counter[str] = Counter()
    for counts in by_type.values():
        totals.update(counts)
    worst.sort(reverse=True)
    return {
        "tolerance": tolerance,
        "total": dict(totals),
        "by_state_type": {key: dict(value) for key, value in sorted(by_type.items())},
        "max_logit_difference": max_logit,
        "max_value_difference": max_value,
        "largest_differences": [
            {"key": key, "state_type": kind, "difference": difference}
            for difference, key, kind in worst[:10]
        ],
    }


def _argmax(values: list[float]) -> int:
    best = -math.inf
    index = 0
    for position, value in enumerate(values):
        if value > best:
            best, index = value, position
    return index
