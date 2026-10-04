"""Carry a checkpoint across a vocabulary change by moving its embedding rows by token.

A vocabulary update adds tokens, and tables are sorted, so new tokens land in the
middle and every row after them moves: an old checkpoint would read each token's
neighbour. ``CheckpointManager`` refuses it by fingerprint, which is right. This
script makes a new checkpoint whose every embedding row sits under the token it was
trained for.

Only embedding tables depend on the vocabulary (no layer's width does), so nothing
else changes. A token the old vocabulary lacked gets the old ``<unknown>`` row,
because that is what the old model saw for it: a state the old model saw is encoded
exactly as before, unless one of its ids now resolves where it used to miss (a power
reached through ``API_SUFFIXES``), which is the point of the update. The optimizer
state is dropped, since its moments are per row; a resumed run starts Adam afresh.

Two steps, because the old vocabulary only exists in the old code:

    # in a checkout of the code the checkpoint was trained with
    uv run python scripts/migrate_vocabulary.py dump old_vocabulary.json

    # here
    uv run python scripts/migrate_vocabulary.py migrate --old-vocabulary old_vocabulary.json \\
        --from-run runs/intent8h --checkpoint latest --to-run runs/intent8h-v4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def dump(out: Path) -> None:
    from sts2rl.encoder.vocabulary import GameVocabulary

    vocabulary = GameVocabulary.from_bundled_data()
    payload = {
        "fingerprint": vocabulary.fingerprint(),
        "tables": {name: list(table.tokens) for name, table in vocabulary.tables.items()},
        "event_options": [list(pair) for pair in vocabulary.event_options.pairs],
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {out}: {len(payload['tables'])} tables, fingerprint {payload['fingerprint'][:12]}")


def embedding_tables(encoder) -> dict[str, str]:
    """Each embedding parameter of a GameEncoder and the vocabulary table it indexes."""
    from sts2rl.encoder.entity_encoder import EntityTransformer
    from sts2rl.encoder.schema import ENTITY_CATEGORICAL, GLOBAL_CATEGORICAL

    names = {f"entity_encoder.global_embeddings.{field}.weight": table for field, table in GLOBAL_CATEGORICAL}
    for kind, columns in ENTITY_CATEGORICAL.items():
        for field, table in columns:
            if field != "entity_zone":
                key = EntityTransformer._embedding_key(kind, field)
                names[f"entity_encoder.entity_embeddings.{key}.weight"] = table
    names["entity_encoder.entity_type_embedding.weight"] = "entity_types"
    names["entity_encoder.zone_embedding.weight"] = "entity_zones"
    names["action_type_embedding.weight"] = "action_types"
    names["map_encoder.node_type_embedding.weight"] = "map_node_types"
    embeddings = {name for name, module in encoder.named_modules() if isinstance(module, torch.nn.Embedding)}
    found = {name.removesuffix(".weight") for name in names}
    if embeddings != found:
        raise SystemExit(f"embedding map out of date: unmapped {sorted(embeddings - found)}, stale {sorted(found - embeddings)}")
    return names


def keys_of(vocabulary, table: str) -> list:
    """The identity of each row, in index order, comparable across vocabularies."""
    from sts2rl.encoder.vocabulary import normalize_token

    if table == "event_options":
        return [(normalize_token(e), normalize_token(o)) for e, o in vocabulary.event_options.pairs]
    return [normalize_token(token) for token in vocabulary.table(table).tokens]


def old_keys(old: dict, table: str) -> list:
    from sts2rl.encoder.vocabulary import normalize_data_type, normalize_token

    if table == "event_options":
        return [(normalize_token(e), normalize_token(o)) for e, o in old["event_options"]]
    return [normalize_token(token) for token in old["tables"][normalize_data_type(table)]]


def migrate(args: argparse.Namespace) -> None:
    from sts2rl.agents.ppo import CandidatePPOAgent
    from sts2rl.encoder import GameEncoder, GameTokenizer, GameVocabulary
    from sts2rl.training.checkpoint import CheckpointManager
    from sts2rl.training.config import TrainingPlan, TrainingState

    old = json.loads(args.old_vocabulary.read_text(encoding="utf-8"))
    source = CheckpointManager(args.from_run, GameVocabulary.from_bundled_data())
    path = source.resolve(args.checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("vocabulary_fingerprint") != old["fingerprint"]:
        raise SystemExit("the checkpoint was not trained with the dumped vocabulary")

    vocabulary = GameVocabulary.from_bundled_data()
    plan = TrainingPlan.from_dict(payload["training_plan"])
    plan = TrainingPlan.from_dict({**plan.to_dict(), "training": {**plan.training.to_dict(), "run_dir": str(args.to_run)}})
    encoder = GameEncoder(vocabulary, plan.encoder)
    old_state = payload["agent_state"]["encoder"]
    new_state = encoder.state_dict()
    tables = embedding_tables(encoder)
    report = {}
    for name, tensor in old_state.items():
        if name not in tables:
            if new_state[name].shape != tensor.shape:
                raise SystemExit(f"{name} changed shape outside the embeddings: {tuple(tensor.shape)} -> {tuple(new_state[name].shape)}")
            new_state[name] = tensor.clone()
            continue
        table = tables[name]
        before = {key: index for index, key in enumerate(old_keys(old, table))}
        after = keys_of(vocabulary, table)
        if len(before) != tensor.shape[0]:
            raise SystemExit(f"{name}: the dump has {len(before)} rows for {table}, the checkpoint {tensor.shape[0]}")
        rows = tensor[1].repeat(len(after), 1)  # unseen -> what the old model saw: <unknown>
        moved = 0
        for index, key in enumerate(after):
            if key in before:
                rows[index] = tensor[before[key]]
                moved += before[key] != index
        new_state[name] = rows
        report[table] = {"rows": len(after), "new": sum(key not in before for key in after),
                         "dropped": sum(key not in set(after) for key in before), "moved": moved}
    encoder.load_state_dict(new_state)

    agent = CandidatePPOAgent(tokenizer=GameTokenizer(vocabulary), game_encoder=encoder, config=plan.ppo)
    agent._return_scale.load(payload["agent_state"]["return_scale"])
    state = TrainingState.from_dict(payload["training_state"])
    target = CheckpointManager(args.to_run, vocabulary)
    target.initialize_run(plan, resume=False)
    saved = target.save(path.name, agent, plan, state, str(Path(args.to_run) / "tensorboard"))
    (Path(args.to_run) / "migration.json").write_text(json.dumps({
        "from": str(path), "old_fingerprint": old["fingerprint"], "new_fingerprint": vocabulary.fingerprint(),
        "tables": report, "optimizer": "dropped (per-row moments do not survive a row move)",
    }, indent=2), encoding="utf-8")
    for table, counts in sorted(report.items()):
        if counts["new"] or counts["dropped"] or counts["moved"]:
            print(f"  {table:24} {counts}")
    print(f"saved {saved}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    dump_parser = sub.add_parser("dump", help="write this checkout's vocabulary tables")
    dump_parser.add_argument("out", type=Path)
    migrate_parser = sub.add_parser("migrate", help="rewrite a checkpoint for this checkout's vocabulary")
    migrate_parser.add_argument("--old-vocabulary", type=Path, required=True)
    migrate_parser.add_argument("--from-run", type=Path, required=True)
    migrate_parser.add_argument("--checkpoint", default="latest")
    migrate_parser.add_argument("--to-run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "dump":
        dump(args.out)
    else:
        migrate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
