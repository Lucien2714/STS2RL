"""Add the ids the installed game defines and the bundled tables lack.

The tables under ``src/sts2rl/data/json`` are a third-party dump (spire-codex) of an older
build, and an id they lack reaches the model as ``<unknown>``: a Flex Potion's power and a
Dark Shackles debuff become the same row. The authority is the game itself, so the ids come
from STS2Simulator's export of the build it loads::

    cd ../STS2Simulator
    dotnet run --project src/Sts2Sim.Spike -c Release -- --export-vocabulary game_vocabulary.json
    cd ../STS2RL
    uv run python scripts/update_vocabulary.py ../STS2Simulator/game_vocabulary.json

Only ids that do not already resolve are added, with the game's own spelling and English
name, so every record that resolved before resolves to the same token. Nothing is removed:
an id the build no longer defines costs one unused row, while dropping one the API still
sends would be a silent ``<unknown>``. Test fixtures the game ships (``MOCK_*``,
``DEPRECATED_*``) never appear in play and are skipped.

Run it again after refreshing the tables from spire-codex, which replaces them wholesale.
Adding rows changes the vocabulary fingerprint, so existing checkpoints stop loading.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sts2rl.data import DEFAULT_DATA_DIR
from sts2rl.encoder.vocabulary import UNKNOWN_INDEX, GameVocabulary

TEST_ONLY_PREFIXES = ("MOCK_", "DEPRECATED_")

# Values the STS2MCP mod writes itself rather than from the game: card-grid screens it
# names (transform, upgrade, select, simple_select), the hand-selection modes, and the
# choose/bundle screens. Every other screen goes out under its class name.
MOD_SELECTION_TYPES = ("transform", "upgrade", "select", "simple_select", "upgrade_select", "choose", "bundle")
# The card-grid screens the mod renames (to transform, upgrade, select, simple_select); their
# class names never reach the API.
MOD_RENAMED_SCREENS = (
    "NDeckTransformSelectScreen",
    "NDeckUpgradeSelectScreen",
    "NDeckCardSelectScreen",
    "NSimpleCardSelectScreen",
)


def unresolved(vocabulary: GameVocabulary, table: str, records: list[dict]) -> list[dict]:
    """Records whose id and name both miss the table, as the tokenizer would look them up."""
    return [
        record
        for record in records
        if not record["id"].startswith(TEST_ONLY_PREFIXES)
        and vocabulary.lookup_first(table, [record["id"], record.get("name")]) == UNKNOWN_INDEX
    ]


def write_table(path: Path, records: list[dict]) -> None:
    """Write in the bundled files' own format, so a diff shows only the added records."""
    text = json.dumps(records, indent=2, ensure_ascii=False)
    with path.open("w", encoding="utf-8", newline="\r\n") as handle:
        handle.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", type=Path, help="the simulator's --export-vocabulary output")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    export = json.loads(args.export.read_text(encoding="utf-8"))
    print(f"game build {export['build']['version']} ({export['build']['commit']})")
    vocabulary = GameVocabulary.from_bundled_data(args.data_dir)
    added_total = 0
    for table, records in sorted(export["tables"].items()):
        path = args.data_dir / f"{table}.json"
        if not path.exists():
            # Fixed tables (rest_options) live in code; report what they lack.
            missing = [r["id"] for r in unresolved(vocabulary, table, records)]
            if missing:
                print(f"  {table:13} is a fixed table in vocabulary.py; it lacks {missing}")
            continue
        missing = unresolved(vocabulary, table, records)
        if not missing:
            continue
        existing = json.loads(path.read_text(encoding="utf-8"))
        for record in missing:
            entry = {"id": record["id"], "name": record.get("name")}
            if table == "events":
                entry["pages"] = record.get("pages", [])
            existing.append(entry)
        existing.sort(key=lambda r: str(r.get("id")))
        added_total += len(missing)
        print(f"  {table:13} +{len(missing)}: {[r['id'] for r in missing]}")
        if not args.dry_run:
            write_table(path, existing)

    # Fixed tables live in vocabulary.py; report what the build can send that they lack.
    fixed = dict(export.get("fixed", {}))
    fixed["selection_types"] = [
        *(name for name in fixed.get("selection_types", []) if name not in MOD_RENAMED_SCREENS),
        *MOD_SELECTION_TYPES,
    ]
    for table, values in sorted(fixed.items()):
        missing = [v for v in values if vocabulary.lookup(table, v) == UNKNOWN_INDEX]
        if missing:
            print(f"  {table:18} (fixed, vocabulary.py) lacks {missing}")

    if args.dry_run:
        print(f"dry run: {added_total} records would be added")
        return 0
    # Every live id must now resolve, and no added name may have made an existing alias
    # ambiguous (an ambiguous display name is dropped, which would unresolve an old id).
    vocabulary = GameVocabulary.from_bundled_data(args.data_dir)
    still = {
        table: [r["id"] for r in unresolved(vocabulary, table, records)]
        for table, records in export["tables"].items()
        if (args.data_dir / f"{table}.json").exists()
    }
    still = {table: ids for table, ids in still.items() if ids}
    print(f"added {added_total} records; still unresolved: {still or 'none'}")
    return 1 if still else 0


if __name__ == "__main__":
    sys.exit(main())
