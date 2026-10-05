"""``sts2rl-surgery``: carry trained weights across a vocabulary or schema change.

    # what the code of this checkout feeds the model (run it in an OLD checkout
    # for a checkpoint written before checkpoints carried their own spec)
    sts2rl-surgery spec old_spec.json

    # rewrite a checkpoint for this checkout's vocabulary and schema
    sts2rl-surgery migrate --from runs/old[/checkpoints/update_000100.pt] --to-run runs/old-v5 \\
        [--old-spec old_spec.json]

    # did the rewrite preserve the model? score recorded decisions with each
    # (the old checkpoint in the old checkout), then compare
    sts2rl-surgery outputs --from runs/old --dataset data/bc/v6 --out before.json
    sts2rl-surgery outputs --from runs/old-v5 --dataset data/bc/v6 --out after.json
    sts2rl-surgery compare before.json after.json

Then ``sts2rl-train --resume`` continues the migrated run, or
``sts2rl-train --init-from runs/old-v5`` starts a new plan from its weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from sts2rl.encoder import GameVocabulary
from sts2rl.encoder.spec import model_spec
from sts2rl.training.surgery import (
    compare_outputs,
    decision_outputs,
    migrate_checkpoint,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sts2rl-surgery",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    spec = commands.add_parser("spec", help="write this checkout's vocabulary and schema")
    spec.add_argument("out", type=Path)

    migrate = commands.add_parser("migrate", help="rewrite a checkpoint for this checkout")
    migrate.add_argument("--from", dest="source", required=True, help="run directory or checkpoint file")
    migrate.add_argument("--to-run", type=Path, required=True, help="a new, empty run directory")
    migrate.add_argument(
        "--old-spec",
        type=Path,
        help="the spec the checkpoint was trained with, if it does not carry one",
    )

    outputs = commands.add_parser("outputs", help="score recorded decisions with a checkpoint")
    outputs.add_argument("--from", dest="source", required=True, help="run directory or checkpoint file")
    outputs.add_argument("--dataset", type=Path, required=True, help="a directory with decisions.jsonl.gz")
    outputs.add_argument("--out", type=Path, required=True)
    outputs.add_argument("--per-type", type=int, default=50, help="decisions per state type (default 50)")

    compare = commands.add_parser("compare", help="compare two `outputs` files")
    compare.add_argument("before", type=Path)
    compare.add_argument("after", type=Path)
    compare.add_argument("--tolerance", type=float, default=1e-5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    if args.command == "spec":
        spec = model_spec(GameVocabulary.from_bundled_data())
        args.out.write_text(json.dumps(spec), encoding="utf-8")
        print(f"wrote {args.out}: fingerprint {spec['fingerprint'][:12]}")
    elif args.command == "migrate":
        old_spec = (
            json.loads(args.old_spec.read_text(encoding="utf-8")) if args.old_spec else None
        )
        saved, migration = migrate_checkpoint(args.source, args.to_run, old_spec=old_spec)
        report = migration.report
        for name, counts in sorted(report["tables"].items()):
            if counts["new"] or counts["dropped"] or counts["moved"]:
                print(f"  rows    {name}: {counts}")
        for name, change in sorted(report["columns"].items()):
            if change["added"] or change["removed"]:
                print(f"  columns {name}: {change}")
        for key in ("zeroed", "fresh", "dropped", "new_entity_kinds"):
            if report[key]:
                print(f"  {key}: {report[key]}")
        verdict = (
            "preserves the old outputs (except ids that now resolve)"
            if migration.output_preserving
            else "CHANGES the outputs; measure how much with `outputs` and `compare`"
        )
        print(f"saved {saved}: the migration {verdict}")
    elif args.command == "outputs":
        result = decision_outputs(args.source, args.dataset, per_type=args.per_type)
        args.out.write_text(json.dumps(result), encoding="utf-8")
        errors = sum("error" in row for row in result["rows"])
        print(f"{len(result['rows'])} decisions ({errors} not encodable) -> {args.out}")
    else:
        before = json.loads(args.before.read_text(encoding="utf-8"))
        after = json.loads(args.after.read_text(encoding="utf-8"))
        print(json.dumps(compare_outputs(before, after, tolerance=args.tolerance), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
