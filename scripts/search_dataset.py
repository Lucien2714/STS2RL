"""Turn the decisions a combat search recorded during training into a cloning dataset.

``sts2rl-train --search-combat`` writes every searched decision to
``<run>/search/decisions.jsonl.gz`` in the cleaned-decision format plus the root's
visit shares (``target_distribution``). This validates each line, keeps the ones
the current action space still reproduces, and writes a dataset directory
``sts2rl-bc-train`` reads: the actor is then trained on the search's soft labels.

    uv run python scripts/search_dataset.py --runs runs/step2-search --out data/distill/v1
    uv run sts2rl-bc-train --dataset data/distill/v1 --out runs/distill-v1

Several runs can be merged; their run ids are prefixed with the run's name so a
lane's episode numbers from two runs never collide, and the holdout split (by run
id) never puts two halves of one episode on both sides.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from sts2rl.agents.action_space import LegalActionProvider, NoLegalActionsError
from sts2rl.offline.clean import DATASET_FORMAT_VERSION, DECISIONS_FILENAME, MANIFEST_FILENAME, Decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="training runs that recorded search decisions")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    provider = LegalActionProvider()
    kept: list[Decision] = []
    rejected: Counter[str] = Counter()
    per_run: Counter[str] = Counter()
    by_state: Counter[str] = Counter()
    for run in args.runs:
        source = run / "search" / DECISIONS_FILENAME
        try:
            with gzip.open(source, "rt", encoding="utf-8") as handle:
                lines = handle.readlines()
        except EOFError:
            # A run still writing, or killed mid-line: every complete line before it stands.
            with gzip.open(source, "rt", encoding="utf-8") as handle:
                lines = []
                try:
                    for line in handle:
                        lines.append(line)
                except EOFError:
                    pass
        for line in lines:
            if not line.endswith("\n"):
                rejected["truncated_line"] += 1
                continue
            try:
                value = json.loads(line)
                value["run_id"] = f"{run.name}:{value['run_id']}"
                decision = Decision.from_json(value)
            except (ValueError, KeyError) as exc:
                rejected[f"invalid: {str(exc)[:60]}"] += 1
                continue
            if decision.target_distribution is None:
                rejected["no_target_distribution"] += 1
                continue
            try:
                current = [action.to_dict() for action in provider.candidates(dict(decision.raw_state))]
            except NoLegalActionsError:
                rejected["no_candidates_now"] += 1
                continue
            if current != [dict(c) for c in decision.candidates]:
                # The label is a position; a candidate set that moved would move it silently.
                rejected["candidates_changed"] += 1
                continue
            kept.append(decision)
            per_run[run.name] += 1
            by_state[decision.state_type or "<unknown>"] += 1

    if not kept:
        print("no usable decisions", dict(rejected))
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out / DECISIONS_FILENAME, "wt", encoding="utf-8", newline="\n") as handle:
        for decision in kept:
            handle.write(json.dumps(decision.to_json()) + "\n")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "source": "combat search (docs/mcts, step 3)",
        "decisions": len(kept),
        "episodes": len({d.run_id for d in kept}),
        "rejected": dict(rejected),
        "runs": [{"path": str(run), "kept": per_run[run.name]} for run in args.runs],
        "by_state_type": dict(by_state),
    }
    (args.out / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("decisions", "episodes", "rejected", "by_state_type")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
