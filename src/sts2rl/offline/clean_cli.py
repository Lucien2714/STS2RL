"""Command line entry point for ``sts2rl-bc-clean``."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sts2rl.offline.clean import CleanResult, clean_records
from sts2rl.offline.record import RecordError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sts2rl-bc-clean",
        description=(
            "Turn recorded human runs into a behavior-cloning dataset of "
            "decisions the current action space can express."
        ),
    )
    parser.add_argument(
        "--records",
        required=True,
        type=Path,
        help="directory of gameplay_records *.json files, or one such file",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="dataset directory to write decisions.jsonl.gz and manifest.json into",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "read the dataset back and tokenize every decision, reporting any "
            "that the encoder cannot accept"
        ),
    )
    parser.add_argument(
        "--only-victories",
        action="store_true",
        help=(
            "clone only runs that were won; a run that died demonstrates "
            "losing from the mistake that killed it onward"
        ),
    )
    parser.add_argument(
        "--min-floor",
        type=int,
        help="clone only runs that reached at least this floor",
    )
    parser.add_argument(
        "--max-unmatched-examples",
        type=int,
        default=20,
        help=(
            "how many demonstrated-but-unreachable actions to name per run in "
            "the manifest (default: 20)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = clean_records(
            args.records,
            args.out,
            max_unmatched_examples=args.max_unmatched_examples,
            only_victories=args.only_victories,
            min_floor=args.min_floor,
        )
    except (RecordError, OSError, ValueError) as exc:
        print(f"sts2rl-bc-clean: {exc}", file=sys.stderr)
        return 2

    _report(result)

    if not result.runs:
        print(
            "sts2rl-bc-clean: every record was excluded; nothing to clean",
            file=sys.stderr,
        )
        return 2

    if args.verify:
        return _verify(result)
    print(
        "\nPass --verify to tokenize every decision and confirm the encoder "
        "accepts it."
    )
    return 0


def _report(result: CleanResult) -> None:
    print(f"records:   {len(result.runs)}")
    if result.excluded:
        print(f"excluded:  {len(result.excluded)}")
        for run in result.excluded:
            print(f"  {run.run_id}: {run.reason}")
    print(f"steps:     {result.steps}")
    print(f"kept:      {result.decisions}")
    print("rejected:")
    rejected = result.rejected
    for reason, count in rejected.items():
        print(f"  {reason:<24} {count}")
    total = result.decisions + sum(rejected.values())
    print(f"  {'TOTAL (= steps)':<24} {total}")
    # Every step lands in exactly one bucket by construction; printing the sum
    # is how a reader confirms that rather than taking it on faith.
    if total != result.steps:
        print(
            f"sts2rl-bc-clean: ledger does not account for every step "
            f"({total} != {result.steps})",
            file=sys.stderr,
        )

    for report in result.runs:
        outcome = (
            "victory"
            if report.victory
            else "abandoned"
            if report.abandoned
            else "defeat"
        )
        print(
            f"\n{report.run_id}  seed={report.seed}  {report.character}  "
            f"{outcome} floor={report.floor_reached}  "
            f"reloads={report.num_reloads}"
        )
        print(f"  kept {report.kept} of {report.steps} steps")
        for example in report.unmatched_examples:
            print(
                f"  unreachable: step {example['step_index']:>4} "
                f"{example['state_type']}/{example['action']['type']} "
                f"({example['reason']})"
            )

    print(f"\nwrote {result.decisions_path}")
    print(f"wrote {result.manifest_path}")


def _verify(result: CleanResult) -> int:
    """Tokenize every decision, which is the only real proof it is usable.

    Imported here rather than at module scope because cleaning depends on the
    action layer and never on the encoder; only this opt-in pass crosses that
    line.
    """
    from sts2rl.offline.dataset import BCDataset, DatasetError

    try:
        dataset = BCDataset.load(result.decisions_path.parent)
    except DatasetError as exc:
        print(f"sts2rl-bc-clean: {exc}", file=sys.stderr)
        return 2

    failures = dataset.verify()
    print(f"\nverify: tokenized {len(dataset)} decisions, {len(failures)} failed")
    for failure in failures[:20]:
        print(
            f"  {failure.run_id} step {failure.step_index} "
            f"({failure.state_type}): {failure.error}",
            file=sys.stderr,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
