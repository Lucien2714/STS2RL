"""Fit the combat search's leaf evaluator to the fights the search played, and check it.

Every searched decision in a ``boss_mcts_eval.py --record-features`` run is one row:
the evaluator's features of that state, labelled with whether the fight was won. The
fit is a logistic regression (``sts2rl.search.fit_weights``). Fights, not rows, are
the unit of evidence, so each fight's rows share one fight's weight: a long fight
should not outvote a short one for being long.

    uv run python scripts/fit_leaf_evaluator.py runs/boss-mcts/fit_round1.jsonl \\
        --out runs/boss-mcts/weights_round1.json --check runs/boss-mcts/holdout.jsonl

Each run is reported with the fitted and the default weights: AUC over fights (the
estimate at a fight's first decision, against its outcome), AUC over decisions, and
a calibration table of predicted against observed win rate. ``--check`` runs are
scored without being fitted on, which is the honest number.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from sts2rl.search.evaluate import DEFAULT_WEIGHTS, FEATURE_SIGNS, features, fit_weights

BUCKETS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0)


def load(paths: Sequence[Path]) -> list[dict]:
    """Fights with their decision features, recomputed from the recorded states when a
    run kept them, so the current feature set is what gets fitted."""
    fights = []
    for path in paths:
        states = path.with_name(path.stem + ".states.jsonl.gz")
        if states.exists():
            with gzip.open(states, "rt", encoding="utf-8") as handle:
                for line in handle:
                    fight = json.loads(line)
                    fight["features"] = [features(state) for state in fight.pop("states")]
                    fights.append(fight)
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                fights.append(json.loads(line))
    return [fight for fight in fights if fight.get("features")]


def rows_of(fights: Sequence[dict]) -> tuple[list[dict], list[bool], list[float]]:
    rows, outcomes, weights = [], [], []
    for fight in fights:
        features = fight["features"]
        rows.extend(features)
        outcomes.extend([bool(fight["won"])] * len(features))
        weights.extend([1.0 / len(features)] * len(features))
    return rows, outcomes, weights


def auc(scores: Sequence[float], outcomes: Sequence[bool]) -> float | None:
    """Probability that a won row outranks a lost one (ties count half)."""
    positives = sum(outcomes)
    negatives = len(outcomes) - positives
    if not positives or not negatives:
        return None
    ranked = sorted(zip(scores, outcomes))
    rank_sum, index = 0.0, 0
    while index < len(ranked):
        end = index
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        rank_sum += (index + 1 + end) / 2 * sum(1 for _, won in ranked[index:end] if won)
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def report(label: str, weights: Mapping[str, float], fights: Sequence[dict]) -> dict:
    rows, outcomes, row_weights = rows_of(fights)
    scores = [_p(weights, phi) for phi in rows]
    first = [_p(weights, fight["features"][0]) for fight in fights]
    won = [bool(fight["won"]) for fight in fights]
    summary = {
        "fights": len(fights),
        "won": sum(won),
        "rows": len(rows),
        "auc_first_decision": auc(first, won),
        "auc_rows": auc(scores, outcomes),
        "calibration": [],
    }
    print(f"{label}: {len(fights)} fights, {sum(won)} won ({sum(won) / max(len(fights), 1):.1%}), {len(rows)} rows")
    print(f"  AUC first decision {_fmt(summary['auc_first_decision'])}   AUC all decisions {_fmt(summary['auc_rows'])}")
    print("  predicted -> observed win rate (rows weighted by fight):")
    for low, high in zip(BUCKETS, BUCKETS[1:]):
        bucket = [(s, o, w) for s, o, w in zip(scores, outcomes, row_weights)
                  if low <= s < high or (high == 1.0 and s == 1.0)]
        if not bucket:
            continue
        mass = sum(w for _, _, w in bucket)
        predicted = sum(s * w for s, _, w in bucket) / mass
        observed = sum(o * w for _, o, w in bucket) / mass
        summary["calibration"].append({"bucket": [low, high], "predicted": predicted, "observed": observed,
                                       "rows": len(bucket), "fights": mass})
        print(f"    [{low:.2f}, {high:.2f})  pred {predicted:.3f}  obs {observed:.3f}  rows {len(bucket):5}  fights {mass:6.1f}")
    return summary


def _p(weights: Mapping[str, float], phi: Mapping[str, float]) -> float:
    """The evaluator's win probability, from features already computed."""
    score = sum(weights.get(name, 0.0) * value for name, value in phi.items())
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="runs to fit on")
    parser.add_argument("--out", type=Path, required=True, help="where to write the fitted weights")
    parser.add_argument("--check", nargs="*", type=Path, default=[], help="runs to score without fitting on")
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--free", action="store_true", help="fit without FEATURE_SIGNS")
    args = parser.parse_args()

    fights = load(args.runs)
    rows, outcomes, row_weights = rows_of(fights)
    weights = fit_weights(rows, outcomes, sample_weights=row_weights, l2=args.l2,
                          signs=None if args.free else FEATURE_SIGNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(weights, indent=2), encoding="utf-8")
    print("fitted weights:")
    for name, value in weights.items():
        print(f"  {name:18} {value:+.3f}   (default {DEFAULT_WEIGHTS.get(name, 0.0):+.2f})")
    results = {"fit": report("fit, in sample", weights, fights),
               "fit_default": report("fit runs, default weights", DEFAULT_WEIGHTS, fights)}
    if args.check:
        checked = load(args.check)
        results["check"] = report("check, held out", weights, checked)
        results["check_default"] = report("check runs, default weights", DEFAULT_WEIGHTS, checked)
    args.out.with_suffix(".report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
