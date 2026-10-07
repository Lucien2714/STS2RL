"""Compare several sts2rl-eval runs seed by seed.

    uv run python scripts/holdout_compare.py runs/eval2p-holdout n102 p060 p108 p164

Reads DIR/<name>.log (and DIR/<name>-fill.log, episodes replayed after an
interruption) as written by sts2rl-eval's one-line-per-episode output, and
prints per-run totals, per-round means, the seeds that reached the act 3 boss,
and a sign test over the per-seed means for every pair of runs.
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
import sys
from collections import defaultdict


def episodes(directory: str, name: str) -> list[dict]:
    rows: list[dict] = []
    for path in (f"{directory}/{name}.log", f"{directory}/{name}-fill.log"):
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                rows += [json.loads(line[8:]) for line in handle if line.startswith("episode ")]
        except FileNotFoundError:
            pass
    return rows


def bosses(row: dict) -> int:
    """Bosses beaten, recovered from the reward: +10 per boss, +1 per floor, -0.01 per step."""
    return round((row["reward"] - ((row["floor"] or 0) - 1) + 0.01 * row["steps"]) / 10)


def sign_test(diffs: list[float]) -> tuple[int, int, float]:
    higher, lower = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n = higher + lower
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(higher, lower) + 1)) / 2**n) if n else 1.0
    return higher, lower, p


def main() -> None:
    directory, names = sys.argv[1], sys.argv[2:]
    per_seed: dict[str, dict[str, list[int]]] = {}
    print("== per run")
    for name in names:
        rows = episodes(directory, name)
        by_seed: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            by_seed[row["seed"]].append(row["floor"] or 0)
        per_seed[name] = by_seed
        floors = [row["floor"] or 0 for row in rows]
        rounds = [statistics.mean(v[i] for v in by_seed.values() if len(v) > i) for i in range(3) if any(len(v) > i for v in by_seed.values())]
        print(
            f"{name}: n={len(floors)} mean {statistics.mean(floors):.2f} median {statistics.median(floors)} max {max(floors)}"
            f" | reached17 {sum(f >= 17 for f in floors)} passed1 {sum(f >= 18 for f in floors)}"
            f" passed2 {sum(f >= 34 for f in floors)} act3boss {sum(f >= 48 for f in floors)}"
            f" wins {sum(bosses(r) >= 3 for r in rows)} died<9 {sum(f < 9 for f in floors)}"
            f" unterminated {sum(not r['terminated'] for r in rows)} | rounds {[f'{x:.2f}' for x in rounds]}"
        )
        print("   act3 boss seeds:", [(r["seed"], bosses(r)) for r in rows if (r["floor"] or 0) >= 48])
    print("== paired by seed (mean of each seed's episodes)")
    for a, b in itertools.combinations(names, 2):
        seeds = sorted(set(per_seed[a]) & set(per_seed[b]))
        mean_a = [statistics.mean(per_seed[a][s]) for s in seeds]
        mean_b = [statistics.mean(per_seed[b][s]) for s in seeds]
        diffs = [y - x for x, y in zip(mean_a, mean_b)]
        higher, lower, p = sign_test(diffs)
        identical = sum(sorted(per_seed[a][s]) == sorted(per_seed[b][s]) for s in seeds)
        print(
            f"{b} vs {a}: {len(seeds)} seeds, {statistics.mean(mean_a):.2f} -> {statistics.mean(mean_b):.2f}"
            f" ({statistics.mean(diffs):+.2f}), higher {higher} / lower {lower} / tied {len(diffs) - higher - lower},"
            f" p={p:.3f}; identical floor sets on {identical} seeds"
        )


if __name__ == "__main__":
    main()
