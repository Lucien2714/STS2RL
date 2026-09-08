"""Summarize a training run: is the score improving, or is it seed luck?

Reads ``metrics.jsonl`` and reports the trend, the per-seed breakdown, and the
PPO diagnostics -- the three things that separate learning from noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


def load(run_dir: Path) -> tuple[list[dict], list[dict]]:
    episodes: list[dict] = []
    updates: list[dict] = []
    with (run_dir / "metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("type") == "episode":
                episodes.append(record)
            elif record.get("type") == "ppo_update":
                updates.append(record)
    return episodes, updates


def _block(rows: list[dict], key: str) -> str:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return "n/a"
    return f"{mean(values):+.3f}"


def report(run_dir: Path, buckets: int) -> None:
    episodes, updates = load(run_dir)
    if not episodes:
        print(f"{run_dir}: no episodes recorded yet")
        return

    print(f"=== {run_dir}  ({len(episodes)} episodes, {len(updates)} updates)")
    reused = sum(1 for row in episodes if row.get("reused_run"))
    errors = sum(row.get("action_errors", 0) for row in episodes)
    print(f"reused runs: {reused}   action errors: {errors}")

    # Throughput and truncation, which is what comparing two backends -- a
    # windowed client against a headless one, or the game against a simulator
    # -- actually turns on.  Truncation is reported beside it rather than
    # buried: a truncated episode leaves the run alive, so the next reset joins
    # it instead of starting its assigned seed, and a seeded experiment stops
    # measuring what it claims to.  A faster backend that truncates more is not
    # faster.
    total_steps = sum(row.get("steps", 0) for row in episodes)
    total_seconds = sum(row.get("duration_seconds", 0.0) for row in episodes)
    truncated = sum(1 for row in episodes if row.get("truncated"))
    per_step = f"{total_seconds / total_steps:.3f}" if total_steps else "n/a"
    print(
        f"throughput: {per_step} s/step "
        f"({total_steps} steps in {total_seconds:.0f}s)   "
        f"truncated: {truncated}/{len(episodes)} "
        f"({100 * truncated / len(episodes):.1f}%)"
    )

    size = max(1, len(episodes) // buckets)
    print(f"\n-- reward and floor by block of {size} episodes")
    print(f"{'episodes':>14}  {'reward':>8}  {'floor':>7}  {'steps':>7}")
    for start in range(0, len(episodes), size):
        rows = episodes[start : start + size]
        print(
            f"{start + 1:>6}-{start + len(rows):<7} "
            f"{_block(rows, 'reward'):>8}  "
            f"{_block(rows, 'floor'):>7}  "
            f"{_block(rows, 'steps'):>7}"
        )

    per_seed: dict[str, list[float]] = {}
    for row in episodes:
        seed = row.get("seed")
        if seed:
            per_seed.setdefault(seed, []).append(row["reward"])
    if per_seed:
        print("\n-- mean reward per seed (a fixed pool makes these comparable)")
        for seed, rewards in sorted(per_seed.items(), key=lambda kv: -mean(kv[1])):
            first = mean(rewards[: max(1, len(rewards) // 2)])
            last = mean(rewards[len(rewards) // 2 :])
            print(
                f"  {seed}  n={len(rewards):<3} mean={mean(rewards):+.3f} "
                f"first-half={first:+.3f} second-half={last:+.3f}"
            )

    rewards = [row["reward"] for row in episodes]
    print(
        f"\noverall reward: mean={mean(rewards):+.3f} "
        f"sd={pstdev(rewards):.3f} best={max(rewards):+.3f}"
    )
    floors = [row["floor"] for row in episodes if row.get("floor") is not None]
    if floors:
        print(f"floor: mean={mean(floors):.2f} best={max(floors)}")

    if updates:
        print("\n-- ppo diagnostics (first and last update)")
        for label, row in (("first", updates[0]), ("last", updates[-1])):
            print(
                f"  {label:>5}: value_loss={row['value_loss']:8.3f} "
                f"policy_loss={row['policy_loss']:+.4f} "
                f"entropy={row['entropy']:.3f} "
                f"grad={row['gradient_norm']:7.3f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--buckets", type=int, default=10)
    args = parser.parse_args()
    report(args.run_dir, args.buckets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
