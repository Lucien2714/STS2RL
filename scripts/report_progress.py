"""Summarize a training run's last N optimizer updates from metrics.jsonl.

Reporting on a window of *updates* rather than episodes is the same choice
checkpointing makes: an update is a fixed rollout of transitions, while episode
length grows about fourfold over a run, so an episode window quietly measures
more training late than early.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def _load(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A record half-written when we read is not an error; the next
                # report will see it whole.
                continue
    return records


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=100,
                        help="How many trailing optimizer updates to summarize.")
    args = parser.parse_args()

    records = _load(args.run_dir / "metrics.jsonl")
    updates = [r for r in records if r.get("type") == "ppo_update"]
    episodes = [r for r in records if r.get("type") == "episode"]
    events = [r for r in records if r.get("type") not in ("ppo_update", "episode")]

    if not updates:
        steps = max((r.get("environment_steps", 0) for r in episodes), default=0)
        print(f"no optimizer updates yet | episodes={len(episodes)} steps={steps}")
        return 0

    window = updates[-args.window:]
    total = int(updates[-1].get("optimizer_update", len(updates)))
    step_lo = window[0].get("environment_steps", 0)
    step_hi = window[-1].get("environment_steps", 0)

    # Episodes are attributed to the window by the env step they finished at,
    # so the episode and PPO halves of a report cover the same training.
    in_window = [e for e in episodes if step_lo <= e.get("environment_steps", -1) <= step_hi]
    floors = [e["floor"] for e in in_window if e.get("floor") is not None]
    durations = [e.get("duration_seconds", 0.0) for e in in_window]
    steps_played = sum(e.get("steps", 0) for e in in_window)

    def avg(key: str) -> float | None:
        values = [r[key] for r in window if key in r]
        return mean(values) if values else None

    print(f"updates {total}  (window of {len(window)}, env steps {step_lo}->{step_hi})")
    print(
        "  ppo    loss={}  policy={}  value={}  entropy={}  grad_norm={}  return_scale={}".format(
            _fmt(avg("loss")), _fmt(avg("policy_loss")), _fmt(avg("value_loss")),
            _fmt(avg("entropy")), _fmt(avg("gradient_norm")), _fmt(avg("return_scale"), 2),
        )
    )
    if in_window:
        print(
            "  episodes {}  floor mean={} max={}  reward mean={}  steps mean={}".format(
                len(in_window),
                _fmt(mean(floors) if floors else None, 2),
                max(floors) if floors else "n/a",
                _fmt(mean(e.get("reward", 0.0) for e in in_window), 2),
                _fmt(steps_played / len(in_window), 1),
            )
        )
        truncated = sum(1 for e in in_window if e.get("truncated"))
        reused = sum(1 for e in in_window if e.get("reused_run"))
        errors = sum(e.get("action_errors", 0) for e in in_window)
        wall = sum(durations)
        rate = (steps_played / wall) if wall else 0.0
        # Per lane, not aggregate: durations are summed across concurrent
        # episodes, so this divides played steps by lane-seconds. Multiply by
        # the lane count for machine throughput.
        print(
            "  health   truncated={}/{}  reused_run={}  action_errors={}  "
            "s/step per lane={}".format(
                truncated, len(in_window), reused, errors,
                _fmt(1 / rate, 3) if rate else "n/a",
            )
        )
    recent = Counter(
        r["type"] for r in events if r.get("global_step", 0) >= step_lo
    )
    if recent:
        print("  events   " + "  ".join(f"{k}={v}" for k, v in recent.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
