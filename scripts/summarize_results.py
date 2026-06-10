"""Summarize checkpoint evaluation CSV output."""

import argparse
import csv
from pathlib import Path


DEFAULT_INPUT = Path("result.csv")


def parse_float(row: dict, key: str, default: float = 0.0) -> float:
    """Parse a float from a CSV row with a default fallback."""
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def parse_int(row: dict, key: str, default: int = 0) -> int:
    """Parse an int from a CSV row with a default fallback."""
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def parse_steps(value: str) -> list[int]:
    """Parse comma-separated episode-ending step counts."""
    steps = []
    for item in (value or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            steps.append(int(item))
        except ValueError:
            pass
    return steps


def load_rows(path: Path) -> list[dict]:
    """Load evaluation CSV rows."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def score_row(row: dict) -> float:
    """Compute a simple ranking score for checkpoint summaries."""
    return (
        parse_float(row, "avg_battle_reward")
        + parse_float(row, "avg_reward") * 0.25
        + parse_float(row, "battle_win_rate") * 100.0
        + parse_float(row, "avg_floor") * 10.0
        - parse_float(row, "timeouts") * 50.0
    )


def describe_row(row: dict, rank: int | None = None) -> str:
    """Format one checkpoint summary row for human-readable output."""
    prefix = f"{rank}. " if rank is not None else ""
    checkpoint = row.get("checkpoint", "unknown")
    steps = parse_steps(row.get("ending_steps", ""))
    step_text = (
        f"episodes ended at steps {', '.join(str(step) for step in steps)}"
        if steps
        else "ending steps unavailable"
    )
    return (
        f"{prefix}{checkpoint}: avg reward {parse_float(row, 'avg_reward'):.1f}, "
        f"battle reward {parse_float(row, 'avg_battle_reward'):.1f}, "
        f"win rate {parse_float(row, 'battle_win_rate'):.1%} "
        f"({parse_int(row, 'battle_wins')} wins / {parse_int(row, 'battle_losses')} losses), "
        f"avg floor {parse_float(row, 'avg_floor'):.1f}, max floor {parse_int(row, 'max_floor')}; "
        f"{step_text}."
    )


def summarize(rows: list[dict], top: int) -> str:
    """Build a readable summary of evaluation rows."""
    if not rows:
        return "No rows found."

    ranked = sorted(rows, key=score_row, reverse=True)
    best = ranked[0]
    worst = ranked[-1]
    total_episodes = sum(parse_int(row, "episodes") for row in rows)
    total_wins = sum(parse_int(row, "battle_wins") for row in rows)
    total_losses = sum(parse_int(row, "battle_losses") for row in rows)
    avg_reward = sum(parse_float(row, "avg_reward") for row in rows) / len(rows)
    avg_floor = sum(parse_float(row, "avg_floor") for row in rows) / len(rows)

    lines = [
        "Evaluation Summary",
        "",
        f"Read {len(rows)} checkpoints covering {total_episodes} episodes.",
        f"Overall battle record: {total_wins} wins / {total_losses} losses.",
        f"Average checkpoint reward: {avg_reward:.1f}; average checkpoint floor: {avg_floor:.1f}.",
        "",
        "Best checkpoint:",
        describe_row(best),
        "",
        "Weakest checkpoint:",
        describe_row(worst),
        "",
        f"Top {min(top, len(ranked))}:",
    ]

    for rank, row in enumerate(ranked[:top], start=1):
        lines.append(describe_row(row, rank))

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse summary CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize evaluate_checkpoints.py CSV output in a readable form."
    )
    parser.add_argument("csv_path", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--top", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    """Run the CSV summary CLI."""
    args = parse_args()
    rows = load_rows(args.csv_path)
    print(summarize(rows, max(1, args.top)))


if __name__ == "__main__":
    main()
