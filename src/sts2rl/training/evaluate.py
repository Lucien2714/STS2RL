"""Score a checkpoint on seeds it trained on and on seeds it has never seen.

A fixed seed pool makes "mean floor" comparable across checkpoints, but on its
own it cannot tell a policy that learned to climb from one that memorised
twelve maps.  Only the gap between the two pools can: a policy that generalises
scores about the same on both, while one that memorised falls off a cliff on
the holdout.  That is what `holdout_seeds` are reserved for, and this is the
thing that finally reads them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import statistics
import threading

from sts2rl.agents import CandidatePPOAgent, EpisodeRunner
from sts2rl.env import ResetSpec
from sts2rl.env.mcp_client import STS2ClientError


@dataclass(frozen=True)
class EpisodeScore:
    """One evaluation episode."""

    seed: str
    pool: str
    reward: float
    floor: int | None
    steps: int
    terminated: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "pool": self.pool,
            "reward": self.reward,
            "floor": self.floor,
            "steps": self.steps,
            "terminated": self.terminated,
        }


def final_floor(raw_state: dict) -> int | None:
    """Return the floor a run ended on, or None when it is unreported."""
    run = raw_state.get("run")
    if not isinstance(run, dict):
        return None
    value = run.get("floor")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def build_schedule(
    training_seeds: Sequence[str],
    holdout_seeds: Sequence[str],
    episodes_per_seed: int,
) -> list[tuple[str, str]]:
    """Return the (seed, pool) pairs to play, interleaved across the pools.

    Interleaving matters when a run is cut short: stopping halfway then leaves
    both pools half measured, rather than one of them not measured at all.
    """
    if episodes_per_seed < 1:
        raise ValueError("episodes_per_seed must be positive")
    if not training_seeds and not holdout_seeds:
        raise ValueError("nothing to evaluate: both seed pools are empty")
    overlap = set(training_seeds) & set(holdout_seeds)
    if overlap:
        raise ValueError(
            "a holdout seed was also trained on, so it measures nothing: "
            f"{sorted(overlap)}"
        )
    schedule: list[tuple[str, str]] = []
    for _ in range(episodes_per_seed):
        schedule.extend((seed, "training") for seed in training_seeds)
        schedule.extend((seed, "holdout") for seed in holdout_seeds)
    return schedule


def summarize(scores: Sequence[EpisodeScore]) -> dict[str, dict[str, float]]:
    """Return per-pool means, which is the comparison this exists for."""
    summary: dict[str, dict[str, float]] = {}
    for pool in sorted({score.pool for score in scores}):
        rows = [score for score in scores if score.pool == pool]
        floors = [row.floor for row in rows if row.floor is not None]
        summary[pool] = {
            "episodes": float(len(rows)),
            "reward": statistics.mean(row.reward for row in rows),
            "floor": statistics.mean(floors) if floors else 0.0,
            "steps": statistics.mean(row.steps for row in rows),
        }
    return summary


def format_report(scores: Sequence[EpisodeScore]) -> str:
    """Render per-seed rows, then the comparison that answers the question."""
    if not scores:
        return "no episodes were scored"

    header = f"{'seed':12} {'pool':9} {'n':>3} {'reward':>9} {'floor':>7}"
    lines = [header]
    for pool in ("training", "holdout"):
        for seed in sorted({s.seed for s in scores if s.pool == pool}):
            rows = [s for s in scores if s.seed == seed and s.pool == pool]
            floors = [r.floor for r in rows if r.floor is not None]
            mean_floor = statistics.mean(floors) if floors else 0.0
            mean_reward = statistics.mean(r.reward for r in rows)
            lines.append(
                f"{seed:12} {pool:9} {len(rows):>3} "
                f"{mean_reward:>+9.3f} {mean_floor:>7.2f}"
            )

    summary = summarize(scores)
    lines.append("")
    lines.append(f"{'pool':12} {'episodes':>9} {'reward':>9} {'floor':>8}")
    for pool, stats in summary.items():
        lines.append(
            f"{pool:12} {stats['episodes']:>9.0f} "
            f"{stats['reward']:>+9.3f} {stats['floor']:>8.2f}"
        )

    if "training" in summary and "holdout" in summary:
        gap = summary["training"]["floor"] - summary["holdout"]["floor"]
        lines.append("")
        lines.append(
            f"floor gap (training - holdout): {gap:+.2f}  "
            "-- near zero means it generalises, a large positive gap means it "
            "memorised the training maps"
        )
    return "\n".join(lines)


class Evaluator:
    """Play a fixed schedule of seeds with a frozen policy."""

    def __init__(
        self,
        runners: Sequence[EpisodeRunner],
        agent: CandidatePPOAgent,
        reset: ResetSpec,
        max_episode_failures: int = 3,
    ) -> None:
        runners = tuple(runners)
        if not runners:
            raise ValueError("at least one runner is required")
        if max_episode_failures < 1:
            raise ValueError("max_episode_failures must be at least 1")
        self.runners = runners
        self.agent = agent
        self.reset = reset
        self.max_episode_failures = max_episode_failures

    def _spec_for(self, seed: str) -> ResetSpec:
        """Return the configured reset, seeded, and forced onto custom mode.

        Standard single-player does not accept a seed, so evaluating on one
        would silently score a random map instead.
        """
        return ResetSpec(
            character=self.reset.character,
            game_mode="custom",
            run_seed=seed,
            start_run_option=self.reset.start_run_option,
            allow_active_run=self.reset.allow_active_run,
            ascension=self.reset.ascension,
            modifiers=self.reset.modifiers,
        )

    def run(self, schedule: Sequence[tuple[str, str]]) -> list[EpisodeScore]:
        """Play every (seed, pool) in the schedule and return the scores.

        The agent is switched to deterministic selection and records no
        rollout, so nothing here can change the checkpoint being measured.
        """
        self.agent.eval()
        pending = list(schedule)
        scores: list[EpisodeScore] = []
        lock = threading.Lock()
        failure: list[BaseException] = []

        def worker(runner: EpisodeRunner, lane: int) -> None:
            consecutive = 0
            while True:
                with lock:
                    if failure or not pending:
                        return
                    seed, pool = pending.pop(0)
                try:
                    result = runner.run(self._spec_for(seed))
                except STS2ClientError as exc:
                    self.agent.abort_lane(lane)
                    consecutive += 1
                    with lock:
                        if consecutive >= self.max_episode_failures:
                            failure.append(exc)
                            return
                        # An unplayed seed is not a score: put it back rather
                        # than reporting a pool it never finished.
                        pending.insert(0, (seed, pool))
                    continue
                consecutive = 0
                with lock:
                    scores.append(
                        EpisodeScore(
                            seed=seed,
                            pool=pool,
                            reward=result.total_reward,
                            floor=final_floor(result.final_state),
                            steps=result.steps,
                            terminated=result.terminated,
                        )
                    )

        threads = [
            threading.Thread(
                target=worker,
                args=(runner, getattr(getattr(runner, "agent", None), "lane", index)),
                daemon=True,
            )
            for index, runner in enumerate(self.runners)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if failure:
            raise failure[0]
        return scores
