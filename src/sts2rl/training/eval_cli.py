"""Command line for scoring a checkpoint against held-out seeds."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import ExitStack
import json
from pathlib import Path

from sts2rl.agents import CandidatePPOAgent, EpisodeRunner
from sts2rl.encoder import GameEncoder, GameTokenizer, GameVocabulary
from sts2rl.env import DEFAULT_ACTION_DELAY_SECONDS, GameEnv
from sts2rl.env.game_env import BACKENDS
from sts2rl.search.shadow import ShadowEnv
from sts2rl.training.checkpoint import CheckpointManager
from sts2rl.training.evaluate import (
    Evaluator,
    build_schedule,
    format_report,
    summarize,
)


def create_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI without touching the game or filesystem."""
    parser = argparse.ArgumentParser(
        description=(
            "Score a checkpoint on the seeds it trained on and on the seeds "
            "reserved from it, and report the gap between them."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="Checkpoint name inside the run, or 'latest'.",
    )
    parser.add_argument(
        "--episodes-per-seed",
        type=int,
        default=3,
        help="How many episodes to play on each seed of each pool.",
    )
    parser.add_argument(
        "--ports",
        help="Comma-separated STS2MCP ports played in parallel.",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        help=(
            "environment to score against; defaults to the one the run trained "
            "on.  Naming the other one measures transfer -- a simulator-trained "
            "checkpoint played by real game clients -- which is a different "
            "question from how well the run learned its own environment."
        ),
    )
    parser.add_argument("--base-url")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--action-delay", type=float)
    parser.add_argument("--device")
    parser.add_argument(
        "--pools",
        default="both",
        choices=("both", "training", "holdout"),
        help="Which seed pools to play. 'both' is what makes the gap readable.",
    )
    parser.add_argument(
        "--holdout-seeds",
        help=(
            "Comma-separated holdout seeds, or 'default' for the bundled set, "
            "used in place of the ones the checkpoint records.  A run trained "
            "without --holdout-seeds records none.  A seed the checkpoint "
            "trained on is refused."
        ),
    )
    parser.add_argument(
        "--only-seeds",
        help=(
            "Comma-separated seeds: play only these, from whichever pool holds "
            "them.  A seed in neither pool is refused."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the per-episode scores and the summary here as JSON.",
    )
    parser.add_argument(
        "--search-combat",
        action="store_true",
        help=(
            "play every fight with the combat search instead of the actor (needs "
            "--backend sim, or --shadow-sims beside real clients), so two macro "
            "policies are compared under the same combat play"
        ),
    )
    parser.add_argument(
        "--shadow-sims",
        help=(
            "Comma-separated simulator base URLs, one per --ports client: each "
            "follows its real client's run so its fights can be searched "
            "(--backend game with --search-combat).  A fight the copy no longer "
            "matches is played by the actor."
        ),
    )
    parser.add_argument("--search-simulations", type=int, default=50)
    parser.add_argument("--search-elite-simulations", type=int, help="elite fights (default: --search-simulations)")
    parser.add_argument("--search-boss-simulations", type=int, help="boss fights (default: --search-simulations)")
    parser.add_argument("--search-depth", type=int, default=2)
    parser.add_argument("--search-weights", help="leaf evaluator weights (JSON); default: the step 1-b fit")
    return parser


def _lane_agent(
    agent: CandidatePPOAgent,
    lane: int,
    env: GameEnv,
    args: argparse.Namespace,
    gate=None,
):
    """The lane's agent: the actor alone, or with the combat search playing fights.

    ``env`` is the simulator the search branches on; beside a real client it is
    the shadow copy, and ``gate`` is the copy's agreement check.
    """
    view = agent.lane_view(lane)
    if not args.search_combat:
        return view
    import random

    from sts2rl.search import CombatSearch, LeafEvaluator, MctsConfig, SearchCombatAgent
    from sts2rl.training.cli import DEFAULT_SEARCH_WEIGHTS

    weights = Path(args.search_weights) if args.search_weights else DEFAULT_SEARCH_WEIGHTS
    search = CombatSearch(
        MctsConfig(simulations=args.search_simulations, turn_depth=args.search_depth),
        evaluator=LeafEvaluator(json.loads(weights.read_text(encoding="utf-8"))),
        # Fixed per lane, so two policies evaluated alike meet the same search luck.
        rng=random.Random(1000 + lane),
    )
    rooms = {"elite": args.search_elite_simulations, "boss": args.search_boss_simulations}
    return SearchCombatAgent(
        view, env, search, run_label=f"eval-lane{lane}",
        room_simulations={room: n for room, n in rooms.items() if n is not None},
        gate=gate,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, evaluate, and return a process exit status."""
    args = create_parser().parse_args(argv)
    try:
        return run_evaluation(args)
    except KeyboardInterrupt:
        return 130


def run_evaluation(args: argparse.Namespace) -> int:
    """Load a checkpoint, play the schedule, and report the comparison."""
    vocabulary = GameVocabulary.from_bundled_data()
    manager = CheckpointManager(args.run_dir, vocabulary)
    loaded = manager.load(args.checkpoint, map_location="cpu")
    plan = loaded.plan

    training_seeds, holdout_seeds = _seed_pools(args, plan.training)
    if not training_seeds and not holdout_seeds:
        raise ValueError(
            "the checkpoint records no seed pool, so there is nothing "
            "comparable to evaluate; train with --seed-pool"
        )

    schedule = build_schedule(
        training_seeds, holdout_seeds, args.episodes_per_seed
    )

    device = args.device or plan.training.device
    agent = CandidatePPOAgent(
        tokenizer=GameTokenizer(vocabulary),
        game_encoder=GameEncoder(vocabulary, plan.encoder),
        config=plan.ppo,
        device=device,
    )
    manager.restore_agent(loaded, agent, plan)

    base_urls = _client_base_urls(args, plan)
    timeout = args.timeout if args.timeout is not None else plan.training.timeout
    backend = args.backend or plan.training.backend
    if args.action_delay is not None:
        delay = args.action_delay
    elif backend == plan.training.backend:
        delay = plan.training.action_delay_seconds
    else:
        # The recorded delay was chosen for the other environment.  A simulator
        # run records 0, which against a real client is a diagnostic setting,
        # not a speedup: actions land while the game is still resolving.
        delay = 0.0 if backend == "sim" else DEFAULT_ACTION_DELAY_SECONDS

    shadow_urls = _shadow_urls(args, backend, len(base_urls))
    if args.search_combat and backend != "sim" and not shadow_urls:
        raise ValueError(
            "--search-combat needs the simulator backend, or --shadow-sims "
            "beside real clients: only a simulator can branch"
        )
    with ExitStack() as clients:
        env_for = [
            clients.enter_context(
                GameEnv(
                    base_url=base_url,
                    timeout=timeout,
                    action_delay_seconds=delay,
                    # The run records which environment it was trained
                    # against, and the two do not start a run the same way:
                    # evaluating a simulator run through the game's menus
                    # fails at the first reset.
                    backend=backend,
                )
            )
            for base_url in base_urls
        ]
        shadows = [
            ShadowEnv(
                env_for[lane],
                clients.enter_context(
                    GameEnv(base_url=url, timeout=timeout, action_delay_seconds=0.0, backend="sim")
                ),
            )
            for lane, url in enumerate(shadow_urls)
        ]
        runners = [
            EpisodeRunner(
                shadows[lane] if shadows else env_for[lane],
                _lane_agent(
                    agent,
                    lane,
                    shadows[lane].sim if shadows else env_for[lane],
                    args,
                    gate=shadows[lane].agrees if shadows else None,
                ),
                max_steps=plan.training.max_steps_per_episode,
                max_state_refreshes=plan.training.max_state_refreshes,
            )
            for lane in range(len(base_urls))
        ]
        evaluator = Evaluator(
            runners,
            agent,
            plan.reset,
            max_episode_failures=plan.training.max_episode_failures,
            on_score=lambda score: print(
                "episode " + json.dumps(score.to_dict(), sort_keys=True), flush=True
            ),
        )
        scores = evaluator.run(schedule)

    print(format_report(scores))
    following = [
        {"lane": lane, "episodes": shadow.finish()} for lane, shadow in enumerate(shadows)
    ]
    for lane in following:
        for number, record in enumerate(lane["episodes"], start=1):
            print(f"shadow lane {lane['lane']} episode {number}: " + json.dumps(record, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "run_dir": str(args.run_dir),
                    "checkpoint": args.checkpoint,
                    "episodes_per_seed": args.episodes_per_seed,
                    "episodes": [score.to_dict() for score in scores],
                    "summary": summarize(scores),
                    **({"shadow": following} if following else {}),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.output}")
    return 0


def _seed_pools(args: argparse.Namespace, training: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the (training, holdout) seeds to play, after --holdout-seeds and --pools."""
    from sts2rl.training.cli import _seed_list
    from sts2rl.training.config import DEFAULT_HOLDOUT_SEEDS

    training_seeds = training.training_seeds  # type: ignore[attr-defined]
    holdout_seeds = training.holdout_seeds  # type: ignore[attr-defined]
    if args.holdout_seeds is not None:
        holdout_seeds = _seed_list(args.holdout_seeds, DEFAULT_HOLDOUT_SEEDS)
        # Checked against the recorded pool whatever --pools says: a seed the
        # checkpoint trained on measures nothing as a holdout.
        overlap = set(holdout_seeds) & set(training_seeds)
        if overlap:
            raise ValueError(
                f"the checkpoint trained on these holdout seeds: {sorted(overlap)}"
            )
    if args.pools == "training":
        holdout_seeds = ()
    elif args.pools == "holdout":
        training_seeds = ()
    if args.only_seeds:
        wanted = tuple(part.strip() for part in str(args.only_seeds).split(",") if part.strip())
        unknown = set(wanted) - set(training_seeds) - set(holdout_seeds)
        if unknown:
            raise ValueError(f"not in the pools being played: {sorted(unknown)}")
        training_seeds = tuple(seed for seed in wanted if seed in training_seeds)
        holdout_seeds = tuple(seed for seed in wanted if seed in holdout_seeds)
    return training_seeds, holdout_seeds


def _shadow_urls(args: argparse.Namespace, backend: str, clients: int) -> tuple[str, ...]:
    """Return one simulator URL per real client, or none without --shadow-sims."""
    if not args.shadow_sims:
        return ()
    urls = tuple(part.strip() for part in str(args.shadow_sims).split(",") if part.strip())
    if backend != "game":
        raise ValueError("--shadow-sims follows real clients: it needs --backend game")
    if not args.search_combat:
        raise ValueError("--shadow-sims is only read by the search: add --search-combat")
    if len(urls) != clients:
        raise ValueError(f"--shadow-sims needs one simulator per client: {len(urls)} for {clients}")
    if len(set(urls)) != len(urls):
        raise ValueError("--shadow-sims repeats a simulator: two runs cannot share one")
    return urls


def _client_base_urls(args: argparse.Namespace, plan: object) -> tuple[str, ...]:
    """Return one base URL per client, preferring the flags over the plan."""
    training = plan.training  # type: ignore[attr-defined]
    if args.ports:
        ports = tuple(
            int(part.strip()) for part in str(args.ports).split(",") if part.strip()
        )
        from dataclasses import replace

        training = replace(training, ports=ports)
    if args.base_url:
        from dataclasses import replace

        training = replace(training, base_url=args.base_url)
    return training.client_base_urls()


if __name__ == "__main__":
    raise SystemExit(main())
