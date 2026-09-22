"""Build a simulator snapshot library by playing a trained policy into a later act.

The simulator writes a snapshot the moment a run it was asked to capture enters
a later act (``sts2sim --snapshots DIR``, reset with ``capture``).  This script
supplies the runs: it plays every seed of the given pools with a checkpoint's
policy, once deterministically, then by sampling until the seed reaches the act
or the attempts run out.

The policy is frozen throughout.  Sampling uses the agent's training mode, which
collects a rollout, so the rollout is made too large to ever fill: an update
here would move the weights between seeds and the library would come from a
different policy for every seed.

Usage:
    uv run python scripts/capture_snapshots.py --run-dir runs/mc5k \
        --library ../STS2RL/data/snapshots/mc5k --ports 15600,15601,15602,15603
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import queue
import threading

from sts2rl.agents import CandidatePPOAgent, EpisodeRunner
from sts2rl.encoder.game_encoder import GameEncoder
from sts2rl.encoder.game_tokenizer import GameTokenizer
from sts2rl.encoder.vocabulary import GameVocabulary
from sts2rl.env import ResetSpec
from sts2rl.env.game_env import GameEnv
from sts2rl.env.mcp_client import GameCharacter, STS2ClientError
from sts2rl.training.checkpoint import CheckpointManager


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--library", type=Path, required=True,
                        help="the directory every sts2sim was started with --snapshots")
    parser.add_argument("--ports", required=True)
    parser.add_argument("--act", type=int, default=2)
    parser.add_argument("--boss", action="store_true",
                        help="capture the act's boss fight rather than the act's start")
    parser.add_argument("--attempts", type=int, default=6,
                        help="sampled attempts per seed after the deterministic one")
    args = parser.parse_args()

    vocabulary = GameVocabulary.from_bundled_data()
    manager = CheckpointManager(args.run_dir, vocabulary)
    loaded = manager.load(args.checkpoint, map_location="cpu")
    plan = loaded.plan
    agent = CandidatePPOAgent(
        tokenizer=GameTokenizer(vocabulary),
        game_encoder=GameEncoder(vocabulary, plan.encoder),
        # Never fills, so sampling never updates: see the module docstring.
        config=replace(plan.ppo, rollout_size=10**9),
        device="cpu",
    )
    manager.restore_agent(loaded, agent, plan)

    character = GameCharacter.get(plan.reset.character)
    target = f"act{args.act}_boss" if args.boss else f"act{args.act}"
    seeds = list(plan.training.training_seeds) + list(plan.training.holdout_seeds)
    ports = [int(p) for p in args.ports.split(",")]

    def captured(seed: str) -> bool:
        return (args.library / character / seed / f"{target}.json").exists()

    def play(pending: list[str], deterministic: bool) -> None:
        agent.train(not deterministic)
        work: queue.Queue[str] = queue.Queue()
        for seed in pending:
            work.put(seed)
        lock = threading.Lock()

        def worker(lane: int, port: int) -> None:
            with ExitStack() as stack:
                env = stack.enter_context(
                    GameEnv(base_url=f"http://localhost:{port}/api/v1", backend="sim",
                            action_delay_seconds=0.0)
                )
                runner = EpisodeRunner(
                    env, agent.lane_view(lane),
                    max_steps=plan.training.max_steps_per_episode,
                    max_state_refreshes=plan.training.max_state_refreshes,
                )
                while True:
                    try:
                        seed = work.get_nowait()
                    except queue.Empty:
                        return
                    spec = ResetSpec(
                        character=plan.reset.character, game_mode="custom",
                        run_seed=seed, sim_capture=True,
                    )
                    try:
                        result = runner.run(spec)
                        floor = result.final_state.get("run", {}).get("floor")
                    except STS2ClientError as exc:
                        agent.abort_lane(lane)
                        floor = f"client error: {exc}"
                    with lock:
                        mark = "captured" if captured(seed) else "-"
                        print(f"  {seed} floor={floor} {mark}", flush=True)

        threads = [threading.Thread(target=worker, args=(lane, port))
                   for lane, port in enumerate(ports)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    print(f"{len(seeds)} seeds, target {target}, library {args.library}")
    print("deterministic pass:")
    play([s for s in seeds if not captured(s)], deterministic=True)
    for attempt in range(1, args.attempts + 1):
        missing = [s for s in seeds if not captured(s)]
        if not missing:
            break
        print(f"sampled attempt {attempt}: {len(missing)} seeds still short of {target}")
        play(missing, deterministic=False)

    train = [s for s in plan.training.training_seeds if captured(s)]
    hold = [s for s in plan.training.holdout_seeds if captured(s)]
    print(f"\ncaptured: training {len(train)}/{len(plan.training.training_seeds)}, "
          f"holdout {len(hold)}/{len(plan.training.holdout_seeds)}")
    (args.library / f"{target}_training_seeds.txt").write_text(",".join(train))
    (args.library / f"{target}_holdout_seeds.txt").write_text(",".join(hold))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
