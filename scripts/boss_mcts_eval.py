"""Play act-1 boss fights from snapshots, by search or by a trained actor.

Step 1 of docs/mcts: does a search that plays the fight well win from these decks?
Each job is one boss snapshot in one world (``reseed`` at reset, so the shuffles and
rolls differ while the deck, relics and HP stay the snapshot's), played to the end
of the fight, never beyond. One worker per simulator port.

    # simulators: sts2sim --port 1560N --snapshots ../STS2RL/data/snapshots/mc5k
    uv run python scripts/boss_mcts_eval.py --policy mcts --simulations 200 \\
        --ports 15600,15601 --out runs/boss-mcts/mcts200.jsonl

``--policy actor`` plays a checkpoint's actor instead (argmax, or ``--sample``). It
imports nothing from ``sts2rl.search``, so it also runs under a checkout from before
that package, which is how a checkpoint trained against an older vocabulary is played.

``--record-features`` writes, for every searched decision, the evaluator's features of
the state, labelled afterwards with whether the fight was won: the data
``sts2rl.search.fit_weights`` fits the leaf evaluator to. The raw states go beside it
in ``<out>.states.jsonl.gz``, so a changed feature set is refitted without replaying.
"""

from __future__ import annotations

import argparse
import gzip
import json
import queue
import random
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sts2rl.env.game_env import GameEnv
from sts2rl.env.state import extract_raw_state

SNAPSHOTS = Path(__file__).resolve().parents[1] / "data/snapshots/mc5k"
COMBAT = frozenset({"monster", "elite", "boss", "hand_select"})


def fight_over(state: Mapping[str, Any]) -> bool:
    """Mirror of sts2rl.search.evaluate.is_fight_over, for checkouts without that package."""
    if state.get("state_type") == "game_over":
        return True
    player = state.get("player") if isinstance(state.get("player"), Mapping) else {}
    return state.get("state_type") not in COMBAT and "energy" not in player


def lost(state: Mapping[str, Any]) -> bool:
    if state.get("state_type") == "game_over":
        return state.get("terminal_reason") != "victory"
    player = state.get("player") if isinstance(state.get("player"), Mapping) else {}
    return float(player.get("hp") or 0) <= 0


def world_seed(snapshot: str, world: int) -> int:
    """A reseed value per (snapshot, world), the same in every run of this script."""
    return random.Random(f"{snapshot}:{world}").getrandbits(32)


def make_mcts_player(args):
    from sts2rl.search import CombatSearch, LeafEvaluator, MctsConfig, play_fight

    weights = json.loads(Path(args.weights).read_text(encoding="utf-8")) if args.weights else None
    evaluator = LeafEvaluator(weights) if weights else LeafEvaluator()
    config = MctsConfig(simulations=args.simulations, turn_depth=args.turn_depth, seconds=args.seconds,
                        reseed=not args.clairvoyant)

    def play(env: GameEnv, state, job_seed: int):
        search = CombatSearch(config, evaluator=evaluator, rng=random.Random(job_seed))
        result = play_fight(env, search, state, keep_decisions=True)
        states = [d.state for d in result.decisions] if args.record_features else []
        seconds = [d.seconds for d in result.decisions]
        simulations = [d.result.root.visits for d in result.decisions]
        return result.final_state, result.last_combat_state, result.steps, seconds, states, simulations

    return play


def make_actor_player(args):
    from dataclasses import replace

    from sts2rl.agents.action_space import LegalActionProvider, NoLegalActionsError
    from sts2rl.agents.ppo import CandidatePPOAgent
    from sts2rl.encoder.game_encoder import GameEncoder
    from sts2rl.encoder.game_tokenizer import GameTokenizer
    from sts2rl.encoder.vocabulary import GameVocabulary
    from sts2rl.env.types import GameObservation
    from sts2rl.training.checkpoint import CheckpointManager

    vocabulary = GameVocabulary.from_bundled_data()
    if args.bc_artifact:
        # A cloned actor (sts2rl-bc-train): only encoder weights, no PPO state.
        from sts2rl.encoder import EncoderConfig
        from sts2rl.training.bc import load_bc_encoder_state

        encoder = GameEncoder(vocabulary, EncoderConfig())
        encoder.load_state_dict(
            load_bc_encoder_state(args.bc_artifact, vocabulary=vocabulary, encoder_config=EncoderConfig())
        )
        agent = CandidatePPOAgent(tokenizer=GameTokenizer(vocabulary), game_encoder=encoder, device="cpu")
    else:
        manager = CheckpointManager(Path(args.run_dir), vocabulary)
        loaded = manager.load(args.checkpoint, map_location="cpu")
        agent = CandidatePPOAgent(
            tokenizer=GameTokenizer(vocabulary),
            game_encoder=GameEncoder(vocabulary, loaded.plan.encoder),
            # Played, never updated: a run's reference policy is not needed here.
            config=replace(loaded.plan.ppo, reference_kl_coefficient=0.0),
            device="cpu",
        )
        manager.restore_agent(loaded, agent, loaded.plan)
    agent.train(args.sample)
    lock = threading.Lock()
    provider = LegalActionProvider()

    def play(env: GameEnv, state, job_seed: int):
        steps = 0
        torch_seed = job_seed
        last_combat = state
        while steps < 600 and not fight_over(state):
            last_combat = state
            try:
                provider.require_candidates(state)
            except NoLegalActionsError:
                break
            observation = GameObservation(raw_state=state, player_detail=env.get_player_detail())
            with lock:
                import torch

                torch.manual_seed(torch_seed + steps)
                action = agent.choose_action(observation, lane=0)
                agent.discard_decision(lane=0)
            state = env.step(action).raw_state
            steps += 1
        return state, last_combat, steps, [], [], []

    return play


def states_path(out: Path) -> Path:
    return out.with_name(out.stem + ".states.jsonl.gz")


def boss_health(state: Mapping[str, Any]) -> float | None:
    battle = state.get("battle") if isinstance(state.get("battle"), Mapping) else {}
    enemies = [e for e in battle.get("enemies") or [] if isinstance(e, Mapping)]
    total = sum(float(e.get("max_hp") or 0) for e in enemies)
    return sum(max(float(e.get("hp") or 0), 0.0) for e in enemies) / total if total else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=("mcts", "actor"), required=True)
    parser.add_argument("--ports", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pools", default="training,holdout")
    parser.add_argument("--count", type=int, default=0, help="snapshots per pool (0 = all)")
    parser.add_argument("--worlds", type=int, default=3, help="reseeded worlds per snapshot")
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--turn-depth", type=int, default=2)
    parser.add_argument("--seconds", type=float, help="mcts: wall-clock budget per decision")
    parser.add_argument("--weights", help="leaf evaluator weights (JSON)")
    parser.add_argument("--clairvoyant", action="store_true",
                        help="mcts: search the real hidden state instead of reseeded worlds (an upper bound)")
    parser.add_argument("--record-features", action="store_true")
    parser.add_argument("--run-dir", help="actor: the run whose checkpoint plays")
    parser.add_argument("--bc-artifact", help="actor: a cloned actor (bc_best.pt) instead of a run's checkpoint")
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--sample", action="store_true", help="actor: sample instead of argmax")
    parser.add_argument("--only", type=Path, help="play only the snapshots listed in this file, one per line")
    args = parser.parse_args()
    only = set(args.only.read_text(encoding="utf-8").split()) if args.only else None

    jobs: queue.Queue = queue.Queue()
    for pool in args.pools.split(","):
        seeds = (SNAPSHOTS / f"act1_boss_{pool}_seeds.txt").read_text(encoding="utf-8").strip().split(",")
        for seed in seeds[: args.count or None]:
            if only is not None and seed not in only:
                continue
            for world in range(args.worlds):
                jobs.put((pool, seed, world))
    done = set()
    if args.out.exists():
        # Resume: a fight already written is not played again.
        for line in args.out.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done.add((record["pool"], record["snapshot"], record["world"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = jobs.qsize()
    play = make_mcts_player(args) if args.policy == "mcts" else make_actor_player(args)
    write_lock = threading.Lock()
    counts = {"won": 0, "played": 0}

    def worker(port: int) -> None:
        with GameEnv(base_url=f"http://localhost:{port}/api/v1", action_delay_seconds=0.0, backend="sim") as env:
            while True:
                try:
                    pool, seed, world = jobs.get_nowait()
                except queue.Empty:
                    return
                if (pool, seed, world) in done:
                    continue
                reseed = world_seed(seed, world)
                started = time.perf_counter()
                state = extract_raw_state(env.client.sim_reset("IRONCLAD", seed, start_act=1, start_boss=True, reseed=reseed))
                final, last_combat, steps, seconds, states, simulations = play(env, state, reseed)
                won = fight_over(final) and not lost(final)
                player = final.get("player") if isinstance(final.get("player"), Mapping) else {}
                record = {
                    "pool": pool, "snapshot": seed, "world": world, "reseed": reseed,
                    "policy": args.policy, "simulations": args.simulations if args.policy == "mcts" else None,
                    "turn_depth": args.turn_depth if args.policy == "mcts" else None,
                    "seconds_budget": args.seconds if args.policy == "mcts" else None,
                    "clairvoyant": bool(args.policy == "mcts" and args.clairvoyant),
                    "won": won, "steps": steps, "final_state": final.get("state_type"),
                    "hp": player.get("hp"), "max_hp": player.get("max_hp"),
                    "boss_hp_left": None if won else boss_health(last_combat),
                    "decision_seconds": seconds, "seconds": time.perf_counter() - started,
                    "decision_simulations": simulations,
                }
                if states:
                    from sts2rl.search.evaluate import features

                    record["features"] = [features(s) for s in states]
                with write_lock:
                    with args.out.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                    if states:
                        # One gzip member per fight: concatenated members read back as one stream.
                        with gzip.open(states_path(args.out), "at", encoding="utf-8") as handle:
                            handle.write(json.dumps({"pool": pool, "snapshot": seed, "world": world,
                                                     "won": won, "states": states}) + "\n")
                    counts["played"] += 1
                    counts["won"] += won
                    print(f"[{counts['played'] + len(done)}/{total}] {pool:8} {seed} w{world} "
                          f"{'WON ' if won else 'lost'} steps {steps:3} {record['seconds']:6.1f}s "
                          f"(won {counts['won']}/{counts['played']})", flush=True)

    threads = [threading.Thread(target=worker, args=(int(p),)) for p in args.ports.split(",")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
