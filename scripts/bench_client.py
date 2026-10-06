"""Time the HTTP client on MCTS-style cycles against one running STS2Simulator.

Start a simulator first, for example on port 15650, then run::

    uv run python scripts/bench_client.py --port 15650

Every variant plays the same work in the same process: reset a run on a seed
with the trainer's own ``GameEnv(backend="sim")`` reset, walk to the first
fight, hold a branch point, and repeat cycles of restore, reseed and a few
combat steps.  Variants are interleaved round by round, so load from other
processes falls on all of them alike, and the report gives medians over rounds.

Variants:

* ``old``  -- ``STS2Client`` over a stock ``requests.Session`` (``trust_env``
  on), which is what the client created before it stopped reading the
  environment.
* ``new``  -- ``STS2Client`` with its own session (``trust_env`` off).
* ``urllib3`` -- ``STS2Client`` over a minimal session that calls a urllib3
  connection pool directly, skipping requests.  Measurement only: it is the
  ceiling for dropping requests, not something the client ships.

CPU time is ``time.process_time`` of this process; on Windows it ticks in
15.6 ms steps, so each round runs enough requests to keep that under a few
percent.  ``--json-probe`` also times parsing one combat response with the
stdlib and, when importable, ``orjson`` (``uv run --with orjson ...``).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from typing import Any
from urllib.parse import urlencode, urlsplit

import requests
import urllib3

from sts2rl.agents.action_space import LegalActionProvider, NoLegalActionsError
from sts2rl.env.game_env import GameEnv
from sts2rl.env.mcp_client import STS2Client
from sts2rl.env.reset import ResetSpec
from sts2rl.search.evaluate import is_fight_over
from sts2rl.search.sim_env import BRANCHABLE, SimulatorSearchEnv

DEFAULT_SEED = "0357NDTC82"


class _Urllib3Response:
    def __init__(self, raw: urllib3.BaseHTTPResponse) -> None:
        self.status_code = raw.status
        self.content = raw.data

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.content)


class Urllib3Session:
    """Just enough of ``requests.Session`` for ``STS2Client``, over urllib3."""

    def __init__(self, base_url: str) -> None:
        parts = urlsplit(base_url)
        self.pool = urllib3.HTTPConnectionPool(
            parts.hostname, parts.port, maxsize=1, retries=False
        )

    def request(self, *, method, url, params, json, timeout):  # noqa: A002
        path = urlsplit(url).path
        if params:
            path = f"{path}?{urlencode(params)}"
        body = None
        headers = None
        if json is not None:
            body = _dumps(json)
            headers = {"Content-Type": "application/json"}
        raw = self.pool.urlopen(
            method, path, body=body, headers=headers, timeout=timeout, retries=False
        )
        return _Urllib3Response(raw)

    def close(self) -> None:
        self.pool.close()


def _dumps(value: Any) -> bytes:
    return json.dumps(value).encode()


def make_client(variant: str, base_url: str) -> STS2Client:
    if variant == "old":
        return STS2Client(base_url=base_url, session=requests.Session(), action_delay_seconds=0)
    if variant == "new":
        return STS2Client(base_url=base_url, action_delay_seconds=0)
    if variant == "urllib3":
        return STS2Client(
            base_url=base_url, session=Urllib3Session(base_url), action_delay_seconds=0
        )
    raise ValueError(variant)


class Counter:
    """Count the requests a client sends, without changing how it sends them."""

    def __init__(self, client: STS2Client) -> None:
        self.count = 0
        send = client._send

        def counted(*args, **kwargs):
            self.count += 1
            return send(*args, **kwargs)

        client._send = counted  # type: ignore[method-assign]


def reach_fight(env: GameEnv, seed: str, legal: LegalActionProvider, max_steps: int = 400):
    state = env.reset(ResetSpec(run_seed=seed))
    for _ in range(max_steps):
        if state.get("state_type") in BRANCHABLE and len(legal.candidates(state)) > 1:
            return state
        state = env.step(legal.require_candidates(state)[0]).raw_state
    raise RuntimeError("no fight reached")


def run_cycles(
    sim: SimulatorSearchEnv,
    point: int,
    legal: LegalActionProvider,
    cycles: int,
    steps: int,
    rng: random.Random,
) -> None:
    for index in range(cycles):
        state = sim.restore(point)
        sim.reseed(rng.getrandbits(32))
        for _ in range(steps):
            if is_fight_over(state):
                break
            try:
                offered = legal.require_candidates(state)
            except NoLegalActionsError:
                break
            state = sim.step(rng.choice(offered))


def json_probe(sim: SimulatorSearchEnv, point: int, repeats: int) -> None:
    client = sim.env.client
    body = client.session.request(
        method="POST",
        url=client._url("sim/restore"),
        params=None,
        json={"id": point},
        timeout=client.timeout,
    ).content
    print(f"json probe: one restore response is {len(body)} bytes")

    def timed(name, loads):
        best = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(repeats):
                loads(body)
            best.append((time.perf_counter() - start) / repeats * 1e6)
        print(f"  {name:8s} median {statistics.median(best):7.1f} us per parse")

    timed("json", json.loads)
    try:
        import orjson
    except ImportError:
        print("  orjson   not importable (not a project dependency)")
    else:
        timed("orjson", orjson.loads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=15650)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--cycles", type=int, default=400)
    parser.add_argument("--steps", type=int, default=4, help="combat steps per cycle")
    parser.add_argument("--variants", default="old,new,urllib3")
    parser.add_argument("--json-probe", action="store_true")
    args = parser.parse_args()
    base_url = f"http://localhost:{args.port}/api/v1"
    variants = args.variants.split(",")
    legal = LegalActionProvider()

    # One simulator serves every variant in turn: each resets its own run on the same
    # seed and holds its own branch point, so all of them search the same fight.
    lanes = {}
    for variant in variants:
        client = make_client(variant, base_url)
        env = GameEnv(client=client, backend="sim")
        lanes[variant] = (client, env)

    results: dict[str, list[tuple[float, float]]] = {v: [] for v in variants}
    for round_index in range(args.rounds):
        order = variants if round_index % 2 == 0 else variants[::-1]
        for variant in order:
            client, env = lanes[variant]
            reach_fight(env, args.seed, legal)
            sim = SimulatorSearchEnv(env)
            point = sim.snapshot()
            counter = Counter(client)
            rng = random.Random(round_index)
            wall, cpu = time.perf_counter(), time.process_time()
            run_cycles(sim, point, legal, args.cycles, args.steps, rng)
            wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
            sim.release(point)
            del client._send  # drop the counting wrapper
            requests_sent = counter.count
            results[variant].append((requests_sent / wall, cpu / requests_sent * 1e3))
            print(
                f"round {round_index} {variant:8s} {requests_sent:5d} requests "
                f"{requests_sent / wall:7.1f} req/s  cpu {cpu / requests_sent * 1e3:6.3f} ms/req"
            )

    print("\nmedian over rounds")
    base = statistics.median(rate for rate, _ in results[variants[0]])
    base_cpu = statistics.median(c for _, c in results[variants[0]])
    for variant in variants:
        rate = statistics.median(r for r, _ in results[variant])
        cpu = statistics.median(c for _, c in results[variant])
        print(
            f"  {variant:8s} {rate:7.1f} req/s ({rate / base - 1:+.0%})  "
            f"cpu {cpu:6.3f} ms/req ({cpu / base_cpu - 1:+.0%})"
        )

    for variant, (client, _) in lanes.items():
        # Keep-alive check: every variant should have opened one connection in total.
        adapter = getattr(client.session, "adapters", {}).get("http://")
        if adapter is not None:
            pools = adapter.poolmanager.pools
            opened = sum(pools[key].num_connections for key in pools.keys())
        else:
            opened = client.session.pool.num_connections
        print(f"{variant}: connections opened {opened}")

    if args.json_probe:
        _, env = lanes[variants[-1]]
        reach_fight(env, args.seed, legal)
        sim = SimulatorSearchEnv(env)
        point = sim.snapshot()
        json_probe(sim, point, 300)
        sim.release(point)


if __name__ == "__main__":
    main()
