"""Time the HTTP client on MCTS-style cycles against one running STS2Simulator.

Start a simulator first, for example on port 15650, then run::

    uv run python scripts/bench_client.py --port 15650

Every variant plays the same work in the same process: reset a run on a seed
with the trainer's own ``GameEnv(backend="sim")`` reset, walk to the first
fight, hold a branch point, and repeat cycles of restore, reseed and a few
combat steps.  Variants are interleaved round by round, so load from other
processes falls on all of them alike, and the report gives medians over rounds.

Variants:

* ``requests-env`` -- the client as it was before d43fcfc: ``requests`` with
  ``trust_env`` on (netrc and proxy lookups on every request).
* ``requests`` -- the client as of d43fcfc: ``requests`` with ``trust_env``
  off.  Restore and reseed are two requests.
* ``urllib3`` -- the current ``STS2Client``, over a urllib3 connection pool.
  Restore and reseed are still two requests, so against ``requests`` this
  isolates the transport.
* ``urllib3-1call`` -- the current client with ``restore(point, seed)``: one
  request when the simulator applies ``reseed`` on ``/sim/restore``.  An older
  simulator does not echo ``reseeded``, and the client then falls back to two
  requests; the report says which happened.

Two costs are reported.  Per request (``ms/req``) compares transports; per
cycle (``ms/cyc``, one restore + reseed + up to ``--steps`` steps) compares
the one-call restore with the two-call one, since it sends fewer requests for
the same work.

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
from typing import Any, Optional

import requests
import urllib3

from sts2rl.agents.action_space import LegalActionProvider, NoLegalActionsError
from sts2rl.env.game_env import GameEnv
from sts2rl.env.mcp_client import (
    RETRYABLE_METHODS,
    STS2Client,
    STS2ClientError,
    _state_of,
)
from sts2rl.env.reset import ResetSpec
from sts2rl.search.evaluate import is_fight_over
from sts2rl.search.sim_env import BRANCHABLE, SimulatorSearchEnv

DEFAULT_SEED = "0357NDTC82"
VARIANTS = ("requests-env", "requests", "urllib3", "urllib3-1call")


class RequestsClient(STS2Client):
    """``STS2Client`` with the transport of d43fcfc: a ``requests.Session``.

    ``_request`` and ``_send`` are that commit's code, so the comparison is
    against what shipped, not against a re-creation of it.  Measurement only.
    """

    def __init__(self, base_url: str, trust_env: bool) -> None:
        super().__init__(base_url=base_url, action_delay_seconds=0)
        self.pool.close()
        self.session = requests.Session()
        self.session.trust_env = trust_env

    def close(self) -> None:
        # STS2Client.close() would close the pool, already closed; the session is ours.
        self.session.close()

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
    ) -> Any:
        response = self._send(method, endpoint, params, json_body)
        try:
            data = response.json()
        except ValueError as exc:
            raise STS2ClientError(
                f"Non-JSON response: HTTP {response.status_code}: {response.text}"
            ) from exc
        if response.status_code >= 400:
            if isinstance(data, dict):
                msg = data.get("error") or data.get("message") or str(data)
            else:
                msg = str(data)
            raise STS2ClientError(f"HTTP {response.status_code}: {msg}", _state_of(data))
        if isinstance(data, dict) and data.get("status") == "error":
            raise STS2ClientError(data.get("error", "Unknown API error"), _state_of(data))
        return data

    def _send(self, method, endpoint, params, json_body):  # type: ignore[override]
        attempt = 0
        while True:
            try:
                return self.session.request(
                    method=method,
                    url=self._url(endpoint),
                    params=params,
                    json=json_body,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                retry = (
                    method in RETRYABLE_METHODS
                    and attempt < self.max_retries
                    and isinstance(exc, (requests.ConnectionError, requests.Timeout))
                )
                if retry:
                    time.sleep(self.retry_backoff_seconds * (2**attempt))
                    attempt += 1
                    continue
                raise STS2ClientError(f"Request failed: {exc}") from exc


def make_client(variant: str, base_url: str) -> STS2Client:
    if variant == "requests-env":
        return RequestsClient(base_url, trust_env=True)
    if variant == "requests":
        return RequestsClient(base_url, trust_env=False)
    if variant in ("urllib3", "urllib3-1call"):
        return STS2Client(base_url=base_url, action_delay_seconds=0)
    raise ValueError(variant)


class ConnectCounter:
    """Count TCP connects per variant, by wrapping urllib3's ``HTTPConnection.connect``.

    Both transports open their sockets there (requests sits on urllib3).  A pool's
    ``num_connections`` would not do: it counts connection objects, and urllib3
    reconnects a dropped keep-alive connection inside the same object.  The bench
    is single-threaded, so the variant running now owns every connect.
    """

    def __init__(self) -> None:
        self.current: str | None = None
        self.connects: dict[str, int] = {}
        self._original = urllib3.connection.HTTPConnection.connect
        counter = self

        def connect(conn, *args, **kwargs):
            if counter.current is not None:
                counter.connects[counter.current] = counter.connects.get(counter.current, 0) + 1
            return counter._original(conn, *args, **kwargs)

        urllib3.connection.HTTPConnection.connect = connect  # type: ignore[method-assign]

    def restore(self) -> None:
        urllib3.connection.HTTPConnection.connect = self._original  # type: ignore[method-assign]


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
    one_call: bool,
) -> None:
    for _ in range(cycles):
        seed = rng.getrandbits(32)
        if one_call:
            state = sim.restore(point, seed)
        else:
            state = sim.restore(point)
            sim.reseed(seed)
        for _ in range(steps):
            if is_fight_over(state):
                break
            try:
                offered = legal.require_candidates(state)
            except NoLegalActionsError:
                break
            state = sim.step(rng.choice(offered))


def json_probe(client: STS2Client, point: int, repeats: int) -> None:
    _, body = client._send("POST", "sim/restore", None, {"id": point})
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
    parser.add_argument("--variants", default="requests,urllib3,urllib3-1call")
    parser.add_argument("--json-probe", action="store_true")
    args = parser.parse_args()
    base_url = f"http://localhost:{args.port}/api/v1"
    variants = args.variants.split(",")
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        parser.error(f"unknown variants {sorted(unknown)}; choose from {VARIANTS}")
    legal = LegalActionProvider()

    # One simulator serves every variant in turn: each resets its own run on the same
    # seed and holds its own branch point, so all of them search the same fight.
    lanes = {}
    for variant in variants:
        client = make_client(variant, base_url)
        env = GameEnv(client=client, backend="sim")
        lanes[variant] = (client, env)
    connects = ConnectCounter()
    try:
        results = measure(args, variants, lanes, legal, connects)
        report(args, variants, lanes, results, connects)
        if args.json_probe:
            client, env = lanes[variants[-1]]
            reach_fight(env, args.seed, legal)
            sim = SimulatorSearchEnv(env)
            point = sim.snapshot()
            try:
                json_probe(client, point, 300)
            finally:
                sim.release(point)
    finally:
        connects.restore()
        for client, _ in lanes.values():
            client.close()


def measure(args, variants, lanes, legal, connects):
    # Per round: (requests/s, cpu ms per request, cycles/s, cpu ms per cycle, requests per cycle)
    results: dict[str, list[tuple[float, float, float, float, float]]] = {v: [] for v in variants}
    for round_index in range(args.rounds):
        order = variants if round_index % 2 == 0 else variants[::-1]
        for variant in order:
            client, env = lanes[variant]
            connects.current = variant
            reach_fight(env, args.seed, legal)
            sim = SimulatorSearchEnv(env)
            point = sim.snapshot()
            counter = Counter(client)
            try:
                rng = random.Random(round_index)
                wall, cpu = time.perf_counter(), time.process_time()
                run_cycles(
                    sim, point, legal, args.cycles, args.steps, rng, variant.endswith("-1call")
                )
                wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
                # Read where the clock stops: releasing the point is one more request.
                sent = counter.count
            finally:
                del client._send  # drop the counting wrapper
                sim.release(point)
            row = (
                sent / wall,
                cpu / sent * 1e3,
                args.cycles / wall,
                cpu / args.cycles * 1e3,
                sent / args.cycles,
            )
            results[variant].append(row)
            print(
                f"round {round_index} {variant:13s} {sent:5d} req {row[0]:7.1f} req/s "
                f"cpu {row[1]:6.3f} ms/req  {row[2]:6.1f} cyc/s cpu {row[3]:6.3f} ms/cyc"
            )
    connects.current = None
    return results


def report(args, variants, lanes, results, connects) -> None:
    print(f"\nmedian over {args.rounds} rounds ({args.cycles} cycles of up to {args.steps} steps)")
    base = variants[0]

    def median(variant: str, column: int) -> float:
        return statistics.median(row[column] for row in results[variant])

    for variant in variants:
        print(
            f"  {variant:13s} {median(variant, 4):4.2f} req/cyc  "
            f"{median(variant, 0):7.1f} req/s ({median(variant, 0) / median(base, 0) - 1:+.0%})  "
            f"cpu {median(variant, 1):6.3f} ms/req ({median(variant, 1) / median(base, 1) - 1:+.0%})  "
            f"{median(variant, 2):6.1f} cyc/s ({median(variant, 2) / median(base, 2) - 1:+.0%})  "
            f"cpu {median(variant, 3):6.3f} ms/cyc ({median(variant, 3) / median(base, 3) - 1:+.0%})"
        )

    for variant, (client, _) in lanes.items():
        # Keep-alive check: TCP connects over every round, resets included.  One
        # means every request of the variant reused one socket.
        line = f"{variant}: TCP connects {connects.connects.get(variant, 0)}"
        if variant.endswith("-1call"):
            line += (
                "; simulator reseeds on restore (one request)"
                if client._restore_reseeds
                else "; simulator ignores reseed on restore (fell back to two requests)"
            )
        print(line)


if __name__ == "__main__":
    main()
