import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import urllib3
from urllib3.exceptions import (
    ConnectTimeoutError,
    DecodeError,
    MaxRetryError,
    NewConnectionError,
    ProtocolError,
    ReadTimeoutError,
)

from sts2rl.env import mcp_client
from sts2rl.env.mcp_client import STS2Client, STS2ClientError


class FakeResponse:
    """What the client reads from a urllib3 response: the status and the body."""

    def __init__(self, data=None, status=200, raw: bytes | None = None):
        self.status = status
        self.data = raw if raw is not None else json.dumps(data).encode()


class FakePool:
    """A urllib3 connection pool that records each request and answers it.

    ``answer`` is a response, or a function of the decoded request returning one.
    """

    def __init__(self, answer=None):
        self.answer = answer if answer is not None else FakeResponse({"state_type": "menu"})
        self.requests: list[dict] = []
        self.closed = False

    def urlopen(self, method, url, body=None, headers=None, **kwargs):
        request = {
            "method": method,
            "url": url,
            "json": json.loads(body) if body is not None else None,
            "headers": headers,
            **kwargs,
        }
        self.requests.append(request)
        return self.answer(request) if callable(self.answer) else self.answer

    def close(self):
        self.closed = True


def test_client_builds_request_from_complete_base_url():
    pool = FakePool()
    client = STS2Client(base_url="http://localhost:16666/api/v1/", timeout=3.5, pool=pool)

    state = client.get_state()

    assert state == {"state_type": "menu"}
    (request,) = pool.requests
    assert (request["method"], request["url"], request["json"]) == (
        "GET",
        "/api/v1/singleplayer?format=json",
        None,
    )
    timeout = request["timeout"]
    assert (timeout.connect_timeout, timeout.read_timeout) == (3.5, 3.5)
    # urllib3 never retries on its own: the client's rule is the only one.
    assert request["retries"] is False


def test_query_parameters_are_url_encoded():
    pool = FakePool(FakeResponse({"results": []}))

    STS2Client(pool=pool).search_wiki("Perfected Strike & co", limit=3)

    assert pool.requests[0]["url"] == (
        "/api/v1/wiki?query=Perfected+Strike+%26+co&item_type=all&limit=3"
    )


def test_an_action_posts_its_json_body():
    pool = FakePool(FakeResponse({"status": "ok"}))

    STS2Client(pool=pool, action_delay_seconds=0).play_card(2, target="JAW_WORM_0")

    (request,) = pool.requests
    assert request["method"] == "POST"
    assert request["url"] == "/api/v1/singleplayer"
    assert request["json"] == {"action": "play_card", "card_index": 2, "target": "JAW_WORM_0"}
    assert request["headers"]["Content-Type"] == "application/json"


def test_client_does_not_close_injected_pool():
    pool = FakePool()
    client = STS2Client(pool=pool)

    client.close()

    assert pool.closed is False


def test_context_manager_closes_owned_pool():
    with STS2Client() as client:
        assert isinstance(client.pool, urllib3.HTTPConnectionPool)
        assert client.pool.pool is not None

    # HTTPConnectionPool.close drops its queue of connections.
    assert client.pool.pool is None


def test_https_is_verified_and_other_schemes_are_refused():
    with STS2Client(base_url="https://sim.example:15600/api/v1") as client:
        assert isinstance(client.pool, urllib3.HTTPSConnectionPool)
        # None resolves to CERT_REQUIRED against the system trust store.
        assert client.pool.cert_reqs in (None, "CERT_REQUIRED")
        assert client.pool.ca_certs is None

    with pytest.raises(ValueError, match="http:// or https://"):
        STS2Client(base_url="ftp://localhost:15600/api/v1")


def test_a_redirect_is_not_followed():
    """Neither the mod nor the simulator redirects; a 3xx is an answer, not a hop."""
    pool = FakePool(FakeResponse({"error": "moved"}, status=302))

    with pytest.raises(STS2ClientError, match="HTTP 302: moved"):
        STS2Client(pool=pool).get_state()

    assert pool.requests[0]["redirect"] is False
    assert len(pool.requests) == 1


def test_http_error_with_non_object_json_has_clear_message():
    client = STS2Client(pool=FakePool(FakeResponse(["backend failure"], status=500)))

    with pytest.raises(STS2ClientError) as caught:
        client.get_state()

    assert str(caught.value) == "HTTP 500: ['backend failure']"


def test_a_non_json_response_is_an_error_with_its_text():
    client = STS2Client(pool=FakePool(FakeResponse(raw=b"<html>oops</html>", status=502)))

    with pytest.raises(STS2ClientError, match="Non-JSON response: HTTP 502: <html>oops</html>"):
        client.get_state()


def test_rejected_actions_carry_the_unchanged_state():
    """The API returns HTTP 200 with status=error and the state it did not change."""
    pool = FakePool(
        FakeResponse(
            {
                "status": "error",
                "error": "card_index 99 out of range",
                "state": {"state_type": "monster"},
            }
        )
    )

    with pytest.raises(STS2ClientError, match="out of range") as caught:
        STS2Client(pool=pool).play_card(99)

    assert caught.value.state == {"state_type": "monster"}


def test_errors_without_a_state_carry_none():
    pool = FakePool(FakeResponse({"error": "Not found"}, status=404))

    with pytest.raises(STS2ClientError) as caught:
        STS2Client(pool=pool).get_state()

    assert caught.value.state is None


def test_accepted_actions_pause_before_the_next_request(monkeypatch):
    """The next action must not reach a game still resolving the last one."""
    slept: list[float] = []
    monkeypatch.setattr(mcp_client.time, "sleep", slept.append)
    client = STS2Client(pool=FakePool(), action_delay_seconds=0.1)

    client.play_card(0)
    client.end_turn()

    assert slept == [0.1, 0.1]


def test_reading_state_does_not_pause(monkeypatch):
    """Only actions change the game, so only actions are worth waiting on."""
    slept: list[float] = []
    monkeypatch.setattr(mcp_client.time, "sleep", slept.append)
    client = STS2Client(pool=FakePool(), action_delay_seconds=0.1)

    client.get_state()

    assert slept == []


def test_rejected_actions_do_not_pause(monkeypatch):
    """A rejected action changed nothing, so there is nothing to settle."""
    slept: list[float] = []
    monkeypatch.setattr(mcp_client.time, "sleep", slept.append)
    pool = FakePool(
        FakeResponse({"status": "error", "error": "no", "state": {"state_type": "map"}})
    )
    client = STS2Client(pool=pool, action_delay_seconds=0.1)

    with pytest.raises(STS2ClientError):
        client.play_card(0)

    assert slept == []


def test_a_zero_delay_skips_the_call_entirely(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(mcp_client.time, "sleep", slept.append)
    client = STS2Client(pool=FakePool(), action_delay_seconds=0.0)

    client.end_turn()

    assert slept == []


def test_a_negative_delay_is_rejected():
    with pytest.raises(ValueError, match="action_delay_seconds"):
        STS2Client(action_delay_seconds=-0.1)


# Every way urllib3 reports a request that produced no response, with
# ``retries=False`` (which raises the underlying error, not MaxRetryError) and
# without.  requests mapped each of them onto ConnectionError or Timeout.
TRANSPORT_FAILURES = [
    pytest.param(lambda: ProtocolError("Connection aborted.", ConnectionResetError(10054)), id="reset"),
    pytest.param(lambda: NewConnectionError(None, "refused"), id="refused"),
    pytest.param(lambda: ConnectTimeoutError("connect timed out"), id="connect-timeout"),
    pytest.param(lambda: ReadTimeoutError(None, "/x", "read timed out"), id="read-timeout"),
    pytest.param(lambda: MaxRetryError(None, "/x", NewConnectionError(None, "refused")), id="max-retry"),
    pytest.param(lambda: ConnectionRefusedError(10061, "refused"), id="oserror"),
]


class FlakyPool(FakePool):
    """A pool whose first ``failures`` requests die in transport."""

    def __init__(self, failures: int, error=None, answer=None):
        super().__init__(answer)
        self.remaining = failures
        self.error = error or (lambda: ProtocolError("Connection aborted.", ConnectionResetError()))
        self.attempts = 0

    def urlopen(self, method, url, body=None, headers=None, **kwargs):
        self.attempts += 1
        if self.remaining:
            self.remaining -= 1
            raise self.error()
        return super().urlopen(method, url, body=body, headers=headers, **kwargs)


@pytest.mark.parametrize("error", TRANSPORT_FAILURES)
def test_a_dropped_read_is_retried(monkeypatch, error):
    """Clients on one machine drop connections; a read replays for free."""
    slept: list[float] = []
    monkeypatch.setattr(mcp_client.time, "sleep", slept.append)
    pool = FlakyPool(failures=2, error=error)
    client = STS2Client(pool=pool, retry_backoff_seconds=0.5)

    state = client.get_state()

    assert state == {"state_type": "menu"}
    assert pool.attempts == 3
    assert slept == [0.5, 1.0]


def test_a_read_that_never_recovers_still_fails(monkeypatch):
    monkeypatch.setattr(mcp_client.time, "sleep", lambda _: None)
    pool = FlakyPool(failures=99)
    client = STS2Client(pool=pool, max_retries=2)

    with pytest.raises(STS2ClientError, match="Request failed"):
        client.get_state()

    assert pool.attempts == 3


@pytest.mark.parametrize("error", TRANSPORT_FAILURES)
def test_a_dropped_action_is_never_replayed(monkeypatch, error):
    """The game may have applied it before the socket died; replaying it twice
    would play the card twice, so an action surfaces instead."""
    monkeypatch.setattr(mcp_client.time, "sleep", lambda _: None)
    pool = FlakyPool(failures=1, error=error)
    client = STS2Client(pool=pool)

    with pytest.raises(STS2ClientError, match="Request failed"):
        client.play_card(0)

    assert pool.attempts == 1


def test_a_dropped_release_is_never_replayed(monkeypatch):
    """DELETE is not a read either: only GET replays."""
    monkeypatch.setattr(mcp_client.time, "sleep", lambda _: None)
    pool = FlakyPool(failures=1)

    with pytest.raises(STS2ClientError, match="Request failed"):
        STS2Client(pool=pool).sim_release(3)

    assert pool.attempts == 1


def test_a_failure_that_is_not_transport_is_not_retried(monkeypatch):
    """A response that arrived but could not be decoded is an answer, not a drop."""
    monkeypatch.setattr(mcp_client.time, "sleep", lambda _: None)
    pool = FlakyPool(failures=5, error=lambda: DecodeError("bad gzip"))

    with pytest.raises(STS2ClientError, match="bad gzip"):
        STS2Client(pool=pool).get_state()

    assert pool.attempts == 1


def test_an_http_error_is_a_real_answer_and_is_not_retried(monkeypatch):
    monkeypatch.setattr(mcp_client.time, "sleep", lambda _: None)
    pool = FakePool(FakeResponse({"error": "nope"}, status=500))
    client = STS2Client(pool=pool)

    with pytest.raises(STS2ClientError, match="HTTP 500"):
        client.get_state()

    assert len(pool.requests) == 1


def test_a_negative_retry_count_is_rejected():
    with pytest.raises(ValueError, match="max_retries"):
        STS2Client(max_retries=-1)


def test_branch_points_use_the_simulator_endpoints():
    pool = FakePool(FakeResponse({"status": "ok", "id": 7}))
    client = STS2Client(base_url="http://localhost:15600/api/v1", pool=pool)

    assert client.sim_snapshot() == 7
    client.sim_restore(7)
    client.sim_release(7)

    assert [(r["method"], r["url"], r["json"]) for r in pool.requests] == [
        ("POST", "/api/v1/sim/snapshot", {}),
        ("POST", "/api/v1/sim/restore", {"id": 7}),
        ("DELETE", "/api/v1/sim/snapshot/7", None),
    ]


def test_restoring_an_unknown_branch_point_fails_loudly():
    pool = FakePool(FakeResponse({"status": "error", "error": "No branch point 9."}, status=400))
    client = STS2Client(pool=pool)

    with pytest.raises(STS2ClientError, match="No branch point 9"):
        client.sim_restore(9)


def test_reseed_and_seeded_reset_reach_the_simulator():
    pool = FakePool(FakeResponse({"status": "ok"}))
    client = STS2Client(base_url="http://localhost:15600/api/v1", pool=pool)

    client.sim_reseed(123)
    client.sim_reset("IRONCLAD", "ABC", start_boss=True, reseed=9)
    client.sim_reset("IRONCLAD", "ABC")

    reseed, seeded, plain = pool.requests
    assert (reseed["url"], reseed["json"]) == ("/api/v1/sim/reseed", {"seed": 123})
    assert seeded["json"]["reseed"] == 9
    # Without a reseed the field is left out, so the simulator restores streams as saved.
    assert "reseed" not in plain["json"]


def _simulator(echoes: bool):
    """Answer restore and reseed as a simulator does, with or without the echo."""

    def answer(request):
        body = request["json"]
        if request["url"].endswith("/sim/restore"):
            response = {"status": "ok", "state": {"state_type": "monster", "from": "restore"}}
            if echoes and "reseed" in body:
                response["reseeded"] = body["reseed"]
            return FakeResponse(response)
        return FakeResponse({"status": "ok", "state": {"state_type": "monster", "from": "reseed"}})

    return answer


def test_a_restore_with_a_reseed_is_one_request():
    pool = FakePool(_simulator(echoes=True))
    client = STS2Client(pool=pool)

    response = client.sim_restore(4, reseed=77)
    client.sim_restore(4, reseed=78)

    assert response["reseeded"] == 77
    assert [(r["url"], r["json"]) for r in pool.requests] == [
        ("/api/v1/sim/restore", {"id": 4, "reseed": 77}),
        ("/api/v1/sim/restore", {"id": 4, "reseed": 78}),
    ]


@pytest.mark.parametrize("echoes, requests_sent", [(True, 1), (False, 2)])
def test_the_search_env_restores_and_reseeds_through_the_client(echoes, requests_sent):
    from sts2rl.env.game_env import GameEnv
    from sts2rl.search.sim_env import SimulatorSearchEnv

    pool = FakePool(_simulator(echoes=echoes))
    sim = SimulatorSearchEnv(GameEnv(client=STS2Client(pool=pool), backend="sim"))

    state = sim.restore(4, 9)
    plain = sim.restore(4)

    # Either way the state comes from the response that applied the reseed.
    assert state == {"state_type": "monster", "from": "reseed" if not echoes else "restore"}
    assert plain["from"] == "restore"
    assert len(pool.requests) == requests_sent + 1
    assert pool.requests[-1]["json"] == {"id": 4}


def test_seed_zero_is_a_seed():
    pool = FakePool(_simulator(echoes=True))

    response = STS2Client(pool=pool).sim_restore(4, reseed=0)

    assert response["reseeded"] == 0
    assert [r["json"] for r in pool.requests] == [{"id": 4, "reseed": 0}]


def test_a_restore_without_a_reseed_sends_no_reseed_field():
    pool = FakePool(_simulator(echoes=True))

    STS2Client(pool=pool).sim_restore(4)

    assert [r["json"] for r in pool.requests] == [{"id": 4}]


def test_a_simulator_that_ignores_the_reseed_gets_a_separate_one(caplog):
    """An old simulator restores, ignores ``reseed``, and does not echo it. Trusting
    that would plan against the real future; the client reseeds itself instead."""
    pool = FakePool(_simulator(echoes=False))
    client = STS2Client(pool=pool)

    with caplog.at_level(logging.WARNING, logger=mcp_client.__name__):
        first = client.sim_restore(4, reseed=77)
        second = client.sim_restore(4, reseed=78)

    # The state handed back is the one after the reseed.
    assert first["state"]["from"] == second["state"]["from"] == "reseed"
    assert [(r["url"], r["json"]) for r in pool.requests] == [
        ("/api/v1/sim/restore", {"id": 4, "reseed": 77}),
        ("/api/v1/sim/reseed", {"seed": 77}),
        # Known now: restore plainly and reseed separately, without asking again.
        ("/api/v1/sim/restore", {"id": 4}),
        ("/api/v1/sim/reseed", {"seed": 78}),
    ]
    assert len([r for r in caplog.records if "ignores 'reseed'" in r.getMessage()]) == 1


def test_a_restore_that_reseeds_another_seed_fails():
    def answer(request):
        return FakeResponse({"status": "ok", "reseeded": 5, "state": {"state_type": "monster"}})

    with pytest.raises(STS2ClientError, match="reseeded 5, asked for 77") as caught:
        STS2Client(pool=FakePool(answer)).sim_restore(4, reseed=77)

    assert caught.value.state == {"state_type": "monster"}


def test_a_refused_reseed_on_restore_carries_the_restored_state():
    pool = FakePool(
        FakeResponse(
            {"status": "error", "error": "No fight to reseed", "state": {"state_type": "map"}},
            status=409,
        )
    )

    with pytest.raises(STS2ClientError, match="No fight to reseed") as caught:
        STS2Client(pool=pool).sim_restore(4, reseed=77)

    assert caught.value.state == {"state_type": "map"}
    assert len(pool.requests) == 1


# ---------------------------------------------------------------------------
# Against a real socket: keep-alive, timeouts, dropped connections, proxies.
# ---------------------------------------------------------------------------


class _Server:
    """A local HTTP/1.1 server that counts connections and requests."""

    def __init__(self) -> None:
        owner = self
        self.connections = 0
        self.paths: list[str] = []
        # True: hold every response until ``release`` is set (at close), so a
        # client timeout fires whatever the machine's load.
        self.stall = False
        self.release = threading.Event()
        self.drops = 0
        # Responses to cut short mid-body: "length" (a fixed Content-Length) or
        # "chunked", one entry per response.
        self.cuts: list[str] = []
        self.lock = threading.Condition()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self):
                super().setup()
                with owner.lock:
                    owner.connections += 1

            def _answer(self):
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                with owner.lock:
                    owner.paths.append(self.path)
                    owner.lock.notify_all()
                    drop = owner.drops > 0
                    owner.drops -= drop
                    cut = owner.cuts.pop(0) if owner.cuts else None
                if drop:
                    # Close without a response: the client sees a reset connection.
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
                if owner.stall:
                    owner.release.wait(10)
                payload = b'{"state_type": "menu"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                if cut == "chunked":
                    # One chunk announced at its full size, half of it sent, no
                    # terminating chunk.
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    self.wfile.write(b"%x\r\n" % len(payload) + payload[: len(payload) // 2])
                elif cut == "length":
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload[: len(payload) // 2])
                else:
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.wfile.flush()
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)

            do_GET = do_POST = do_DELETE = _answer

            def log_message(self, *args):
                pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                pass  # a client that timed out leaves a write to a closed socket

        self.httpd = Server(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/api/v1"

    def wait_for_requests(self, count: int) -> int:
        """Wait until the server has read ``count`` requests; return how many it read.

        A client gives up on its own clock, and the server thread can read the
        request after that, so counts are synchronised here rather than by timing.
        """
        with self.lock:
            self.lock.wait_for(lambda: len(self.paths) >= count, timeout=10)
            return len(self.paths)

    def close(self) -> None:
        self.release.set()
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    running = _Server()
    yield running
    running.close()


def test_keep_alive_sends_every_request_over_one_connection(server):
    with STS2Client(base_url=server.base_url, action_delay_seconds=0) as client:
        for _ in range(20):
            client.get_state()
            client.end_turn()

    assert len(server.paths) == 40
    assert server.connections == 1


def test_environment_proxies_and_netrc_are_ignored(server, monkeypatch, tmp_path):
    """A system proxy could only capture a local request it was never meant for."""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://proxy.invalid:9")
        monkeypatch.setenv(name.lower(), "http://proxy.invalid:9")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(tmp_path / "missing.pem"))
    monkeypatch.setenv("CURL_CA_BUNDLE", str(tmp_path / "missing.pem"))
    netrc = tmp_path / ".netrc"
    netrc.write_text("machine 127.0.0.1 login user password secret\n")
    monkeypatch.setenv("NETRC", str(netrc))

    with STS2Client(base_url=server.base_url) as client:
        assert client.get_state() == {"state_type": "menu"}
        assert client.pool.proxy is None

    # Origin-form, straight to the server: a proxy would have been sent the absolute URL.
    assert server.paths == ["/api/v1/singleplayer?format=json"]


def test_a_read_timeout_is_retried_on_reads_only(server, monkeypatch):
    monkeypatch.setattr(mcp_client.time, "sleep", lambda _: None)
    server.stall = True  # no response until the test ends
    with STS2Client(base_url=server.base_url, timeout=0.3, max_retries=2) as client:
        with pytest.raises(STS2ClientError, match="Read timed out"):
            client.get_state()
        assert server.wait_for_requests(3) == 3

        with pytest.raises(STS2ClientError, match="Read timed out"):
            client.end_turn()
        assert server.wait_for_requests(4) == 4
        assert server.paths[-1] == "/api/v1/singleplayer"


def test_a_connection_reset_mid_request_is_retried_for_a_read(server, monkeypatch):
    monkeypatch.setattr(mcp_client.time, "sleep", lambda _: None)
    server.drops = 1
    with STS2Client(base_url=server.base_url) as client:
        assert client.get_state() == {"state_type": "menu"}

    assert len(server.paths) == 2


def test_a_connection_reset_mid_action_is_not_replayed(server, monkeypatch):
    monkeypatch.setattr(mcp_client.time, "sleep", lambda _: None)
    server.drops = 1
    with STS2Client(base_url=server.base_url, action_delay_seconds=0) as client:
        with pytest.raises(STS2ClientError, match="Request failed"):
            client.end_turn()
        # The connection is replaced, and the next request works.
        assert client.get_state() == {"state_type": "menu"}

    assert server.paths == ["/api/v1/singleplayer", "/api/v1/singleplayer?format=json"]


def test_an_unreachable_server_is_retried_for_a_read_and_then_fails(monkeypatch):
    attempts: list[float] = []
    monkeypatch.setattr(mcp_client.time, "sleep", attempts.append)
    # The port stays bound, and never listens, until the assertions are done: no
    # other process can take it in between.  Linux refuses at once; Windows retries
    # the SYN for about two seconds, which the short timeout turns into a connect
    # timeout.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{probe.getsockname()[1]}/api/v1"
        with STS2Client(base_url=url, max_retries=2, timeout=0.3) as client:
            with pytest.raises(STS2ClientError, match="Request failed"):
                client.get_state()

        assert len(attempts) == 2


@pytest.mark.parametrize("cut", ["length", "chunked"])
def test_a_body_cut_short_is_retried_for_a_read(server, monkeypatch, cut):
    """The connection dies mid-body: the response never arrived whole."""
    monkeypatch.setattr(mcp_client.time, "sleep", lambda _: None)
    server.cuts = [cut]
    with STS2Client(base_url=server.base_url) as client:
        assert client.get_state() == {"state_type": "menu"}

    assert server.wait_for_requests(2) == 2
    assert server.paths == ["/api/v1/singleplayer?format=json"] * 2


@pytest.mark.parametrize("cut", ["length", "chunked"])
def test_a_body_cut_short_is_not_resent_for_an_action(server, monkeypatch, cut):
    """The game applied the action before answering; resending would apply it twice."""
    monkeypatch.setattr(mcp_client.time, "sleep", lambda _: None)
    server.cuts = [cut]
    with STS2Client(base_url=server.base_url, action_delay_seconds=0) as client:
        with pytest.raises(STS2ClientError, match="Request failed"):
            client.end_turn()

    assert server.wait_for_requests(1) == 1
    assert server.paths == ["/api/v1/singleplayer"]
