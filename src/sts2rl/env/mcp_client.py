"""HTTP client for the STS2MCP local API.

Generated from the STS2MCP API documentation:
https://github.com/Gennadiyev/STS2MCP/blob/main/docs/raw-full.md
"""


from __future__ import annotations

import json
import logging
import time
from typing import Any, Literal, Optional
from urllib.parse import urlencode, urlsplit

import urllib3
from urllib3.exceptions import HTTPError as Urllib3Error
from urllib3.exceptions import (
    ClosedPoolError,
    MaxRetryError,
    ProtocolError,
    ProxyError,
    SSLError,
)
from urllib3.exceptions import TimeoutError as Urllib3TimeoutError

logger = logging.getLogger(__name__)


GameMode = Literal["singleplayer", "multiplayer"]
ResponseFormat = Literal["json", "markdown"]
WikiItemType = Literal["all", "card", "relic"]
GameCharacter = {0: "IRONCLAD", 1: "SILENT", 2: "REGENT", 3: "NECROBINDER", 4: "DEFECT"}

# Pause after every accepted action POST, giving the game room to transition.
#
# The mod settles the game before it captures the state it returns, so this is
# not what makes that state correct.  It paces the *next* request instead: an
# action sent while the game is still resolving the previous one is the case
# the settle logic does not cover.  A rejected action changed nothing, so it is
# not followed by a pause.
#
# Every action opens a window the game accepts nothing in -- combat reports
# is_play_phase false while it deals the hand and resolves the enemy turn, and
# cards report can_play true throughout it.  0.1s covered that with one client;
# with several sharing a machine the transitions run longer, so the pause is
# 0.2s.  Set --action-delay to trade it back for throughput.
DEFAULT_ACTION_DELAY_SECONDS = 0.2

# Retry a dropped connection, but only for reads.
#
# Several game clients on one machine drop TCP connections under load: two
# long runs died on a GET with WinError 10061 and 10054, tens of episodes
# apart.  A read is idempotent, so replaying it is free.  An action is not:
# the connection can drop *after* the game applied it, and replaying would
# play the card twice.  A failed action therefore still surfaces, where
# GameEnv.step already turns it into an action_error the episode recovers
# from.
RETRYABLE_METHODS = frozenset({"GET"})
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 0.5


# Transport failures: the request never produced a response, so nothing about
# the game changed.  urllib3's ``TimeoutError`` covers connect and read timeouts
# and a refused connection (``NewConnectionError`` derives from it);
# ``ProtocolError`` is a connection reset or aborted mid-request (WinError
# 10054); ``OSError`` is a socket error urllib3 did not wrap.  These are the
# failures requests reported as ``ConnectionError`` or ``Timeout``, so the
# classification is unchanged with one exception: a body cut short mid-read
# also arrives as ``ProtocolError`` (requests raised ``ChunkedEncodingError``,
# not retried).  Replaying a GET for it is as safe as for any other drop.
# Everything else urllib3 raises (``DecodeError``, ``LocationValueError``, ...)
# is not a drop and fails at once.  An HTTP error status is not here either:
# it is a real answer.
TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    Urllib3TimeoutError,
    ProtocolError,
    MaxRetryError,
    ClosedPoolError,
    SSLError,
    ProxyError,
    OSError,
)

_JSON_HEADERS = {"Content-Type": "application/json"}


def _new_pool(base_url: str, timeout: Optional[float]) -> urllib3.HTTPConnectionPool:
    """Return one keep-alive connection pool for the API's host.

    The client talks to urllib3 directly, below ``requests``.  Against one
    local simulator that cuts the client's CPU per request by about half
    (0.47 ms to 0.27 ms), and the trainer is one GIL-bound process whose
    search sends thousands of local requests per decision.  What requests
    adds per request is work this API never needs: ``.netrc`` and proxy
    lookups (two registry reads on Windows), cookies, redirects, and header
    merging.

    Nothing is read from the environment: no ``HTTP_PROXY``/``HTTPS_PROXY``/
    ``NO_PROXY`` (either case), no ``.netrc``, and no ``REQUESTS_CA_BUNDLE`` or
    ``CURL_CA_BUNDLE``.  The API is a game client or simulator on this machine
    or the LAN: it takes no credentials, and a system proxy can only add a hop
    or capture a ``localhost`` request it was never meant for.  A caller that
    needs a proxy or a custom CA injects its own pool.

    Schemes: ``http://`` is what the mod and the simulator serve.
    ``https://`` is supported with urllib3's default verification -- the
    certificate must chain to the operating system's trust store and match
    the host.  Anything else is refused here.

    Redirects are not followed (requests followed them): neither the mod nor
    the simulator ever redirects, so a 3xx raises ``STS2ClientError`` like a
    4xx, and is not retried.  No cookies are kept.  urllib3 sends
    ``Accept-Encoding: identity``, so no response arrives compressed.

    One connection: each trainer lane owns its own ``STS2Client`` and sends
    one request at a time.  ``block=False`` keeps a second concurrent caller
    working instead of waiting: it gets a temporary connection, closed after
    use.  urllib3's own retries are off; ``STS2Client._send`` holds the one
    retry rule.
    """
    scheme = urlsplit(base_url).scheme
    if scheme not in ("http", "https"):
        raise ValueError(f"base_url must be http:// or https://, got {base_url!r}")
    return urllib3.connection_from_url(
        base_url,
        maxsize=1,
        block=False,
        retries=False,
        timeout=urllib3.Timeout(connect=timeout, read=timeout),
    )


def _state_of(data: Any) -> Optional[dict[str, Any]]:
    """Return the game state an action response carries, if it carries one."""
    if isinstance(data, dict):
        state = data.get("state")
        if isinstance(state, dict):
            return state
    return None


class STS2ClientError(Exception):
    """Raised when the STS2_MCP API returns an error or invalid response.

    A rejected action changes nothing and the API returns the unchanged state
    alongside the error, so it is carried here and callers can use it instead
    of issuing a second request.
    """

    def __init__(self, message: str, state: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.state = state


class STS2Client:
    """
    Python client for the STS2_MCP local HTTP API.

    Default base URL:
        http://localhost:15526/api/v1

    Transport: one urllib3 keep-alive pool per client (``_new_pool``), which
    reads no proxy, ``.netrc`` or CA bundle from the environment and follows
    no redirects.  ``pool`` injects another; the client then leaves it open.

    Example:
        client = STS2Client()

        state = client.get_state()
        print(state["state_type"])

        if state["state_type"] in {"monster", "elite", "boss"}:
            client.play_card(card_index=0, target="JAW_WORM_0")
            client.end_turn()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:15526/api/v1",
        mode: GameMode = "singleplayer",
        timeout: float = 10.0,
        pool: urllib3.HTTPConnectionPool | None = None,
        action_delay_seconds: float = DEFAULT_ACTION_DELAY_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    ) -> None:
        if action_delay_seconds < 0:
            raise ValueError("action_delay_seconds must not be negative")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must not be negative")
        self.base_url = base_url.rstrip("/")
        self.mode = mode
        self.timeout = timeout
        self.action_delay_seconds = action_delay_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._base_path = urlsplit(self.base_url).path
        # Connect and read limits of ``timeout`` seconds each, as requests
        # applied one number.  Passed per request, so an injected pool obeys it.
        self._timeout = urllib3.Timeout(connect=timeout, read=timeout)
        self._owns_pool = pool is None
        self.pool = pool if pool is not None else _new_pool(self.base_url, timeout)
        # Whether /sim/restore applies a reseed itself: None until a response
        # tells (see ``sim_restore``).
        self._restore_reseeds: Optional[bool] = None

    def close(self) -> None:
        """Close the internally created connection pool.

        An injected pool remains owned by its caller and is deliberately not
        closed here.
        """
        if self._owns_pool:
            self.pool.close()

    def __enter__(self) -> "STS2Client":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    # -------------------------
    # Internal request helpers
    # -------------------------

    def _url(self, endpoint: str) -> str:
        """Build an absolute API URL for an endpoint."""
        endpoint = endpoint.lstrip("/")
        return f"{self.base_url}/{endpoint}"

    def _target(self, endpoint: str, params: Optional[dict[str, Any]]) -> str:
        """Build what the pool requests: the endpoint's path and query string."""
        target = f"{self._base_path}/{endpoint.lstrip('/')}"
        if params:
            target = f"{target}?{urlencode(params, doseq=True)}"
        return target

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
    ) -> Any:
        status, content = self._send(method, endpoint, params, json_body)

        try:
            data = json.loads(content)
        except ValueError as exc:
            # UnicodeDecodeError is a ValueError too: undecodable bytes land here.
            text = content.decode("utf-8", errors="replace")
            raise STS2ClientError(f"Non-JSON response: HTTP {status}: {text}") from exc

        # A 3xx is an error too: redirects are not followed (see ``_new_pool``),
        # and its body is not the game state.
        if status >= 300:
            if isinstance(data, dict):
                msg = data.get("error") or data.get("message") or str(data)
            else:
                msg = str(data)
            raise STS2ClientError(f"HTTP {status}: {msg}", _state_of(data))

        if isinstance(data, dict) and data.get("status") == "error":
            raise STS2ClientError(
                data.get("error", "Unknown API error"),
                _state_of(data),
            )

        return data

    def _send(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict[str, Any]],
        json_body: Optional[dict[str, Any]],
    ) -> tuple[int, bytes]:
        """Send one request, retrying a dropped connection on reads only.

        Returns the HTTP status and the whole response body; the connection
        goes back to the pool for the next request.
        """
        target = self._target(endpoint, params)
        if json_body is None:
            body, headers = None, None
        else:
            # As requests encoded ``json=``: ASCII-escaped, NaN refused.
            body = json.dumps(json_body, allow_nan=False).encode("utf-8")
            headers = _JSON_HEADERS
        attempt = 0
        while True:
            try:
                response = self.pool.urlopen(
                    method,
                    target,
                    body=body,
                    headers=headers,
                    retries=False,
                    redirect=False,
                    timeout=self._timeout,
                    # Read the whole body now, which hands the connection back
                    # to the pool for the next request.
                    preload_content=True,
                )
                return response.status, response.data
            except (Urllib3Error, OSError) as exc:
                if self._should_retry(method, exc, attempt):
                    time.sleep(self.retry_backoff_seconds * (2**attempt))
                    attempt += 1
                    continue
                raise STS2ClientError(
                    f"Request failed: method={method} endpoint={endpoint} "
                    f"params={params} body={json_body} error={exc}"
                ) from exc

    def _should_retry(
        self,
        method: str,
        exc: BaseException,
        attempt: int,
    ) -> bool:
        """Return whether replaying this request is both safe and useful."""
        if method not in RETRYABLE_METHODS or attempt >= self.max_retries:
            return False
        # Only transport failures: the request never produced a response, so
        # nothing about the game changed.  An HTTP error is a real answer and
        # is not retried.
        return isinstance(exc, TRANSPORT_ERRORS)

    def _get(self, endpoint: str, params: Optional[dict[str, Any]] = None) -> Any:
        """Send a GET request to an API endpoint."""
        return self._request("GET", endpoint, params=params)

    def _post(self, endpoint: str, body: dict[str, Any]) -> Any:
        """Send a POST request to an API endpoint."""
        return self._request("POST", endpoint, json_body=body)

    def _game_endpoint(self) -> str:
        """Return the endpoint for the configured game mode."""
        return self.mode

    def action(self, action: str, **kwargs: Any) -> Any:
        """
        Generic action POST helper.

        Example:
            client.action("play_card", card_index=0, target="JAW_WORM_0")
        """
        body = {"action": action}
        body.update(kwargs)
        result = self._post(self._game_endpoint(), body)
        if self.action_delay_seconds:
            time.sleep(self.action_delay_seconds)
        return result

    # -------------------------
    # GET endpoints
    # -------------------------

    def get_state(self, format: ResponseFormat = "json") -> Any:
        """
        Read current singleplayer or multiplayer state.
        """
        return self._get(self._game_endpoint(), {"format": format})

    def get_singleplayer_state(self, format: ResponseFormat = "json") -> Any:
        return self._get("singleplayer", {"format": format})

    def get_multiplayer_state(self, format: ResponseFormat = "json") -> Any:
        return self._get("multiplayer", {"format": format})

    def sim_reset(
        self,
        character: str,
        seed: str,
        ascension: int = 0,
        mode: str = "run",
        max_fights: int = 12,
        start_act: int = 1,
        capture: bool = False,
        start_boss: bool = False,
        reseed: int | None = None,
    ) -> Any:
        """Start a run on the STS2Simulator backend.

        The simulator has no menus, so it starts a run in one request rather
        than the navigation ``ResetController`` performs.  A real client serves
        no such endpoint and answers 404.

        ``reseed`` replaces the run's combat streams before the first room: a
        snapshot otherwise restores them as they were, and every episode from
        it would play the same shuffles and the same enemy rolls.
        """
        body = {
            "character": character,
            "seed": seed,
            "ascension": ascension,
            "mode": mode,
            "max_fights": max_fights,
            "start_act": start_act,
            "capture": capture,
            "start_boss": start_boss,
        }
        if reseed is not None:
            body["reseed"] = reseed
        return self._post("sim/reset", body)

    def sim_reseed(self, seed: int) -> Any:
        """Redraw what the fight has not revealed: future shuffles, rolls, and draw order.

        Everything visible stays as it was.  A search reseeds each simulation
        from a branch point, so its statistics average over the futures a
        player could face rather than planning against the one the
        simulator's state already holds.  The response carries the state.
        """
        return self._post("sim/reseed", {"seed": seed})

    def sim_info(self) -> dict[str, Any]:
        """Read the simulator's build and counters; 404 against a real client."""
        return self._get("sim/info")

    def sim_snapshot(self) -> int:
        """Hold the simulator's current state as a branch point and return its id.

        The point stays held, and restorable any number of times, until it is
        released.  A real client has no such endpoint.
        """
        return int(self._post("sim/snapshot", {})["id"])

    def sim_restore(self, snapshot_id: int, reseed: int | None = None) -> Any:
        """Return the simulator to a branch point, then redraw its future if asked.

        The response carries the state, after the reseed when there is one.
        ``reseed`` is the seed ``sim_reseed`` takes, an unsigned 32-bit
        integer.  A search restores and reseeds once per simulation; one
        request builds and serialises the state once instead of twice.

        Contract with the simulator: ``POST /sim/restore`` with
        ``{"id": ..., "reseed": ...}`` restores, then reseeds, and answers
        ``{"status": "ok", "reseeded": <seed>, "state": ...}``.  ``reseeded``
        echoes the seed it applied.  A refused reseed is an error response, and
        the restore stands.

        An older simulator ignores the unknown field: it restores, does not
        reseed, and answers without ``reseeded``.  A search would then plan
        against the one real future and nothing would fail.  So a missing echo
        is never trusted: the client sends ``sim_reseed`` itself, logs once,
        and from then on uses two requests (restore, then reseed).
        """
        if reseed is None or self._restore_reseeds is False:
            restored = self._post("sim/restore", {"id": snapshot_id})
            return restored if reseed is None else self.sim_reseed(reseed)
        restored = self._post("sim/restore", {"id": snapshot_id, "reseed": reseed})
        echoed = restored.get("reseeded") if isinstance(restored, dict) else None
        if echoed is None:
            self._restore_reseeds = False
            logger.warning(
                "%s/sim/restore ignores 'reseed' (no 'reseeded' in its response): "
                "reseeding with a separate /sim/reseed request from now on",
                self.base_url,
            )
            return self.sim_reseed(reseed)
        if echoed != reseed:
            raise STS2ClientError(
                f"sim/restore reseeded {echoed!r}, asked for {reseed!r}",
                _state_of(restored),
            )
        self._restore_reseeds = True
        return restored

    def sim_release(self, snapshot_id: int) -> None:
        """Release a branch point the caller no longer needs."""
        self._request("DELETE", f"sim/snapshot/{snapshot_id}")

    def get_player_detail(self) -> dict[str, Any]:
        """Read run-level player detail, the only source of the master deck.

        JSON only, and independent of the singleplayer/multiplayer routing, so
        it never returns 409.  With no active run it answers HTTP 200 with
        ``in_run: false`` rather than an error status.
        """
        return self._get("player")

    def get_profile(self) -> dict[str, Any]:
        return self._get("profile")

    def get_compendium(self) -> dict[str, Any]:
        return self._get("compendium")

    def search_wiki(
        self,
        query: str,
        item_type: WikiItemType = "all",
        limit: int = 10,
    ) -> dict[str, Any]:
        return self._get(
            "wiki",
            {
                "query": query,
                "item_type": item_type,
                "limit": limit,
            },
        )

    def get_profiles(self) -> dict[str, Any]:
        return self._get("profiles")

    # -------------------------
    # Profile actions
    # -------------------------

    def switch_profile(self, profile_id: int) -> Any:
        return self._post(
            "profiles",
            {
                "action": "switch",
                "profile_id": profile_id,
            },
        )

    def delete_profile(self, profile_id: int) -> Any:
        return self._post(
            "profiles",
            {
                "action": "delete",
                "profile_id": profile_id,
            },
        )

    # -------------------------
    # Menu actions
    # -------------------------

    def menu_select(self, option: str, seed: Optional[str] = None) -> Any:
        body: dict[str, Any] = {
            "option": option,
        }
        if seed is not None:
            body["seed"] = seed
        return self.action("menu_select", **body)

    # -------------------------
    # Combat actions
    # -------------------------

    def play_card(self, card_index: int, target: Optional[str] = None) -> Any:
        body: dict[str, Any] = {
            "card_index": card_index,
        }
        if target is not None:
            body["target"] = target
        return self.action("play_card", **body)

    def use_potion(self, slot: int, target: Optional[str] = None) -> Any:
        body: dict[str, Any] = {
            "slot": slot,
        }
        if target is not None:
            body["target"] = target
        return self.action("use_potion", **body)

    def discard_potion(self, slot: int) -> Any:
        return self.action("discard_potion", slot=slot)

    def end_turn(self) -> Any:
        return self.action("end_turn")

    def undo_end_turn(self) -> Any:
        """
        Multiplayer only.
        """
        return self.action("undo_end_turn")

    def combat_select_card(self, card_index: int) -> Any:
        return self.action("combat_select_card", card_index=card_index)

    def combat_confirm_selection(self) -> Any:
        return self.action("combat_confirm_selection")

    # -------------------------
    # Reward actions
    # -------------------------

    def claim_reward(self, index: int) -> Any:
        return self.action("claim_reward", index=index)

    def select_card_reward(self, card_index: int) -> Any:
        return self.action("select_card_reward", card_index=card_index)

    def skip_card_reward(self) -> Any:
        return self.action("skip_card_reward")

    def proceed(self) -> Any:
        return self.action("proceed")

    # -------------------------
    # Event actions
    # -------------------------

    def choose_event_option(self, index: int) -> Any:
        return self.action("choose_event_option", index=index)

    def advance_dialogue(self) -> Any:
        return self.action("advance_dialogue")

    # -------------------------
    # Rest site / shop / map
    # -------------------------

    def choose_rest_option(self, index: int) -> Any:
        return self.action("choose_rest_option", index=index)

    def shop_purchase(self, index: int) -> Any:
        return self.action("shop_purchase", index=index)

    def choose_map_node(self, index: int) -> Any:
        return self.action("choose_map_node", index=index)

    # -------------------------
    # Card selection overlay
    # -------------------------

    def select_card(self, index: int) -> Any:
        return self.action("select_card", index=index)

    def confirm_selection(self) -> Any:
        return self.action("confirm_selection")

    def cancel_selection(self) -> Any:
        return self.action("cancel_selection")

    # -------------------------
    # Bundle selection
    # -------------------------

    def select_bundle(self, index: int) -> Any:
        return self.action("select_bundle", index=index)

    def confirm_bundle_selection(self) -> Any:
        return self.action("confirm_bundle_selection")

    def cancel_bundle_selection(self) -> Any:
        return self.action("cancel_bundle_selection")

    # -------------------------
    # Relic / treasure actions
    # -------------------------

    def select_relic(self, index: int) -> Any:
        return self.action("select_relic", index=index)

    def skip_relic_selection(self) -> Any:
        return self.action("skip_relic_selection")

    def claim_treasure_relic(self, index: int) -> Any:
        return self.action("claim_treasure_relic", index=index)

    # -------------------------
    # Crystal Sphere actions
    # -------------------------

    def crystal_sphere_set_tool(self, tool: Literal["big", "small"]) -> Any:
        return self.action("crystal_sphere_set_tool", tool=tool)

    def crystal_sphere_click_cell(self, x: int, y: int) -> Any:
        return self.action("crystal_sphere_click_cell", x=x, y=y)

    def crystal_sphere_proceed(self) -> Any:
        return self.action("crystal_sphere_proceed")
