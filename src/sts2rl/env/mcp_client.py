"""HTTP client for the STS2MCP local API.

Generated from the STS2MCP API documentation:
https://github.com/Gennadiyev/STS2MCP/blob/main/docs/raw-full.md
"""


from __future__ import annotations

import time
from typing import Any, Literal, Optional

import requests


GameMode = Literal["singleplayer", "multiplayer"]
ResponseFormat = Literal["json", "markdown"]
WikiItemType = Literal["all", "card", "relic"]
GameCharacter = {0: "IRONCLAD", 1: "SILENT", 2: "REGENT", 3: "NECROBINDER", 4: "DEFECT"}

# Pause after every accepted action POST.
#
# The mod settles the game before it captures the state it returns, so this is
# not what makes that state correct.  It paces the *next* request instead: an
# action sent while the game is still resolving the previous one is the case
# the settle logic does not cover.  A rejected action changed nothing, so it is
# not followed by a pause.
DEFAULT_ACTION_DELAY_SECONDS = 0.1


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
        session: requests.Session | None = None,
        action_delay_seconds: float = DEFAULT_ACTION_DELAY_SECONDS,
    ) -> None:
        if action_delay_seconds < 0:
            raise ValueError("action_delay_seconds must not be negative")
        self.base_url = base_url.rstrip("/")
        self.mode = mode
        self.timeout = timeout
        self.action_delay_seconds = action_delay_seconds
        self._owns_session = session is None
        self.session = session if session is not None else requests.Session()

    def close(self) -> None:
        """Close the internally created HTTP session.

        An injected session remains owned by its caller and is deliberately not
        closed here.
        """
        if self._owns_session:
            self.session.close()

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

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
    ) -> Any:
        try:
            response = self.session.request(
                method=method,
                url=self._url(endpoint),
                params=params,
                json=json_body,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise STS2ClientError(
                f"Request failed: method={method} endpoint={endpoint} "
                f"params={params} body={json_body} error={exc}"
            ) from exc

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
            raise STS2ClientError(
                data.get("error", "Unknown API error"),
                _state_of(data),
            )

        return data

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
