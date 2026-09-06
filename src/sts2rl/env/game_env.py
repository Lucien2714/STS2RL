"""Raw STS2 environment boundary backed by the local STS2MCP API."""

from __future__ import annotations

from sts2rl.actions.dispatcher import ActionDispatcher
from sts2rl.actions.game_action import GameAction
from sts2rl.env.mcp_client import STS2Client, STS2ClientError
from sts2rl.env.reset import ResetController, ResetSpec
from sts2rl.env.state import extract_raw_state, response_state as _response_state
from sts2rl.env.types import EnvStep, RawState


class GameEnv:
    """Reset and step one raw STS2MCP single-player environment."""

    def __init__(
        self,
        base_url: str = "http://localhost:15526/api/v1",
        timeout: float = 20.0,
        client: STS2Client | None = None,
    ) -> None:
        self._owns_client = client is None
        self.client = (
            client
            if client is not None
            else STS2Client(
                base_url=base_url,
                mode="singleplayer",
                timeout=timeout,
            )
        )
        self.action_dispatcher = ActionDispatcher(self.client)
        self.reset_controller = ResetController(self.client, self.action_dispatcher)

    def reset(self, spec: ResetSpec | None = None) -> RawState:
        """Navigate menus until a run is active and return its raw state."""
        return self.reset_controller.reset(spec or ResetSpec())

    def step(self, action: GameAction) -> EnvStep:
        """Dispatch one typed action and return the raw environment result."""
        if not isinstance(action, GameAction):
            raise TypeError(f"action must be GameAction, got {type(action).__name__}")

        action_payload = action.to_dict()
        try:
            api_result = self.action_dispatcher.dispatch(action)
        except STS2ClientError as exc:
            raw_state = self._state_after(exc.state)
            return EnvStep(
                raw_state=raw_state,
                done=self._is_done(raw_state),
                info={
                    "error": str(exc),
                    "action": action_payload,
                    "action_error": True,
                },
            )

        raw_state = self._state_after(_response_state(api_result))
        info: dict = {
            "api_result": api_result,
            "action": action_payload,
            "action_error": False,
        }
        if isinstance(api_result, dict) and api_result.get("state_wait_timed_out"):
            info["state_wait_timed_out"] = True
        return EnvStep(
            raw_state=raw_state,
            done=self._is_done(raw_state),
            info=info,
        )

    def get_state(self) -> RawState:
        """Return and validate the current raw STS2MCP state."""
        return extract_raw_state(self.client.get_state())

    def get_player_detail(self) -> RawState | None:
        """Return run-level player detail, or None when no run is active.

        ``in_run: false`` arrives as a normal HTTP 200 body rather than an
        error, so it is translated to None here instead of reaching callers as
        a payload with no deck in it.
        """
        detail = self.client.get_player_detail()
        if not isinstance(detail, dict) or not detail.get("in_run"):
            return None
        return detail

    def is_end_state(self) -> bool:
        """Return whether the current backend state is game over."""
        return self._is_done(self.get_state())

    def close(self) -> None:
        """Close the STS2MCP client created by this environment."""
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "GameEnv":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _state_after(self, state: RawState | None) -> RawState:
        """Use the state an action response carried, or read it if it carried none.

        Every action response embeds the resulting state, so the common path
        costs no extra request. A response whose state could not be read
        (``state_error``) falls back to one GET.
        """
        if state is not None and state.get("state_type") is not None:
            return state
        return self.get_state()

    @staticmethod
    def _is_done(raw_state: RawState) -> bool:
        return raw_state.get("state_type") == "game_over"
