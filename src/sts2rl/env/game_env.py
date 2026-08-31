"""Raw STS2 environment boundary backed by the local STS2MCP API."""

from __future__ import annotations

import logging
from typing import Any

from sts2rl.actions.dispatcher import ActionDispatcher
from sts2rl.actions.game_action import GameAction
from sts2rl.env.mcp_client import GameCharacter, STS2Client, STS2ClientError
from sts2rl.env.reset import ResetSpec
from sts2rl.env.types import EnvStep, RawState


logger = logging.getLogger(__name__)


class GameEnv:
    """Reset and step one raw STS2MCP single-player environment."""

    MAX_RESET_TRANSITIONS = 10

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

    def reset(self, spec: ResetSpec | None = None) -> RawState:
        """Navigate menus until a run is active and return its raw state."""
        spec = spec or ResetSpec()
        character_id = GameCharacter.get(spec.character)
        if character_id is None:
            raise ValueError(f"Unsupported character index: {spec.character}")

        raw_state = self.get_state()

        if raw_state.get("state_type") == "game_over":
            raw_state = self._menu_select_state("main_menu")

        for _ in range(self.MAX_RESET_TRANSITIONS):
            if raw_state.get("state_type") != "menu":
                break

            menu_screen = raw_state.get("menu_screen")

            if menu_screen == "main":
                if "abandon_run" in raw_state.get("options", []):
                    raw_state = self._menu_select_state("abandon_run")
                    raw_state = self._menu_select_state("yes")
                raw_state = self._menu_select_state("singleplayer")
                continue

            if menu_screen == "singleplayer":
                raw_state = self._menu_select_state(
                    spec.game_mode,
                    seed=spec.run_seed if spec.uses_seed else None,
                )
                continue

            if menu_screen == "custom_run":
                raw_state = self._menu_select_state(spec.start_run_option)
                continue

            if menu_screen == "character_select":
                raw_state = self._menu_select_state(character_id)
                logger.info("Selected character: %s", character_id)
                raw_state = self._menu_select_state(spec.start_run_option)
                continue

            if menu_screen == "tutorial_prompt":
                raw_state = self._menu_select_state("no")
                continue

            raise STS2ClientError(
                f"Reset stopped at unsupported menu_screen={menu_screen!r}: "
                f"{raw_state}"
            )

        if raw_state.get("state_type") == "menu":
            raise STS2ClientError(
                f"Reset did not leave menu after {self.MAX_RESET_TRANSITIONS} "
                f"transitions: {raw_state}"
            )

        run = raw_state.get("run", {})
        logger.info(
            "Started run: act=%s floor=%s ascension=%s",
            run.get("act", 0),
            run.get("floor", 0),
            run.get("ascension", 0),
        )
        return raw_state

    def step(self, action: GameAction) -> EnvStep:
        """Dispatch one typed action and return the raw environment result."""
        if not isinstance(action, GameAction):
            raise TypeError(f"action must be GameAction, got {type(action).__name__}")

        action_payload = action.to_dict()
        try:
            api_result = self.action_dispatcher.dispatch(action)
        except STS2ClientError as exc:
            raw_state = self.get_state()
            return EnvStep(
                raw_state=raw_state,
                done=self._is_done(raw_state),
                info={
                    "error": str(exc),
                    "action": action_payload,
                    "action_error": True,
                },
            )

        raw_state = self._state_from_action_result(api_result)
        return EnvStep(
            raw_state=raw_state,
            done=self._is_done(raw_state),
            info={
                "api_result": api_result,
                "action": action_payload,
                "action_error": False,
            },
        )

    def get_state(self) -> RawState:
        """Return and validate the current raw STS2MCP state."""
        return self._state_from_action_result(self.client.get_state())

    def get_player_detail(self) -> RawState:
        """Return full local player details for the active run."""
        return self._state_from_action_result(self.client.get_player_detail())

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

    def _menu_select_state(self, option: str, seed: str | None = None) -> RawState:
        """Select one menu option and return the resulting raw state."""
        logger.info("Reset menu_select option=%s seed=%s", option, seed)
        raw_state = self._state_from_action_result(
            self.client.menu_select(option, seed=seed)
        )
        logger.info(
            "Reset menu_select option=%s -> state_type=%s menu_screen=%s",
            option,
            raw_state.get("state_type"),
            raw_state.get("menu_screen"),
        )
        return raw_state

    @staticmethod
    def _state_from_action_result(api_result: Any) -> RawState:
        """Extract a raw state from the response shapes returned by STS2MCP."""
        if isinstance(api_result, dict):
            if isinstance(api_result.get("state"), dict):
                return api_result["state"]
            if isinstance(api_result.get("raw_state"), dict):
                return api_result["raw_state"]
            if api_result.get("state_type") is not None:
                return api_result

        raise STS2ClientError(f"Response did not include a game state: {api_result}")

    @staticmethod
    def _is_done(raw_state: RawState) -> bool:
        return raw_state.get("state_type") == "game_over"
