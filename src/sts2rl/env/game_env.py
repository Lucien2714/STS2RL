"""Thin STS2 environment adapter for communicating with the local STS2MCP API."""

import logging
from collections.abc import Mapping
from typing import Any

from sts2rl.actions.dispatcher import ActionDispatcher
from sts2rl.actions.game_action import GameAction
from sts2rl.env.mcp_client import STS2Client, STS2ClientError, GameCharacter

logger = logging.getLogger(__name__)


class GameEnv:
    """Manage reset flow and action dispatch for a running STS2MCP game."""

    def __init__(
        self,
        character: int = 0,
        seed: int | None = None,
        base_url: str = "http://localhost:15526/api/v1",
        timeout: float = 20.0,
        game_mode: str = "standard",
        run_seed: str | None = None,
        start_run_option: str = "confirm",
    ) -> None:
        self.character = character
        self.game_mode = game_mode
        self.run_seed = run_seed
        self.start_run_option = start_run_option
        self.client = STS2Client(
            base_url=base_url,
            mode="singleplayer",
            timeout=timeout,
        )
        self.action_dispatcher = ActionDispatcher(self.client)

    def reset(self, run_seed: str | None = None) -> dict:
        """Navigate menus until a run state is active and return raw game state."""
        if run_seed is not None:
            self.run_seed = run_seed

        raw_state = self.client.get_state()

        if raw_state.get("state_type") == "game_over":
            raw_state = self._menu_select_state("main_menu")

        for _ in range(10):
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
                    self.game_mode,
                    seed=self.run_seed if self.game_mode in {"custom", "daily"} else None,
                )
                continue

            if menu_screen == "custom_run":
                raw_state = self._menu_select_state(self.start_run_option)
                continue

            if menu_screen == "character_select":
                character_id = GameCharacter.get(self.character, "IRONCLAD")
                raw_state = self._menu_select_state(character_id)
                logger.info("Selected character: %s", character_id)
                raw_state = self._menu_select_state(self.start_run_option)
                continue

            if menu_screen == "tutorial_prompt":
                raw_state = self._menu_select_state("no")
                continue

            raise STS2ClientError(
                f"Reset stopped at unsupported menu_screen={menu_screen}: {raw_state}"
            )

        if raw_state.get("state_type") == "menu":
            raise STS2ClientError(
                f"Reset did not leave menu after 10 transitions: {raw_state}"
            )

        run = raw_state.get("run", {})
        logger.info(
            "Started run: act=%s floor=%s ascension=%s",
            run.get("act", 0),
            run.get("floor", 0),
            run.get("ascension", 0),
        )
        return raw_state

    def step(
        self,
        action: GameAction | Mapping[str, Any],
    ) -> tuple[dict, bool, dict]:
        """Dispatch a typed action and return raw state, done flag, and API info.

        Mapping input remains accepted while agents are migrated to GameAction.
        The environment converts it once at this boundary; ActionDispatcher only
        receives the typed representation.
        """
        action_payload: object = action
        try:
            game_action = GameAction.coerce(action)
            action_payload = game_action.to_dict()
            api_result = self.action_dispatcher.dispatch(game_action)
        except Exception as exc:
            raw_state = self.client.get_state()
            done = raw_state.get("state_type") == "game_over"
            info = {
                "error": str(exc),
                "raw_state": raw_state,
                "action": action_payload,
                "action_error": True,
            }
            return raw_state, bool(done), info

        raw_state = self._state_from_action_result(api_result)
        done = raw_state.get("state_type") == "game_over"
        info = {
            "api_result": api_result,
            "raw_state": raw_state,
            "action": action_payload,
            "action_error": False,
        }
        return raw_state, bool(done), info

    def get_state(self) -> dict:
        """Return the current raw game state from STS2MCP."""
        return self.client.get_state()

    def get_player_detail(self) -> dict:
        """Return full local player details for the active run."""
        return self.client.get_player_detail()

    def _state_from_action_result(self, api_result) -> dict:
        """Extract the raw state payload from supported STS2MCP response shapes."""
        if isinstance(api_result, dict):
            if isinstance(api_result.get("state"), dict):
                return api_result["state"]
            if isinstance(api_result.get("raw_state"), dict):
                return api_result["raw_state"]
            if api_result.get("state_type") is not None:
                return api_result

        raise STS2ClientError(f"Action response did not include state: {api_result}")

    def _menu_select_state(self, option: str, seed: str | None = None) -> dict:
        """Select a menu option and return the resulting raw state."""
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

    def is_end_state(self) -> bool:
        """Return whether the current backend state is game_over."""
        return self.get_state().get("state_type", False) == "game_over"

    def close(self) -> None:
        """Close environment resources when needed."""
        pass


Game = GameEnv
