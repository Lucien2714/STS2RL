"""Reset configuration and menu navigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sts2rl.actions.dispatcher import ActionDispatcher
from sts2rl.actions.game_action import MenuSelectAction
from sts2rl.env.mcp_client import GameCharacter, STS2ClientError
from sts2rl.env.state import extract_raw_state
from sts2rl.env.types import RawState


@dataclass(frozen=True)
class ResetSpec:
    """Describe how an environment reset should start a run."""

    character: int = 0
    game_mode: str = "standard"
    run_seed: str | None = None
    start_run_option: str = "confirm"
    allow_active_run: bool = False

    def __post_init__(self) -> None:
        if self.game_mode not in {"standard", "custom", "daily"}:
            raise ValueError(f"Unsupported game mode: {self.game_mode!r}")
        if self.start_run_option not in {"confirm", "embark"}:
            raise ValueError(
                f"Unsupported start-run option: {self.start_run_option!r}"
            )

    @property
    def uses_seed(self) -> bool:
        """Return whether reset should pass a seed to STS2MCP."""
        return self.game_mode in {"custom", "daily"} and self.run_seed is not None


def enabled_option_names(raw_state: RawState) -> set[str]:
    """Return enabled menu option names normalized for comparisons."""
    names: set[str] = set()
    options = raw_state.get("options", [])
    if not isinstance(options, list):
        return names

    for option in options:
        if isinstance(option, str):
            names.add(option.casefold())
            continue
        if not isinstance(option, dict) or option.get("enabled") is False:
            continue
        name = option.get("name") or option.get("option") or option.get("id")
        if isinstance(name, str):
            names.add(name.casefold())
    return names


class ResetController:
    """Drive the STS2MCP menu state machine until a run becomes active."""

    MAX_TRANSITIONS = 10

    def __init__(
        self,
        client: Any,
        dispatcher: ActionDispatcher,
        max_transitions: int = MAX_TRANSITIONS,
    ) -> None:
        if max_transitions < 1:
            raise ValueError("max_transitions must be at least 1")
        self.client = client
        self.dispatcher = dispatcher
        self.max_transitions = max_transitions

    def reset(self, spec: ResetSpec) -> RawState:
        """Start a new run, or explicitly reuse an active run when permitted."""
        character_id = GameCharacter.get(spec.character)
        if character_id is None:
            raise ValueError(f"Unsupported character index: {spec.character}")

        raw_state = extract_raw_state(self.client.get_state())
        state_type = raw_state.get("state_type")
        if state_type not in {"menu", "game_over"}:
            if spec.allow_active_run:
                return raw_state
            raise STS2ClientError(
                "Cannot start a fresh run while another run is active; "
                "pass ResetSpec(allow_active_run=True) to reuse it"
            )

        seen_states: set[tuple[Any, ...]] = set()
        for _ in range(self.max_transitions):
            state_type = raw_state.get("state_type")
            if state_type not in {"menu", "game_over"}:
                return raw_state

            signature = self._menu_signature(raw_state)
            if signature in seen_states:
                raise STS2ClientError(
                    f"Reset menu stopped making progress: {raw_state}"
                )
            seen_states.add(signature)

            if state_type == "game_over":
                raw_state = self._select(raw_state, "main_menu")
            else:
                raw_state = self._advance_menu(raw_state, spec, character_id)

        raise STS2ClientError(
            f"Reset did not start a run after {self.max_transitions} "
            f"transitions: {raw_state}"
        )

    def _advance_menu(
        self,
        raw_state: RawState,
        spec: ResetSpec,
        character_id: str,
    ) -> RawState:
        menu_screen = raw_state.get("menu_screen")

        if menu_screen == "main":
            options = enabled_option_names(raw_state)
            if "abandon_run" in options:
                return self._select(raw_state, "abandon_run")
            return self._select(raw_state, "singleplayer")

        if menu_screen == "popup":
            options = enabled_option_names(raw_state)
            for option in ("confirm", "yes"):
                if option in options:
                    return self._select(raw_state, option)
            raise STS2ClientError(
                f"Reset cannot confirm the current popup: {raw_state}"
            )

        if menu_screen == "singleplayer":
            return self._select(
                raw_state,
                spec.game_mode,
                seed=spec.run_seed if spec.uses_seed else None,
            )

        if menu_screen == "custom_run":
            if self._needs_character_selection(
                raw_state, character_id, missing_means_selected=True
            ):
                return self._select(raw_state, character_id)
            return self._select(
                raw_state,
                spec.start_run_option,
                seed=spec.run_seed if spec.uses_seed else None,
            )

        if menu_screen == "character_select":
            if self._needs_character_selection(
                raw_state, character_id, missing_means_selected=False
            ):
                return self._select(raw_state, character_id)
            return self._select(raw_state, spec.start_run_option)

        if menu_screen == "tutorial_prompt":
            return self._select(raw_state, "no")

        raise STS2ClientError(
            f"Reset stopped at unsupported menu_screen={menu_screen!r}: "
            f"{raw_state}"
        )

    def _select(
        self,
        raw_state: RawState,
        option: str,
        seed: str | None = None,
    ) -> RawState:
        options = enabled_option_names(raw_state)
        advertised_options = raw_state.get("options")
        if (
            isinstance(advertised_options, list)
            and advertised_options
            and option.casefold() not in options
        ):
            raise STS2ClientError(
                f"Menu option {option!r} is not enabled; available={sorted(options)}"
            )
        response = self.dispatcher.dispatch(MenuSelectAction(option, seed=seed))
        return extract_raw_state(response)

    @staticmethod
    def _needs_character_selection(
        raw_state: RawState,
        character_id: str,
        *,
        missing_means_selected: bool,
    ) -> bool:
        characters = raw_state.get("characters")
        if not isinstance(characters, list) or not characters:
            return not missing_means_selected

        target = character_id.casefold()
        for character in characters:
            if not isinstance(character, dict):
                continue
            identifier = character.get("id") or character.get("name")
            if isinstance(identifier, str) and identifier.casefold() == target:
                return not bool(character.get("selected"))
        return True

    @staticmethod
    def _menu_signature(raw_state: RawState) -> tuple[Any, ...]:
        characters = raw_state.get("characters", [])
        character_state = tuple(
            (
                str(character.get("id") or character.get("name")),
                bool(character.get("selected")),
            )
            for character in characters
            if isinstance(character, dict)
        ) if isinstance(characters, list) else ()
        return (
            raw_state.get("state_type"),
            raw_state.get("menu_screen"),
            tuple(sorted(enabled_option_names(raw_state))),
            character_state,
        )
