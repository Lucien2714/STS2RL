"""Reset configuration and menu navigation."""

from __future__ import annotations

from dataclasses import dataclass
import time
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
    ascension: int | None = None
    modifiers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.game_mode not in {"standard", "custom", "daily"}:
            raise ValueError(f"Unsupported game mode: {self.game_mode!r}")
        object.__setattr__(self, "modifiers", tuple(self.modifiers))
        if any(not isinstance(key, str) or not key for key in self.modifiers):
            raise ValueError("modifiers must be non-empty strings")
        if len(set(self.modifiers)) != len(self.modifiers):
            raise ValueError("modifiers must not repeat")
        if self.modifiers and self.game_mode != "custom":
            raise ValueError("modifiers are only offered by the custom-run screen")
        if self.start_run_option not in {"confirm", "embark"}:
            raise ValueError(
                f"Unsupported start-run option: {self.start_run_option!r}"
            )
        if self.ascension is not None and self.ascension < 0:
            raise ValueError("ascension must not be negative")

    @property
    def uses_seed(self) -> bool:
        """Return whether reset should pass a seed to STS2MCP."""
        return self.game_mode in {"custom", "daily"} and self.run_seed is not None

    @property
    def enforces_modifiers(self) -> bool:
        """Return whether reset must reconcile the custom-run modifier set.

        An empty tuple is a real setting -- "every modifier off" -- so custom
        runs always reconcile.  Only the custom-run screen offers modifiers at
        all, so no other mode does.
        """
        return self.game_mode == "custom"


def enabled_option_names(raw_state: RawState) -> set[str]:
    """Return enabled menu option names normalized for comparisons.

    Some screens advertise their options inside the block named after the
    state type rather than at the top level; ``game_over`` is one.
    """
    names: set[str] = set()
    options = raw_state.get("options")
    if options is None:
        nested = raw_state.get(str(raw_state.get("state_type")))
        if isinstance(nested, dict):
            options = nested.get("options")
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


MENU_STATE_TYPES = frozenset({"menu", "game_over"})

# The API reports "unknown" while a screen is still loading. It is neither a
# menu to navigate nor a state any agent can act in, so reset waits it out
# rather than handing it to the caller.
TRANSITIONAL_STATE_TYPES = frozenset({"unknown"})


def is_run_state(raw_state: RawState) -> bool:
    """Return whether this state is a settled, playable part of a run."""
    state_type = raw_state.get("state_type")
    return (
        state_type not in MENU_STATE_TYPES
        and state_type not in TRANSITIONAL_STATE_TYPES
    )


def _selected_modifiers(raw_state: RawState) -> set[str] | None:
    """Return the ticked modifier keys, or None when the screen reports none.

    A build that does not advertise ``selected`` tells us nothing about the
    modifier set, and guessing would start runs under silently wrong rules.
    """
    selected = raw_state.get("selected")
    if not isinstance(selected, dict):
        return None
    modifiers = selected.get("modifiers")
    if not isinstance(modifiers, list):
        return None
    return {str(key) for key in modifiers}


def _ascension_level(raw_state: RawState) -> int | None:
    """Return the ascension level the character screen currently shows."""
    ascension = raw_state.get("ascension")
    if isinstance(ascension, dict):
        level = ascension.get("level")
    else:
        level = ascension
    if isinstance(level, bool) or not isinstance(level, int):
        return None
    return level


def selected_character_id(raw_state: RawState) -> str | None:
    """Return the currently selected character id, or None if none is reported.

    The menu reports the selection in one of two shapes depending on screen and
    mod build: a top-level ``selected`` object, or a ``selected`` flag on each
    character entry.  Reading only the per-entry flag makes an already-selected
    character look unselected, so reset re-sends the same option forever.
    """
    selected = raw_state.get("selected")
    if isinstance(selected, dict):
        identifier = selected.get("character") or selected.get("id")
        if isinstance(identifier, str) and identifier:
            return identifier
    elif isinstance(selected, str) and selected:
        return selected

    characters = raw_state.get("characters")
    if isinstance(characters, list):
        for character in characters:
            if isinstance(character, dict) and character.get("selected"):
                identifier = character.get("id") or character.get("name")
                if isinstance(identifier, str) and identifier:
                    return identifier
    return None


class ResetController:
    """Drive the STS2MCP menu state machine until a run becomes active."""

    MAX_TRANSITIONS = 16
    START_POLL_ATTEMPTS = 20
    START_POLL_SECONDS = 0.25
    MAX_ASCENSION_STEPS = 30
    MAX_MODIFIER_STEPS = 40

    def __init__(
        self,
        client: Any,
        dispatcher: ActionDispatcher,
        max_transitions: int = MAX_TRANSITIONS,
        start_poll_attempts: int = START_POLL_ATTEMPTS,
        start_poll_seconds: float = START_POLL_SECONDS,
        max_ascension_steps: int = MAX_ASCENSION_STEPS,
        max_modifier_steps: int = MAX_MODIFIER_STEPS,
    ) -> None:
        if max_transitions < 1:
            raise ValueError("max_transitions must be at least 1")
        self.client = client
        self.dispatcher = dispatcher
        self.max_transitions = max_transitions
        self.start_poll_attempts = start_poll_attempts
        self.start_poll_seconds = start_poll_seconds
        self.max_ascension_steps = max_ascension_steps
        self.max_modifier_steps = max_modifier_steps
        # Whether the most recent reset joined a run already in progress
        # instead of starting the one it was asked for.  A reused run ignores
        # the requested seed, so a seeded experiment has to be able to see it.
        self.reused_active_run = False

    def reset(self, spec: ResetSpec) -> RawState:
        """Start a new run, or explicitly reuse an active run when permitted."""
        character_id = GameCharacter.get(spec.character)
        if character_id is None:
            raise ValueError(f"Unsupported character index: {spec.character}")

        self.reused_active_run = False
        raw_state = extract_raw_state(self.client.get_state())
        if raw_state.get("state_type") in TRANSITIONAL_STATE_TYPES:
            raw_state = self._await_stable_state(raw_state)
        if raw_state.get("state_type") not in MENU_STATE_TYPES:
            if spec.allow_active_run:
                self.reused_active_run = True
                return raw_state
            raise STS2ClientError(
                "Cannot start a fresh run while another run is active; "
                "pass ResetSpec(allow_active_run=True) to reuse it"
            )

        seen_states: set[tuple[Any, ...]] = set()
        for _ in range(self.max_transitions):
            state_type = raw_state.get("state_type")
            if is_run_state(raw_state):
                return raw_state
            if state_type in TRANSITIONAL_STATE_TYPES:
                raw_state = self._await_stable_state(raw_state)
                continue

            signature = self._menu_signature(raw_state)
            if signature in seen_states:
                # A screen we have seen before is not proof of a stall: the
                # menu can simply not have caught up.  Confirming an abandoned
                # run returns the pre-abandon main menu often enough to kill a
                # long training run, and that screen is one we just came from.
                # Only a repeat that survives settling is a real stall.
                raw_state = self._await_stable_state(raw_state)
                signature = self._menu_signature(raw_state)
                if signature in seen_states:
                    raise STS2ClientError(
                        f"Reset menu stopped making progress: {raw_state}"
                    )
            seen_states.add(signature)

            if state_type == "game_over":
                advanced = self._leave_game_over(raw_state)
            else:
                advanced = self._advance_menu(raw_state, spec, character_id)

            # A menu that reports no change may simply not have applied the
            # transition yet, so give it a moment before calling it a stall.
            if (
                not is_run_state(advanced)
                and self._menu_signature(advanced) == signature
            ):
                advanced = self._await_stable_state(advanced)
            raw_state = advanced

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
            # Only daily accepts a seed at the submenu; a custom run's seed
            # belongs to confirm/embark on the custom-run screen itself.
            seeded_here = spec.uses_seed and spec.game_mode == "daily"
            return self._select(
                raw_state,
                spec.game_mode,
                seed=spec.run_seed if seeded_here else None,
            )

        if menu_screen in {"custom_run", "character_select"}:
            if self._needs_character_selection(
                raw_state,
                character_id,
                missing_means_selected=menu_screen == "custom_run",
            ):
                return self._select(raw_state, character_id)
            if menu_screen == "custom_run" and spec.enforces_modifiers:
                raw_state = self._apply_modifiers(raw_state, spec.modifiers)
            if spec.ascension is not None:
                raw_state = self._apply_ascension(raw_state, spec.ascension)
            return self._start_run(
                raw_state,
                spec,
                seed=spec.run_seed if spec.uses_seed else None,
            )

        if menu_screen == "tutorial_prompt":
            return self._select(raw_state, "no")

        raise STS2ClientError(
            f"Reset stopped at unsupported menu_screen={menu_screen!r}: "
            f"{raw_state}"
        )

    def _apply_ascension(self, raw_state: RawState, target: int) -> RawState:
        """Step the ascension level toward the target one press at a time.

        ``ascension_up`` / ``ascension_down`` each move by one and are only
        advertised while that direction is still available, so the loop stops
        as soon as the level matches or the screen stops offering the button.
        """
        for _ in range(self.max_ascension_steps):
            level = _ascension_level(raw_state)
            if level is None or level == target:
                return raw_state
            option = "ascension_up" if level < target else "ascension_down"
            if option not in enabled_option_names(raw_state):
                return raw_state
            raw_state = self._select(raw_state, option)
        return raw_state

    def _apply_modifiers(
        self,
        raw_state: RawState,
        desired: tuple[str, ...],
    ) -> RawState:
        """Toggle the custom-run screen until exactly ``desired`` is ticked.

        The screen remembers what was ticked last, including what a human
        ticked by hand, so the modifier set is state we must reconcile rather
        than assume.  A run started under the wrong modifiers is still a valid
        run, which is exactly why this has to be checked: nothing downstream
        would notice.

        Modifiers come in mutually exclusive groups, so enabling one can
        disable another.  Each toggle therefore changes one key and re-reads
        the authoritative set rather than predicting the result.
        """
        wanted = set(desired)
        for _ in range(self.max_modifier_steps):
            current = _selected_modifiers(raw_state)
            if current is None:
                return raw_state
            difference = current.symmetric_difference(wanted)
            if not difference:
                return raw_state
            key = sorted(difference)[0]
            raw_state = self._toggle_modifier(raw_state, key)
        raise STS2ClientError(
            f"Reset could not settle the custom-run modifiers on {sorted(wanted)}; "
            f"the screen still reports {sorted(_selected_modifiers(raw_state) or ())}"
        )

    def _toggle_modifier(self, raw_state: RawState, key: str) -> RawState:
        """Flip one modifier and return a state carrying the resulting set.

        A toggle response reports ``modifiers`` instead of a full state, so the
        authoritative set is spliced into the screen we already have; only a
        response that omits it costs an extra read.
        """
        option = f"modifier_{key}"
        options = enabled_option_names(raw_state)
        if options and option.casefold() not in options:
            raise STS2ClientError(
                f"Custom run does not offer {option!r}; available={sorted(options)}"
            )
        response = self.dispatcher.dispatch(MenuSelectAction(option))
        if isinstance(response, dict) and isinstance(response.get("modifiers"), list):
            updated = dict(raw_state)
            selected = dict(updated.get("selected") or {})
            selected["modifiers"] = list(response["modifiers"])
            updated["selected"] = selected
            return updated
        return extract_raw_state(self.client.get_state())

    def _leave_game_over(self, raw_state: RawState) -> RawState:
        """Return to the main menu, tolerating a screen that already left.

        The game-over screen dismisses itself, so the request can arrive after
        the menu has moved on and be rejected as an unknown option.  That is
        the outcome we wanted, not a failure.
        """
        try:
            return self._select(raw_state, "main_menu")
        except STS2ClientError:
            return self._await_stable_state(raw_state)

    def _start_run(
        self,
        raw_state: RawState,
        spec: ResetSpec,
        *,
        seed: str | None = None,
    ) -> RawState:
        """Press the start button, then wait for the run to actually begin.

        Starting is not instantaneous: the menu first disables every control
        while the run loads, so the screen briefly advertises character buttons
        and nothing else.  Pressing start again there fails, so a screen that no
        longer offers it is treated as already-started and waited on instead.
        """
        option = self._start_option(raw_state, spec.start_run_option)
        if option is None:
            return self._await_stable_state(raw_state)
        started = self._select(raw_state, option, seed=seed)
        if is_run_state(started):
            return started
        return self._await_stable_state(started)

    @staticmethod
    def _start_option(raw_state: RawState, preferred: str) -> str | None:
        """Pick a usable start button, or None when the screen offers none.

        A menu that advertises no options at all tells us nothing, so the
        preferred button is sent as before; only an explicit option list that
        omits every start button means the run is already on its way.
        """
        advertised = raw_state.get("options")
        if not isinstance(advertised, list) or not advertised:
            return preferred
        options = enabled_option_names(raw_state)
        return next(
            (
                candidate
                for candidate in (preferred, "confirm", "embark")
                if candidate.casefold() in options
            ),
            None,
        )

    def _await_stable_state(self, raw_state: RawState) -> RawState:
        """Poll until the game settles into a run, or into a different menu.

        Both the start button and the first screen of a run take a moment to
        land, and in between the API reports either an unchanged menu or a
        transitional state. Returning either would hand the caller something
        no agent can act in.
        """
        for _ in range(self.start_poll_attempts):
            time.sleep(self.start_poll_seconds)
            polled = extract_raw_state(self.client.get_state())
            if is_run_state(polled):
                return polled
            if polled.get("state_type") in MENU_STATE_TYPES and self._menu_signature(
                polled
            ) != self._menu_signature(raw_state):
                return polled
        return raw_state

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
        selected = selected_character_id(raw_state)
        if selected is not None:
            return selected.casefold() != target
        return not missing_means_selected

    @staticmethod
    def _menu_signature(raw_state: RawState) -> tuple[Any, ...]:
        return (
            raw_state.get("state_type"),
            raw_state.get("menu_screen"),
            tuple(sorted(enabled_option_names(raw_state))),
            selected_character_id(raw_state),
            _ascension_level(raw_state),
        )
