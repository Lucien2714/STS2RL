"""Raw STS2 environment boundary backed by the local STS2MCP API."""

from __future__ import annotations

from sts2rl.actions.dispatcher import ActionDispatcher
from sts2rl.actions.game_action import GameAction
from sts2rl.env.mcp_client import (
    DEFAULT_ACTION_DELAY_SECONDS,
    STS2Client,
    STS2ClientError,
)
from sts2rl.env.reset import ResetController, ResetSpec
from sts2rl.env.state import extract_raw_state, response_state as _response_state
from sts2rl.env.types import EnvStep, RawState


# Resolving a card reward is not finished until the screen is left, because
# declining one puts it straight back with nothing to say it was refused.
CARD_REWARD_DECISIONS = frozenset({"select_card_reward", "skip_card_reward"})


def _selection_is_complete(prompt: dict) -> bool:
    """Return whether a selection prompt is full and ready to be confirmed.

    Only a prompt that reports how many it wants can be known to be full.  A
    "choose" screen picks immediately and leaves ``selected_count`` at zero, so
    it never looks complete and is correctly left alone.
    """
    if prompt.get("can_confirm") is not True:
        return False
    selected = prompt.get("selected_count")
    maximum = prompt.get("max_select")
    if isinstance(selected, bool) or not isinstance(selected, int):
        return False
    if isinstance(maximum, bool) or not isinstance(maximum, int):
        return False
    return selected >= maximum


class GameEnv:
    """Reset and step one raw STS2MCP single-player environment."""

    def __init__(
        self,
        base_url: str = "http://localhost:15526/api/v1",
        timeout: float = 20.0,
        client: STS2Client | None = None,
        action_delay_seconds: float = DEFAULT_ACTION_DELAY_SECONDS,
    ) -> None:
        self._owns_client = client is None
        self.client = (
            client
            if client is not None
            else STS2Client(
                base_url=base_url,
                mode="singleplayer",
                timeout=timeout,
                action_delay_seconds=action_delay_seconds,
            )
        )
        self.action_dispatcher = ActionDispatcher(self.client)
        self.reset_controller = ResetController(self.client, self.action_dispatcher)

    def reset(self, spec: ResetSpec | None = None) -> RawState:
        """Navigate menus until a run is active and return its raw state."""
        return self.reset_controller.reset(spec or ResetSpec())

    @property
    def reused_active_run(self) -> bool:
        """Whether the last reset joined a run instead of starting one."""
        return self.reset_controller.reused_active_run

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
        raw_state, finished = self._finish_resolved_screen(action, raw_state)
        info: dict = {
            "api_result": api_result,
            "action": action_payload,
            "action_error": False,
        }
        if finished:
            info["auto_proceeded"] = True
        if isinstance(api_result, dict) and api_result.get("state_wait_timed_out"):
            info["state_wait_timed_out"] = True
        return EnvStep(
            raw_state=raw_state,
            done=self._is_done(raw_state),
            info=info,
        )

    def _finish_resolved_screen(
        self,
        action: GameAction,
        raw_state: RawState,
    ) -> tuple[RawState, bool]:
        """Complete a choice the game leaves half-made.

        Two screens hand back a decision that is not finished yet, and in both
        cases the unfinished half is a loop rather than a choice:

        * Resolving a card reward lands back on ``rewards`` with the card still
          listed and nothing marking it refused, so an agent offered that
          screen again is offered the same card again.  The screen is left.
        * Picking a bundle opens a preview whose only moves are confirming and
          cancelling, and cancelling returns to the same list of bundles.  The
          pick is confirmed.
        * Filling a card-selection prompt leaves confirming and cancelling, and
          cancelling puts every card back for the same deterministic policy to
          pick again.  "Choose a card to upgrade" loops forever that way.  The
          selection is confirmed once the prompt will take no more cards.

        This is the one place the environment sends a request the agent did not
        choose.  It is safe because nothing claimable is left behind: the
        reward screen withholds the card claim until every gold and relic has
        been taken, and a bundle preview offers no third option.  A card reward
        reached straight from an event does not land back on ``rewards`` and is
        left alone.

        **A potion against a full belt is the one thing this can abandon**, and
        it is abandoned knowingly.  ``_reward_actions`` drops such a potion from
        the outright set rather than offering a claim the game would refuse,
        which promotes the card while the potion is still listed; leaving then
        takes the potion with it.  Two things make that acceptable.  The agent
        is not trapped into it -- discarding a held potion frees a slot, which
        moves the reward back into the outright set where it is claimed before
        the card -- and the reward model scores no potions either way, so the
        loss it prices is zero.  A traced human made the same call twice,
        walking away from a Fire Potion and a Speed Potion with a full belt.
        Gold and relics are never skipped, so they can never be lost this way.
        """
        if action.action_type in CARD_REWARD_DECISIONS:
            if raw_state.get("state_type") != "rewards":
                return raw_state, False
            follow_up = self.client.proceed
        elif action.action_type == "select_card":
            prompt = raw_state.get("card_select")
            if raw_state.get("state_type") != "card_select" or not isinstance(
                prompt, dict
            ):
                return raw_state, False
            if not _selection_is_complete(prompt):
                return raw_state, False
            follow_up = self.client.confirm_selection
        elif action.action_type == "combat_select_card":
            prompt = raw_state.get("hand_select")
            if raw_state.get("state_type") != "hand_select" or not isinstance(
                prompt, dict
            ):
                return raw_state, False
            if not _selection_is_complete(prompt):
                return raw_state, False
            follow_up = self.client.combat_confirm_selection
        elif action.action_type == "select_bundle":
            bundle = raw_state.get("bundle_select")
            if raw_state.get("state_type") != "bundle_select" or not isinstance(
                bundle, dict
            ):
                return raw_state, False
            if bundle.get("can_confirm") is not True:
                return raw_state, False
            follow_up = self.client.confirm_bundle_selection
        else:
            return raw_state, False

        try:
            response = follow_up()
        except STS2ClientError:
            # Nothing is lost by staying: the screen is still legal to act on.
            return raw_state, False
        return self._state_after(_response_state(response)), True

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
