"""The default action for states that no agent decides.

:func:`default_action` answers "what should we send when no agent is choosing?"
by asking the screen's own action space for a legal candidate, rather than
assuming ``proceed``. ``proceed`` is accepted on only a handful of screens
(see :data:`~sts2rl.action_spaces.base.LEGAL_ACTION_TYPES`) — notably **not** in
combat, on the map, on ``card_reward``, or during events — so it is not a safe
universal filler.

When a screen genuinely has no legal action right now — combat outside the
player's play phase is the common case — the answer is
:data:`~sts2rl.action_spaces.base.REFRESH_STATE_ACTION`, a no-op that re-reads
state instead of posting a move the API would reject.

The legality table itself lives in :mod:`sts2rl.action_spaces.base` so the
per-screen spaces can consult it without importing this module (which imports
them). Torch-free by package rule.
"""

from __future__ import annotations

from sts2rl.action_spaces.base import (
    PROCEED_STATE_TYPES,
    REFRESH_STATE_ACTION,
    is_legal_action_type,
)
from sts2rl.action_spaces.battle import BattleActionSpace
from sts2rl.action_spaces.event import EventActionSpace
from sts2rl.action_spaces.map import MapActionSpace
from sts2rl.action_spaces.rest import RestActionSpace
from sts2rl.action_spaces.reward import RewardActionSpace
from sts2rl.action_spaces.shop import ShopActionSpace

_BATTLE_STATE_TYPES = frozenset({"monster", "elite", "boss", "hand_select"})

# state_type -> the action space that enumerates its legal candidates. `card_select`
# is deliberately absent: it is battle-scoped or event-scoped depending on
# raw_state["in_battle"], resolved in _action_space_for.
_ACTION_SPACES_BY_STATE_TYPE = {
    "monster": BattleActionSpace,
    "elite": BattleActionSpace,
    "boss": BattleActionSpace,
    "hand_select": BattleActionSpace,
    "map": MapActionSpace,
    "rewards": RewardActionSpace,
    "card_reward": RewardActionSpace,
    "treasure": RewardActionSpace,
    "shop": ShopActionSpace,
    "fake_merchant": ShopActionSpace,
    "rest": RestActionSpace,
    "rest_site": RestActionSpace,
    "event": EventActionSpace,
}

# One instance per space is enough: action spaces are stateless.
_SPACE_INSTANCES: dict[type, object] = {}


def can_proceed(raw_state: dict) -> bool:
    """Return whether ``proceed`` is an accepted action for this state."""
    return raw_state.get("state_type") in PROCEED_STATE_TYPES


def _action_space_for(raw_state: dict):
    """Return the action space controlling a state, or None if unmapped."""
    state_type = raw_state.get("state_type")
    if state_type == "card_select":
        space_class = BattleActionSpace if raw_state.get("in_battle") else EventActionSpace
    else:
        space_class = _ACTION_SPACES_BY_STATE_TYPE.get(str(state_type))
    if space_class is None:
        return None
    return _SPACE_INSTANCES.setdefault(space_class, space_class())


def default_action(raw_state: dict) -> dict:
    """Return a legal action for a state that no agent is choosing for.

    Preference order: the first legal candidate the screen's own action space
    enumerates, then ``proceed`` where the screen accepts it, then a state
    refresh. The last case covers combat outside the player's play phase, where
    the correct move is to wait for the server to settle rather than to act.
    """
    state_type = raw_state.get("state_type")

    space = _action_space_for(raw_state)
    if space is not None:
        for candidate in space.candidates(raw_state):
            action = dict(candidate["action"])
            if is_legal_action_type(state_type, action.get("type")):
                return action

    if state_type in PROCEED_STATE_TYPES:
        return {"type": "proceed", "action_key": "proceed"}

    return dict(REFRESH_STATE_ACTION)


def is_waiting_state(raw_state: dict) -> bool:
    """Return whether a state has nothing legal to do but is not terminal.

    Used by the step loop to fold a wait into a cheap state re-read instead of
    counting it as a real transition.
    """
    return (
        raw_state.get("state_type") in _BATTLE_STATE_TYPES
        and default_action(raw_state)["type"] == REFRESH_STATE_ACTION["type"]
    )
