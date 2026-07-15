"""Legal-action enumeration per screen (torch-free game rules).

An :class:`~sts2rl.action_spaces.base.ActionSpace` answers "which actions are
legal right now, and what is each one's stable key?" — pure game-rule logic with
no learning or representation concerns. The matching featurizer lives in
``encoders/``; the two are composed by the candidate-action agents.

Import rule: modules here may import only the stdlib, ``sts2rl.data``, and each
other — never ``agents``, ``encoders``, or torch.
"""

from sts2rl.action_spaces.base import ActionSpace
from sts2rl.action_spaces.battle import BattleActionSpace
from sts2rl.action_spaces.event import EventActionSpace
from sts2rl.action_spaces.map import MapActionSpace
from sts2rl.action_spaces.rest import RestActionSpace
from sts2rl.action_spaces.reward import RewardActionSpace
from sts2rl.action_spaces.shop import ShopActionSpace

__all__ = [
    "ActionSpace",
    "BattleActionSpace",
    "EventActionSpace",
    "MapActionSpace",
    "RestActionSpace",
    "RewardActionSpace",
    "ShopActionSpace",
]
