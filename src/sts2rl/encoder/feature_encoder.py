"""Feature-encoding contract consumed by policy implementations."""

from __future__ import annotations

from typing import Protocol

from torch import Tensor

from sts2rl.actions import GameAction
from sts2rl.env.types import RawState


class FeatureEncoder(Protocol):
    """Encode raw states and contextual action candidates as flat tensors."""

    state_dim: int
    action_dim: int

    def encode_state(self, state: RawState) -> Tensor:
        """Encode one raw state as a flat float tensor."""

    def encode_action(self, state: RawState, action: GameAction) -> Tensor:
        """Encode one structured action in the context of its state."""
