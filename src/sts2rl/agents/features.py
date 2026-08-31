"""Lightweight fixed-size features for the first PPO implementation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from sts2rl.actions import GameAction
from sts2rl.env.types import RawState


class FeatureEncoder(Protocol):
    """Tensor feature boundary consumed by CandidatePPOAgent."""

    state_dim: int
    action_dim: int

    def encode_state(self, state: RawState) -> Tensor:
        """Encode one raw state as a flat float tensor."""

    def encode_action(self, state: RawState, action: GameAction) -> Tensor:
        """Encode one structured action in the context of its state."""


@dataclass(frozen=True)
class HashingFeatureEncoder:
    """Encode arbitrary raw states and structured actions with stable hashing.

    This is a runnable baseline, not a replacement for the retained domain
    StateEncoder. A later encoder can implement the same two methods.
    """

    state_dim: int = 512
    action_dim: int = 128

    def __post_init__(self) -> None:
        if self.state_dim < 1 or self.action_dim < 1:
            raise ValueError("feature dimensions must be positive")

    def encode_state(self, state: RawState) -> Tensor:
        return self._encode(state, self.state_dim)

    def encode_action(self, state: RawState, action: GameAction) -> Tensor:
        del state
        return self._encode(action.to_dict(), self.action_dim)

    def _encode(self, value: object, dimension: int) -> Tensor:
        features = torch.zeros(dimension, dtype=torch.float32)
        for path, scalar in self._flatten(value):
            if isinstance(scalar, bool):
                token = f"{path}={scalar}"
                features[self._bucket(token, dimension)] += 1.0
            elif isinstance(scalar, (int, float)):
                magnitude = math.copysign(math.log1p(abs(float(scalar))), float(scalar))
                features[self._bucket(path, dimension)] += magnitude
            else:
                token = f"{path}={scalar}"
                features[self._bucket(token, dimension)] += 1.0
        norm = torch.linalg.vector_norm(features)
        return features / norm if norm > 0 else features

    def _flatten(self, value: object, path: str = "root"):
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                yield from self._flatten(value[key], f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                yield from self._flatten(item, f"{path}.{index}")
        elif value is not None:
            yield path, value

    @staticmethod
    def _bucket(token: str, dimension: int) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % dimension
