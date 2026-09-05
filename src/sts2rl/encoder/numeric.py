"""Numeric feature conversion with explicit missing-value masks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import Tensor


@dataclass(frozen=True)
class NumericFeature:
    """One scalar model input and whether its source value was present."""

    value: float
    present: bool

    def __post_init__(self) -> None:
        normalized_value = float(self.value)
        if not math.isfinite(normalized_value):
            raise ValueError("numeric feature value must be finite")
        if not self.present and normalized_value != 0.0:
            raise ValueError("a missing numeric feature must use value 0.0")
        object.__setattr__(self, "value", normalized_value)

    @classmethod
    def missing(cls) -> NumericFeature:
        """Return the canonical representation of a missing numeric value."""
        return cls(0.0, False)


def linear_feature(value: object) -> NumericFeature:
    """Parse one finite scalar without changing its scale."""
    parsed = _finite_float(value)
    if parsed is None:
        return NumericFeature.missing()
    return NumericFeature(parsed, True)


def signed_log_feature(value: object) -> NumericFeature:
    """Compress an unbounded scalar while preserving its sign and zero."""
    parsed = _finite_float(value)
    if parsed is None:
        return NumericFeature.missing()
    transformed = math.copysign(math.log1p(abs(parsed)), parsed)
    return NumericFeature(transformed, True)


def ratio_feature(numerator: object, denominator: object) -> NumericFeature:
    """Return a finite ratio when both values and a positive divisor exist."""
    parsed_numerator = _finite_float(numerator)
    parsed_denominator = _finite_float(denominator)
    if (
        parsed_numerator is None
        or parsed_denominator is None
        or parsed_denominator <= 0
    ):
        return NumericFeature.missing()
    return NumericFeature(parsed_numerator / parsed_denominator, True)


def pack_numeric(features: Iterable[NumericFeature]) -> tuple[Tensor, Tensor]:
    """Pack scalar features into float values and a boolean presence mask."""
    packed = tuple(features)
    values = torch.tensor(
        [feature.value for feature in packed],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [feature.present for feature in packed],
        dtype=torch.bool,
    )
    return values, mask


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
