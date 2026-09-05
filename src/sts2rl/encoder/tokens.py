"""Immutable structured tokens shared by tokenizers and trainable encoders.

This module is the contract between three otherwise independent layers::

    GameObservation
        -> GameTokenizer
        -> TokenizedState / TokenizedDecision
        -> GameEncoder
        -> PPO

The tokenizer creates these objects on CPU.  PPO keeps the CPU objects in its
rollout and calls ``.to(device)`` only when the encoder needs them.  The token
objects contain no trainable parameters and do not interpret raw game JSON.

Tensor conventions used throughout this module:

* categorical IDs use ``torch.long`` and index a vocabulary/embedding table;
* numeric values use ``torch.float32``;
* activity, presence, and graph masks use ``torch.bool``;
* the last dimension is the feature dimension unless documented otherwise.

Numeric masks are deliberately separate from numeric values.  A value of zero
with a true mask means "present and equal to zero"; zero with a false mask means
"missing".  See :mod:`sts2rl.encoder.numeric` for feature construction.

Dataclasses are frozen so PPO cannot accidentally replace a stored rollout
field.  This is shallow immutability: callers must still treat the contained
PyTorch tensors as read-only and avoid in-place tensor operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor


Device = str | torch.device


def _require_tensor(
    name: str,
    value: Tensor,
    *,
    dimensions: int,
    dtype: torch.dtype,
) -> None:
    """Validate the common rank and dtype parts of the token contract."""
    if value.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    if value.dtype != dtype:
        raise TypeError(f"{name} must use dtype {dtype}")


def _require_matching_numeric(
    prefix: str,
    values: Tensor,
    mask: Tensor,
    *,
    dimensions: int,
) -> None:
    """Validate a numeric tensor and its one-to-one presence mask."""
    _require_tensor(
        f"{prefix}_numeric",
        values,
        dimensions=dimensions,
        dtype=torch.float32,
    )
    _require_tensor(
        f"{prefix}_numeric_mask",
        mask,
        dimensions=dimensions,
        dtype=torch.bool,
    )
    if values.shape != mask.shape:
        raise ValueError(f"{prefix} numeric values and mask must have equal shape")


def _validate_kind(kind: str) -> None:
    """Keep internal routing keys canonical and safe to compare directly."""
    if not kind:
        raise ValueError("entity kind must be a non-empty string")
    if kind != kind.strip() or kind != kind.casefold():
        raise ValueError("entity kind must be a normalized lowercase string")


@dataclass(frozen=True)
class EntityReference:
    """Reference one semantic entity or map node by kind and local index.

    Ordinary references address row ``index`` of
    ``TokenizedState.entities[kind]``.  Keeping an index local to its entity
    kind lets tokenizers and encoders operate on dense per-kind matrices rather
    than thousands of tiny per-entity tensors.

    ``kind="map_node"`` is reserved for ``TokenizedMap`` and addresses one row
    of its node tensors.  Map nodes are separate because their graph edges and
    topological order cannot be represented by ordinary entity batches alone.

    Attributes:
        kind: Normalized lowercase routing name such as ``"card"`` or
            ``"enemy"``.
        index: Zero-based row local to the entity batch or map-node matrix
            selected by ``kind``.
    """

    kind: str
    index: int

    def __post_init__(self) -> None:
        _validate_kind(self.kind)
        if self.index < 0:
            raise ValueError("entity reference index must not be negative")


@dataclass(frozen=True)
class TokenizedEntityBatch:
    """All entities of one kind stored in dense feature matrices.

    Cards, relics, potions, enemies, powers, and choices each have different
    feature schemas, so every kind owns a separate batch.  Within one batch,
    row ``i`` across every tensor describes the same entity.  This avoids three
    small tensor allocations per entity and lets the future encoder process a
    complete kind with one embedding/projection call.

    Attributes:
        kind: Entity projection/routing name, such as ``"card"`` or
            ``"enemy"``.  Map nodes are not allowed here.
        categorical: Long tensor
            ``[entity_count, categorical_feature_count]``.  A card row can
            contain card ID, type, rarity, zone, and target IDs.
        numeric: Float32 tensor ``[entity_count, numeric_feature_count]``.
        numeric_mask: Bool tensor matching ``numeric``.  False entries mark
            source values that were absent rather than numerically zero.
        active: Bool tensor ``[entity_count]`` containing each entity's active
            state.  False means inactive only when the matching active-mask
            value is true.
        active_mask: Bool tensor ``[entity_count]``.  True means ``active`` was
            explicitly observed; false means the active state is unknown.
        owners: One optional reference per row.  Powers can use this to name
            the player, enemy, or pet that owns them.  An omitted empty tuple
            is expanded to ``None`` for every row.
        children: One tuple of references per row.  Bundles can use this to
            name their contained cards.  An omitted empty tuple is expanded to
            an empty child tuple for every row.

    Repeated, observationally identical cards in an unordered zone can occupy
    one row with ``copy_count`` in the numeric schema.  Actionable hand or
    selection cards remain separate rows so raw action handles still resolve
    one-to-one.  Grouping policy belongs to ``GameTokenizer``, not this class.
    """

    kind: str
    categorical: Tensor
    numeric: Tensor
    numeric_mask: Tensor
    active: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.bool)
    )
    active_mask: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.bool)
    )
    owners: tuple[EntityReference | None, ...] = ()
    children: tuple[tuple[EntityReference, ...], ...] = ()

    def __post_init__(self) -> None:
        _validate_kind(self.kind)
        _require_tensor(
            f"{self.kind} categorical",
            self.categorical,
            dimensions=2,
            dtype=torch.long,
        )
        _require_matching_numeric(
            self.kind,
            self.numeric,
            self.numeric_mask,
            dimensions=2,
        )

        entity_count = self.categorical.shape[0]
        if self.numeric.shape[0] != entity_count:
            raise ValueError(f"all {self.kind} tensors must have equal row count")

        _require_tensor(
            f"{self.kind} active",
            self.active,
            dimensions=1,
            dtype=torch.bool,
        )
        _require_tensor(
            f"{self.kind} active_mask",
            self.active_mask,
            dimensions=1,
            dtype=torch.bool,
        )
        active_supplied = self.active.numel() > 0
        active_mask_supplied = self.active_mask.numel() > 0
        if active_supplied != active_mask_supplied:
            raise ValueError("entity active and active_mask must be supplied together")
        if not active_supplied:
            object.__setattr__(
                self,
                "active",
                torch.zeros(
                    entity_count,
                    dtype=torch.bool,
                    device=self.categorical.device,
                ),
            )
            object.__setattr__(
                self,
                "active_mask",
                torch.zeros(
                    entity_count,
                    dtype=torch.bool,
                    device=self.categorical.device,
                ),
            )
        if self.active.shape[0] != entity_count:
            raise ValueError(f"{self.kind} active must contain one value per row")
        if self.active_mask.shape[0] != entity_count:
            raise ValueError(
                f"{self.kind} active_mask must contain one value per row"
            )

        owners = tuple(self.owners) or (None,) * entity_count
        children = tuple(tuple(row) for row in self.children) or (
            (),
        ) * entity_count
        if len(owners) != entity_count:
            raise ValueError(f"{self.kind} owners must contain one value per row")
        if len(children) != entity_count:
            raise ValueError(f"{self.kind} children must contain one tuple per row")
        object.__setattr__(self, "owners", owners)
        object.__setattr__(self, "children", children)

    @property
    def entity_count(self) -> int:
        """Return the number of entity rows in this batch."""
        return self.categorical.shape[0]

    def to(self, device: Device) -> TokenizedEntityBatch:
        """Return a moved copy; relationship references are reused."""
        return TokenizedEntityBatch(
            kind=self.kind,
            categorical=self.categorical.to(device),
            numeric=self.numeric.to(device),
            numeric_mask=self.numeric_mask.to(device),
            active=self.active.to(device),
            active_mask=self.active_mask.to(device),
            owners=self.owners,
            children=self.children,
        )


@dataclass(frozen=True)
class TokenizedMap:
    """Tensor representation of one validated, directed map graph.

    The STS map is a directed acyclic graph (DAG).  Every column in
    ``edge_index`` is ``[parent, child]``: traversal proceeds from the current
    floor toward a boss.  The map encoder groups nodes by their distance to a
    sink and walks those levels so a parent always aggregates already-computed
    representations of all its children.

    Attributes:
        node_categorical: Long tensor ``[node_count, categorical_feature_count]``
            containing values such as node type.
        node_numeric: Float32 tensor ``[node_count, numeric_feature_count]``
            containing positions, distances, and structural summaries.
        node_numeric_mask: Bool tensor matching ``node_numeric``.
        edge_index: Long tensor ``[2, edge_count]``.  Row 0 contains parents;
            row 1 contains their corresponding children.
        reachable_mask: Bool tensor ``[node_count]``.  True means the node is
            still reachable from the current decision; global map pooling must
            ignore false entries.
        candidate_indices: Long tensor ``[candidate_count]`` mapping the
            currently legal map choices to node rows.
        boss_indices: Long tensor ``[boss_count]`` identifying terminal nodes.
        candidate_type_counts: Float32 tensor
            ``[candidate_count, map_node_type_count]`` summarizing the complete
            reachable subtree below each candidate, not only its direct child.
        current_index: Current node row, or None before a concrete current map
            position exists.

    Coordinates are not stored as identity fields here.  ``GameTokenizer``
    resolves stable ``(col, row)`` coordinates into these dense node indices
    before constructing this object.
    """

    node_categorical: Tensor
    node_numeric: Tensor
    node_numeric_mask: Tensor
    edge_index: Tensor
    reachable_mask: Tensor
    candidate_indices: Tensor
    boss_indices: Tensor
    candidate_type_counts: Tensor
    current_index: int | None

    def __post_init__(self) -> None:
        _require_tensor(
            "map node_categorical",
            self.node_categorical,
            dimensions=2,
            dtype=torch.long,
        )
        _require_matching_numeric(
            "map node",
            self.node_numeric,
            self.node_numeric_mask,
            dimensions=2,
        )
        _require_tensor(
            "map edge_index",
            self.edge_index,
            dimensions=2,
            dtype=torch.long,
        )
        _require_tensor(
            "map reachable_mask",
            self.reachable_mask,
            dimensions=1,
            dtype=torch.bool,
        )
        _require_tensor(
            "map candidate_indices",
            self.candidate_indices,
            dimensions=1,
            dtype=torch.long,
        )
        _require_tensor(
            "map boss_indices",
            self.boss_indices,
            dimensions=1,
            dtype=torch.long,
        )
        _require_tensor(
            "map candidate_type_counts",
            self.candidate_type_counts,
            dimensions=2,
            dtype=torch.float32,
        )

        node_count = self.node_categorical.shape[0]
        # Every node-level matrix/mask must describe the same ordered node set.
        if self.node_numeric.shape[0] != node_count:
            raise ValueError("all map node tensors must have the same row count")
        if self.edge_index.shape[0] != 2:
            raise ValueError("map edge_index must have shape [2, edge_count]")
        if self.reachable_mask.shape[0] != node_count:
            raise ValueError("map reachable_mask must contain one value per node")
        if self.candidate_type_counts.shape[0] != self.candidate_indices.shape[0]:
            raise ValueError(
                "candidate_type_counts must contain one row per candidate"
            )

        self._validate_indices(node_count)

    def _validate_indices(self, node_count: int) -> None:
        """Ensure every dense graph reference addresses an existing node row."""
        index_tensors = {
            "edge_index": self.edge_index,
            "candidate_indices": self.candidate_indices,
            "boss_indices": self.boss_indices,
        }
        for name, values in index_tensors.items():
            if values.numel() == 0:
                continue
            minimum = int(values.min().item())
            maximum = int(values.max().item())
            if minimum < 0 or maximum >= node_count:
                raise ValueError(f"map {name} contains an out-of-range node index")

        if self.current_index is not None:
            if not 0 <= self.current_index < node_count:
                raise ValueError("map current_index is out of range")

    def to(self, device: Device) -> TokenizedMap:
        """Return a copy whose tensors reside on the requested device."""
        return TokenizedMap(
            node_categorical=self.node_categorical.to(device),
            node_numeric=self.node_numeric.to(device),
            node_numeric_mask=self.node_numeric_mask.to(device),
            edge_index=self.edge_index.to(device),
            reachable_mask=self.reachable_mask.to(device),
            candidate_indices=self.candidate_indices.to(device),
            boss_indices=self.boss_indices.to(device),
            candidate_type_counts=self.candidate_type_counts.to(device),
            current_index=self.current_index,
        )


@dataclass(frozen=True)
class TokenizedState:
    """Structured state tokens and their optional full-map graph.

    Attributes:
        global_categorical: Rank-1 long tensor containing state-wide vocabulary
            IDs such as state type and character.
        global_numeric: Rank-1 float32 tensor containing run/player scalars such
            as floor, HP, gold, and energy.
        global_numeric_mask: Rank-1 bool tensor matching ``global_numeric``.
        entities: Immutable mapping from entity kind to its dense batch.  Row
            order inside each batch is tokenizer-defined and captured in the
            rollout; it is never reconstructed during PPO update.
        game_map: Full map DAG on map screens, otherwise None.

    ``TokenizedState`` validates internal owner/child references immediately.
    Action references are validated later by ``TokenizedDecision``, once the
    state and complete candidate set are available together.
    """

    global_categorical: Tensor
    global_numeric: Tensor
    global_numeric_mask: Tensor
    entities: Mapping[str, TokenizedEntityBatch] = field(default_factory=dict)
    game_map: TokenizedMap | None = None

    def __post_init__(self) -> None:
        _require_tensor(
            "state global_categorical",
            self.global_categorical,
            dimensions=1,
            dtype=torch.long,
        )
        _require_matching_numeric(
            "state global",
            self.global_numeric,
            self.global_numeric_mask,
            dimensions=1,
        )
        entities = dict(self.entities)
        object.__setattr__(self, "entities", MappingProxyType(entities))
        for kind, batch in self.entities.items():
            if kind != batch.kind:
                raise ValueError("entity mapping key must match batch kind")
            if kind == "map_node":
                raise ValueError("map nodes must be stored in TokenizedMap")
            for owner in batch.owners:
                if owner is not None:
                    self.validate_reference(owner)
            for row in batch.children:
                for child in row:
                    self.validate_reference(child)

    def validate_reference(self, reference: EntityReference) -> None:
        """Validate reference range and semantic kind against this state."""
        if reference.kind == "map_node":
            if self.game_map is None:
                raise ValueError("map-node reference requires TokenizedMap")
            if reference.index >= self.game_map.node_categorical.shape[0]:
                raise ValueError("map-node reference is out of range")
            return

        batch = self.entities.get(reference.kind)
        if batch is None:
            raise ValueError(f"entity reference has unknown kind {reference.kind!r}")
        if reference.index >= batch.entity_count:
            raise ValueError("entity reference is out of range")

    def to(self, device: Device) -> TokenizedState:
        """Return a recursively moved copy without mutating this state."""
        return TokenizedState(
            global_categorical=self.global_categorical.to(device),
            global_numeric=self.global_numeric.to(device),
            global_numeric_mask=self.global_numeric_mask.to(device),
            entities={
                kind: batch.to(device) for kind, batch in self.entities.items()
            },
            game_map=self.game_map.to(device) if self.game_map is not None else None,
        )


@dataclass(frozen=True)
class TokenizedAction:
    """One complete candidate action expressed through semantic references.

    Attributes:
        action_type: Scalar long tensor containing an ``action_types``
            vocabulary index.
        numeric: Rank-1 float32 tensor of genuinely numeric action parameters,
            for example a Crystal Sphere coordinate when position is semantic.
        numeric_mask: Rank-1 bool tensor matching ``numeric``.
        source: Optional entity that performs or supplies the action, such as a
            hand card, potion, reward item, or selectable option.
        target: Optional affected entity, normally an enemy or map node.

    Screen-local values such as ``card_index`` and option ``index`` must not be
    copied into ``numeric``.  ``GameTokenizer`` first resolves them to the
    entity referenced by ``source`` or ``target``.  Actions such as end turn,
    proceed, skip, and confirm legitimately have neither reference.
    """

    action_type: Tensor
    numeric: Tensor
    numeric_mask: Tensor
    source: EntityReference | None = None
    target: EntityReference | None = None

    def __post_init__(self) -> None:
        _require_tensor(
            "action action_type",
            self.action_type,
            dimensions=0,
            dtype=torch.long,
        )
        _require_matching_numeric(
            "action",
            self.numeric,
            self.numeric_mask,
            dimensions=1,
        )

    def to(self, device: Device) -> TokenizedAction:
        """Return a copy whose tensors reside on the requested device."""
        return TokenizedAction(
            action_type=self.action_type.to(device),
            numeric=self.numeric.to(device),
            numeric_mask=self.numeric_mask.to(device),
            source=self.source,
            target=self.target,
        )


@dataclass(frozen=True)
class TokenizedDecision:
    """One state and the complete ordered candidate set scored by PPO.

    Candidate order is part of the rollout contract: the sampled action index
    and old log probability only remain meaningful when PPO updates against the
    exact same ordered candidates.  Consequently a decision cannot be empty,
    and every source/target reference is checked against its state at creation.

    Attributes:
        state: Tokenized state shared by all candidates.
        actions: Complete candidate sequence in the same order returned by
            ``LegalActionProvider``.
    """

    state: TokenizedState
    actions: tuple[TokenizedAction, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(self.actions))
        if not self.actions:
            raise ValueError("a tokenized decision requires at least one action")
        for action in self.actions:
            if action.source is not None:
                self.state.validate_reference(action.source)
            if action.target is not None:
                self.state.validate_reference(action.target)

    def to(self, device: Device) -> TokenizedDecision:
        """Return a recursively moved copy without mutating this decision."""
        return TokenizedDecision(
            state=self.state.to(device),
            actions=tuple(action.to(device) for action in self.actions),
        )
