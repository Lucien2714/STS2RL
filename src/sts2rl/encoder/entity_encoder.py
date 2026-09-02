"""Trainable, permutation-aware encoder for non-map game entities."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor, nn

from sts2rl.encoder.game_tokenizer import (
    ENTITY_CATEGORICAL_FIELDS,
    ENTITY_NUMERIC_FIELDS,
    GLOBAL_CATEGORICAL_FIELDS,
    GLOBAL_NUMERIC_FIELDS,
)
from sts2rl.encoder.tokens import EntityReference, TokenizedState
from sts2rl.encoder.vocabulary import GameVocabulary


GLOBAL_CATEGORICAL_VOCABS: Mapping[str, str] = MappingProxyType(
    {"state_type": "state_types", "character": "characters"}
)

ENTITY_CATEGORICAL_VOCABS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "player": ("characters", "owner_types"),
        "card": (
            "cards",
            "card_types",
            "rarities",
            "card_zones",
            "entity_zones",
            "target_types",
            "enchantments",
            "selection_types",
        ),
        "relic": ("relics", "rarities", "entity_zones"),
        "potion": ("potions", "target_types", "entity_zones"),
        "orb": ("orbs", "entity_zones"),
        "pet": ("monsters", "owner_types", "entity_zones"),
        "enemy": ("monsters", "owner_types", "entity_zones"),
        "power": ("powers", "power_types", "owner_types", "entity_zones"),
        "intent": ("intents", "entity_zones"),
        "reward": (
            "reward_types",
            "potions",
            "relics",
            "cards",
            "entity_zones",
        ),
        "shop_item": (
            "shop_categories",
            "cards",
            "relics",
            "potions",
            "card_types",
            "rarities",
            "target_types",
            "entity_zones",
        ),
        "event_option": ("events", "event_options", "entity_zones"),
        "rest_option": ("rest_options", "entity_zones"),
        "bundle": ("selection_types", "entity_zones"),
        "crystal_cell": ("crystal_item_types", "entity_zones"),
        "crystal_tool": ("crystal_tools", "entity_zones"),
    }
)


@dataclass(frozen=True)
class EncoderConfig:
    """Shared trainable-encoder dimensions and deterministic dropout policy."""

    hidden_dim: int = 128
    entity_layers: int = 2
    entity_heads: int = 4
    entity_ff_dim: int = 256
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("hidden_dim", "entity_layers", "entity_heads", "entity_ff_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.entity_heads:
            raise ValueError("hidden_dim must be divisible by entity_heads")
        if self.dropout != 0.0:
            raise ValueError("dropout must remain 0.0 for deterministic PPO ratios")


@dataclass(frozen=True)
class EncodedEntities:
    """Contextual state vector and per-kind entity row embeddings."""

    state_embedding: Tensor
    entity_embeddings: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        if self.state_embedding.ndim != 1:
            raise ValueError("state_embedding must have shape [hidden_dim]")
        embeddings = dict(self.entity_embeddings)
        hidden_dim = self.state_embedding.shape[0]
        for kind, values in embeddings.items():
            if values.ndim != 2 or values.shape[1] != hidden_dim:
                raise ValueError(
                    f"{kind} embeddings must have shape [count, hidden_dim]"
                )
        object.__setattr__(self, "entity_embeddings", MappingProxyType(embeddings))

    def reference(self, reference: EntityReference) -> Tensor:
        """Return the embedding addressed by a validated non-map reference."""
        if reference.kind == "map_node":
            raise ValueError("map-node embeddings belong to MapDAGEncoder")
        try:
            return self.entity_embeddings[reference.kind][reference.index]
        except (KeyError, IndexError) as exc:
            raise ValueError(f"unresolved encoded entity reference: {reference}") from exc


class EntityTransformer(nn.Module):
    """Encode globals and heterogeneous entities without positional embeddings."""

    RELATION_KINDS = frozenset({"power", "intent"})

    def __init__(
        self,
        vocabulary: GameVocabulary,
        config: EncoderConfig = EncoderConfig(),
    ) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.config = config

        self.global_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(
                    self._vocabulary_size(table),
                    config.hidden_dim,
                    padding_idx=0,
                )
                for field, table in GLOBAL_CATEGORICAL_VOCABS.items()
            }
        )
        self.entity_embeddings = nn.ModuleDict(
            {
                self._embedding_key(kind, field): nn.Embedding(
                    self._vocabulary_size(table),
                    config.hidden_dim,
                    padding_idx=0,
                )
                for kind, tables in ENTITY_CATEGORICAL_VOCABS.items()
                for field, table in zip(ENTITY_CATEGORICAL_FIELDS[kind], tables)
                if field != "entity_zone"
            }
        )
        self.global_numeric_projection = nn.Linear(
            2 * len(GLOBAL_NUMERIC_FIELDS), config.hidden_dim
        )
        self.entity_numeric_projections = nn.ModuleDict(
            {
                kind: nn.Linear(
                    2 * len(ENTITY_NUMERIC_FIELDS[kind]) + 2,
                    config.hidden_dim,
                )
                for kind in ENTITY_NUMERIC_FIELDS
            }
        )
        self.entity_type_embedding = nn.Embedding(
            vocabulary.size("entity_types"),
            config.hidden_dim,
            padding_idx=0,
        )
        self.zone_embedding = nn.Embedding(
            vocabulary.size("entity_zones"),
            config.hidden_dim,
            padding_idx=0,
        )
        self.power_owner_projection = nn.Linear(
            config.hidden_dim, config.hidden_dim, bias=False
        )
        self.intent_owner_projection = nn.Linear(
            config.hidden_dim, config.hidden_dim, bias=False
        )
        self.bundle_child_projection = nn.Linear(
            config.hidden_dim, config.hidden_dim, bias=False
        )
        self.state_token = nn.Parameter(torch.empty(config.hidden_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.entity_heads,
            dim_feedforward=config.entity_ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.entity_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        nn.init.normal_(self.state_token, std=0.02)

    def forward(self, state: TokenizedState) -> EncodedEntities:
        """Return contextual embeddings while preserving per-kind row addressing."""
        if not isinstance(state, TokenizedState):
            raise TypeError("state must be a TokenizedState")

        embeddings = {
            kind: self._project_entity_batch(kind, state)
            for kind in ENTITY_CATEGORICAL_FIELDS
        }
        self._aggregate_owned_relations(
            embeddings,
            state,
            "power",
            self.power_owner_projection,
        )
        self._aggregate_owned_relations(
            embeddings,
            state,
            "intent",
            self.intent_owner_projection,
        )
        bundled_children = self._aggregate_bundle_children(embeddings, state)

        sequence = [self._project_state(state).unsqueeze(0)]
        locations: list[tuple[str, Tensor]] = []
        for kind in ENTITY_CATEGORICAL_FIELDS:
            if kind in self.RELATION_KINDS:
                continue
            count = embeddings[kind].shape[0]
            included = torch.arange(count, device=embeddings[kind].device)
            excluded = bundled_children.get(kind)
            if excluded:
                mask = torch.ones(count, dtype=torch.bool, device=included.device)
                mask[list(excluded)] = False
                included = included[mask]
            if included.numel():
                sequence.append(embeddings[kind].index_select(0, included))
                locations.append((kind, included))

        contextual = self.transformer(torch.cat(sequence).unsqueeze(0)).squeeze(0)
        contextual = self.final_norm(contextual)
        output_embeddings = dict(embeddings)
        offset = 1
        for kind, indices in locations:
            length = indices.shape[0]
            output_embeddings[kind] = output_embeddings[kind].index_copy(
                0,
                indices,
                contextual[offset : offset + length],
            )
            offset += length
        return EncodedEntities(contextual[0], output_embeddings)

    def _project_state(self, state: TokenizedState) -> Tensor:
        result = self.state_token
        for column, field in enumerate(GLOBAL_CATEGORICAL_FIELDS):
            result = result + self.global_embeddings[field](
                state.global_categorical[column]
            )
        numeric = torch.cat(
            [state.global_numeric, state.global_numeric_mask.to(torch.float32)]
        )
        return result + self.global_numeric_projection(numeric)

    def _project_entity_batch(
        self, kind: str, state: TokenizedState
    ) -> Tensor:
        batch = state.entities[kind]
        result = torch.zeros(
            (batch.entity_count, self.config.hidden_dim),
            dtype=torch.float32,
            device=batch.categorical.device,
        )
        for column, field in enumerate(ENTITY_CATEGORICAL_FIELDS[kind]):
            if field == "entity_zone":
                categorical = self.zone_embedding(batch.categorical[:, column])
            else:
                categorical = self.entity_embeddings[
                    self._embedding_key(kind, field)
                ](batch.categorical[:, column])
            result = result + categorical
        numeric = torch.cat(
            [
                batch.numeric,
                batch.numeric_mask.to(torch.float32),
                batch.active.to(torch.float32).unsqueeze(1),
                batch.active_mask.to(torch.float32).unsqueeze(1),
            ],
            dim=1,
        )
        result = result + self.entity_numeric_projections[kind](numeric)
        type_index = self.vocabulary.lookup("entity_types", kind)
        return result + self.entity_type_embedding(
            torch.tensor(type_index, dtype=torch.long, device=result.device)
        )

    def _aggregate_owned_relations(
        self,
        embeddings: dict[str, Tensor],
        state: TokenizedState,
        relation_kind: str,
        projection: nn.Linear,
    ) -> None:
        relations = embeddings[relation_kind]
        owners = state.entities[relation_kind].owners
        for owner_kind in embeddings:
            pairs = [
                (relation_index, owner.index)
                for relation_index, owner in enumerate(owners)
                if owner is not None and owner.kind == owner_kind
            ]
            if not pairs:
                continue
            relation_indices = torch.tensor(
                [pair[0] for pair in pairs],
                dtype=torch.long,
                device=relations.device,
            )
            owner_indices = torch.tensor(
                [pair[1] for pair in pairs],
                dtype=torch.long,
                device=relations.device,
            )
            sums = torch.zeros_like(embeddings[owner_kind]).index_add(
                0,
                owner_indices,
                relations.index_select(0, relation_indices),
            )
            counts = torch.zeros(
                embeddings[owner_kind].shape[0],
                dtype=relations.dtype,
                device=relations.device,
            ).index_add(
                0,
                owner_indices,
                torch.ones(len(pairs), dtype=relations.dtype, device=relations.device),
            )
            means = sums / counts.clamp_min(1).unsqueeze(1)
            embeddings[owner_kind] = embeddings[owner_kind] + projection(means)

    def _aggregate_bundle_children(
        self,
        embeddings: dict[str, Tensor],
        state: TokenizedState,
    ) -> dict[str, set[int]]:
        excluded: dict[str, set[int]] = {}
        bundle_updates = torch.zeros_like(embeddings["bundle"])
        for bundle_index, children in enumerate(state.entities["bundle"].children):
            if not children:
                continue
            child_values = [embeddings[child.kind][child.index] for child in children]
            mean = torch.stack(child_values).mean(dim=0)
            bundle_updates[bundle_index] = self.bundle_child_projection(mean)
            for child in children:
                excluded.setdefault(child.kind, set()).add(child.index)
        embeddings["bundle"] = embeddings["bundle"] + bundle_updates
        return excluded

    def _vocabulary_size(self, table: str) -> int:
        if table == "event_options":
            return len(self.vocabulary.event_options)
        return self.vocabulary.size(table)

    @staticmethod
    def _embedding_key(kind: str, field: str) -> str:
        return f"{kind}__{field}"
