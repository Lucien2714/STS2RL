"""Composed trainable encoder and candidate actor-critic heads."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from sts2rl.encoder.entity_encoder import (
    EncodedEntities,
    EncoderConfig,
    EntityTransformer,
)
from sts2rl.encoder.schema import ACTION_NUMERIC_FIELDS
from sts2rl.encoder.map_encoder import EncodedMap, MapDAGEncoder
from sts2rl.encoder.tokens import (
    EntityReference,
    TokenizedAction,
    TokenizedDecision,
    TokenizedState,
)
from sts2rl.encoder.vocabulary import GameVocabulary


@dataclass(frozen=True)
class EncodedDecision:
    """One fused state embedding and one row per dynamic candidate action."""

    state_embedding: Tensor
    candidate_embeddings: Tensor


@dataclass(frozen=True)
class PolicyValueOutput:
    """Actor logits and scalar critic value for one tokenized decision."""

    logits: Tensor
    value: Tensor
    encoded: EncodedDecision


class GameEncoder(nn.Module):
    """Share one structured representation across every game screen."""

    def __init__(
        self,
        vocabulary: GameVocabulary,
        config: EncoderConfig = EncoderConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        self.entity_encoder = EntityTransformer(vocabulary, config)
        self.map_encoder = MapDAGEncoder(vocabulary, config)
        self.no_map_embedding = nn.Parameter(torch.empty(hidden_dim))
        self.state_fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        self.state_norm = nn.LayerNorm(hidden_dim)

        self.action_type_embedding = nn.Embedding(
            vocabulary.size("action_types"),
            hidden_dim,
            padding_idx=0,
        )
        self.action_numeric_projection = nn.Linear(
            2 * len(ACTION_NUMERIC_FIELDS), hidden_dim
        )
        self.source_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.target_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.no_source_embedding = nn.Parameter(torch.empty(hidden_dim))
        self.no_target_embedding = nn.Parameter(torch.empty(hidden_dim))
        self.candidate_norm = nn.LayerNorm(hidden_dim)
        self.candidate_bias = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        for parameter in (
            self.no_map_embedding,
            self.no_source_embedding,
            self.no_target_embedding,
        ):
            nn.init.normal_(parameter, std=0.02)

    def forward(self, decision: TokenizedDecision) -> EncodedDecision:
        """Encode one state and its complete ordered dynamic candidate set."""
        state_embedding, entities, encoded_map = self._encode_state(decision.state)
        candidates = self._encode_actions(
            decision.actions,
            entities.entity_embeddings,
            encoded_map,
        )
        return EncodedDecision(state_embedding, candidates)

    def policy_value(self, decision: TokenizedDecision) -> PolicyValueOutput:
        """Run the shared encoder and produce actor logits and critic value."""
        encoded = self(decision)
        logits = (
            encoded.candidate_embeddings.matmul(encoded.state_embedding)
            / math.sqrt(self.config.hidden_dim)
        ) + self.candidate_bias(encoded.candidate_embeddings).squeeze(-1)
        value = self.value_head(encoded.state_embedding).squeeze(-1)
        return PolicyValueOutput(logits, value, encoded)

    def value(self, state: TokenizedState) -> Tensor:
        """Estimate one state value without requiring candidate actions."""
        state_embedding, _, _ = self._encode_state(state)
        return self.value_head(state_embedding).squeeze(-1)

    def _encode_state(
        self, state: TokenizedState
    ) -> tuple[Tensor, EncodedEntities, EncodedMap | None]:
        entities = self.entity_encoder(state)
        encoded_map = self.map_encoder(state.game_map, entities.state_embedding)
        map_context = (
            encoded_map.global_embedding
            if encoded_map is not None
            else self.no_map_embedding
        )
        state_embedding = self.state_norm(
            self.state_fusion(torch.cat([entities.state_embedding, map_context]))
        )
        return state_embedding, entities, encoded_map

    def _encode_actions(
        self,
        actions: Sequence[TokenizedAction],
        entity_embeddings: Mapping[str, Tensor],
        encoded_map: EncodedMap | None,
    ) -> Tensor:
        """Encode every candidate of one decision as a single [K, hidden] batch."""
        result = self.action_type_embedding(
            torch.stack([action.action_type for action in actions])
        )
        numeric = torch.stack(
            [
                torch.cat([action.numeric, action.numeric_mask.to(torch.float32)])
                for action in actions
            ]
        )
        result = result + self.action_numeric_projection(numeric)
        result = result + self._role_embeddings(
            [action.source for action in actions],
            entity_embeddings,
            encoded_map,
            self.source_projection,
            self.no_source_embedding,
        )
        result = result + self._role_embeddings(
            [action.target for action in actions],
            entity_embeddings,
            encoded_map,
            self.target_projection,
            self.no_target_embedding,
        )
        return self.candidate_norm(result)

    @staticmethod
    def _role_embeddings(
        references: Sequence[EntityReference | None],
        entity_embeddings: Mapping[str, Tensor],
        encoded_map: EncodedMap | None,
        projection: nn.Linear,
        missing: Tensor,
    ) -> Tensor:
        """Gather one referenced embedding per candidate, or the missing vector.

        References are grouped by entity kind so each kind costs one gather
        rather than one indexing call per candidate.
        """
        rows = missing.expand(len(references), -1)
        grouped: dict[str, list[tuple[int, int]]] = {}
        for slot, reference in enumerate(references):
            if reference is not None:
                grouped.setdefault(reference.kind, []).append((slot, reference.index))
        if not grouped:
            return rows

        slots: list[int] = []
        gathered: list[Tensor] = []
        for kind, pairs in grouped.items():
            if kind == "map_node":
                if encoded_map is None:
                    raise ValueError(
                        "map-node action reference requires an encoded map"
                    )
                source = encoded_map.node_embeddings
            else:
                source = entity_embeddings[kind]
            gathered.append(
                source.index_select(
                    0,
                    torch.tensor(
                        [index for _, index in pairs],
                        dtype=torch.long,
                        device=source.device,
                    ),
                )
            )
            slots.extend(slot for slot, _ in pairs)

        values = projection(torch.cat(gathered))
        return rows.index_copy(
            0,
            torch.tensor(slots, dtype=torch.long, device=values.device),
            values,
        )
