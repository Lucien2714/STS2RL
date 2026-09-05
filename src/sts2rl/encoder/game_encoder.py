"""Composed trainable encoder and candidate actor-critic heads."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn

from sts2rl.encoder.entity_encoder import (
    EncodedEntities,
    EncoderConfig,
    EntityTransformer,
)
from sts2rl.encoder.game_tokenizer import ACTION_NUMERIC_FIELDS
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
        candidates = torch.stack(
            [
                self._encode_action(action, entities.entity_embeddings, encoded_map)
                for action in decision.actions
            ]
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

    def _encode_action(
        self,
        action: TokenizedAction,
        entity_embeddings: Mapping[str, Tensor],
        encoded_map: EncodedMap | None,
    ) -> Tensor:
        result = self.action_type_embedding(action.action_type)
        numeric = torch.cat(
            [action.numeric, action.numeric_mask.to(torch.float32)]
        )
        result = result + self.action_numeric_projection(numeric)
        source = self._reference_embedding(
            action.source,
            entity_embeddings,
            encoded_map,
        )
        target = self._reference_embedding(
            action.target,
            entity_embeddings,
            encoded_map,
        )
        result = result + (
            self.source_projection(source)
            if source is not None
            else self.no_source_embedding
        )
        result = result + (
            self.target_projection(target)
            if target is not None
            else self.no_target_embedding
        )
        return self.candidate_norm(result)

    @staticmethod
    def _reference_embedding(
        reference: EntityReference | None,
        entity_embeddings: Mapping[str, Tensor],
        encoded_map: EncodedMap | None,
    ) -> Tensor | None:
        if reference is None:
            return None
        if reference.kind == "map_node":
            if encoded_map is None:
                raise ValueError("map-node action reference requires an encoded map")
            return encoded_map.node_embeddings[reference.index]
        return entity_embeddings[reference.kind][reference.index]
