"""Player-conditioned reverse DAG encoder for complete run maps."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from sts2rl.encoder.entity_encoder import EncoderConfig
from sts2rl.encoder.game_tokenizer import MAP_NUMERIC_FIELDS
from sts2rl.encoder.tokens import TokenizedMap
from sts2rl.encoder.vocabulary import GameVocabulary


@dataclass(frozen=True)
class EncodedMap:
    """Base nodes, future-aware nodes, and one reachable-map summary."""

    base_node_embeddings: Tensor
    node_embeddings: Tensor
    global_embedding: Tensor

    def __post_init__(self) -> None:
        if self.base_node_embeddings.ndim != 2:
            raise ValueError("base_node_embeddings must have shape [nodes, hidden]")
        if self.node_embeddings.shape != self.base_node_embeddings.shape:
            raise ValueError("base and future node embeddings must have equal shape")
        if self.global_embedding.ndim != 1:
            raise ValueError("global_embedding must have shape [hidden]")
        if self.global_embedding.shape[0] != self.node_embeddings.shape[1]:
            raise ValueError("map embeddings must use one hidden dimension")


class MapDAGEncoder(nn.Module):
    """Propagate future information from bosses toward current candidates."""

    def __init__(
        self,
        vocabulary: GameVocabulary,
        config: EncoderConfig = EncoderConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        self.node_type_embedding = nn.Embedding(
            vocabulary.size("map_node_types"),
            hidden_dim,
            padding_idx=0,
        )
        self.node_numeric_projection = nn.Linear(
            2 * len(MAP_NUMERIC_FIELDS), hidden_dim
        )
        self.child_query = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.child_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.child_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.node_update = nn.Sequential(
            nn.Linear(3 * hidden_dim, config.entity_ff_dim),
            nn.GELU(),
            nn.Linear(config.entity_ff_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.pool_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pool_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pool_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pool_context = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pool_norm = nn.LayerNorm(hidden_dim)
        self.empty_map_embedding = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.empty_map_embedding, std=0.02)

    def forward(
        self,
        game_map: TokenizedMap | None,
        player_context: Tensor,
    ) -> EncodedMap | None:
        """Encode a map, or skip the module entirely for non-map states."""
        if game_map is None:
            return None
        if player_context.ndim != 1 or player_context.shape[0] != self.config.hidden_dim:
            raise ValueError("player_context must have shape [hidden_dim]")

        base = self.node_type_embedding(game_map.node_categorical[:, 0])
        numeric = torch.cat(
            [game_map.node_numeric, game_map.node_numeric_mask.to(torch.float32)],
            dim=1,
        )
        base = base + self.node_numeric_projection(numeric)

        adjacency = self._adjacency(game_map)
        future: list[Tensor | None] = [None] * base.shape[0]
        for node in reversed(game_map.topological_order.detach().cpu().tolist()):
            children = sorted(adjacency[node])
            if children:
                child_values = torch.stack(
                    [self._require_future(future[child]) for child in children]
                )
                query = self.child_query(
                    torch.cat([base[node], player_context])
                )
                scores = self.child_key(child_values).matmul(query)
                weights = torch.softmax(scores / math.sqrt(self.config.hidden_dim), dim=0)
                child_context = torch.sum(
                    weights.unsqueeze(1) * self.child_value(child_values), dim=0
                )
            else:
                child_context = torch.zeros_like(player_context)
            update = self.node_update(
                torch.cat([base[node], child_context, player_context])
            )
            future[node] = self.node_norm(base[node] + update)

        if future:
            future_nodes = torch.stack(
                [self._require_future(value) for value in future]
            )
        else:
            future_nodes = base
        global_embedding = self._pool_reachable(
            future_nodes,
            game_map.reachable_mask,
            player_context,
        )
        return EncodedMap(base, future_nodes, global_embedding)

    def _pool_reachable(
        self,
        nodes: Tensor,
        reachable_mask: Tensor,
        player_context: Tensor,
    ) -> Tensor:
        reachable = nodes[reachable_mask]
        if reachable.shape[0] == 0:
            pooled = self.empty_map_embedding
        else:
            query = self.pool_query(player_context)
            scores = self.pool_key(reachable).matmul(query)
            weights = torch.softmax(scores / math.sqrt(self.config.hidden_dim), dim=0)
            pooled = torch.sum(
                weights.unsqueeze(1) * self.pool_value(reachable), dim=0
            )
        return self.pool_norm(pooled + self.pool_context(player_context))

    @staticmethod
    def _adjacency(game_map: TokenizedMap) -> list[set[int]]:
        adjacency = [set() for _ in range(game_map.node_categorical.shape[0])]
        edges = game_map.edge_index.detach().cpu()
        for parent, child in zip(edges[0].tolist(), edges[1].tolist()):
            adjacency[parent].add(child)
        return adjacency

    @staticmethod
    def _require_future(value: Tensor | None) -> Tensor:
        if value is None:
            raise RuntimeError("topological order did not encode a child first")
        return value
