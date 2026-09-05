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
        base = self.node_type_embedding(game_map.node_categorical[:, 0])
        numeric = torch.cat(
            [game_map.node_numeric, game_map.node_numeric_mask.to(torch.float32)],
            dim=1,
        )
        base = base + self.node_numeric_projection(numeric)

        adjacency = self._adjacency(game_map)
        future_nodes = self._propagate_from_sinks(base, adjacency, player_context)
        global_embedding = self._pool_reachable(
            future_nodes,
            game_map.reachable_mask,
            player_context,
        )
        return EncodedMap(base, future_nodes, global_embedding)

    def _propagate_from_sinks(
        self,
        base: Tensor,
        adjacency: list[set[int]],
        player_context: Tensor,
    ) -> Tensor:
        """Aggregate every node's reachable future in one pass per DAG level.

        Nodes are grouped by their longest distance to a sink, so a whole level
        can attend over its children at once: every child sits at a strictly
        lower level and is therefore already final.
        """
        if base.shape[0] == 0:
            return base
        future = torch.zeros_like(base)
        for level in self._levels(adjacency):
            nodes = torch.tensor(level, dtype=torch.long, device=base.device)
            node_base = base.index_select(0, nodes)
            context = player_context.expand(len(level), -1)
            children = [sorted(adjacency[node]) for node in level]
            width = max(len(row) for row in children)
            if width:
                padded = torch.tensor(
                    [row + [0] * (width - len(row)) for row in children],
                    dtype=torch.long,
                    device=base.device,
                )
                mask = torch.tensor(
                    [
                        [True] * len(row) + [False] * (width - len(row))
                        for row in children
                    ],
                    dtype=torch.bool,
                    device=base.device,
                )
                values = future.index_select(0, padded.reshape(-1)).view(
                    len(level), width, -1
                )
                query = self.child_query(torch.cat([node_base, context], dim=1))
                scores = torch.einsum(
                    "nch,nh->nc", self.child_key(values), query
                ) / math.sqrt(self.config.hidden_dim)
                weights = torch.softmax(
                    scores.masked_fill(~mask, float("-inf")), dim=1
                )
                child_context = torch.einsum(
                    "nc,nch->nh", weights, self.child_value(values)
                )
            else:
                child_context = torch.zeros_like(node_base)
            update = self.node_update(
                torch.cat([node_base, child_context, context], dim=1)
            )
            future = future.index_copy(
                0, nodes, self.node_norm(node_base + update)
            )
        return future

    @staticmethod
    def _levels(adjacency: list[set[int]]) -> list[list[int]]:
        """Group nodes by longest distance to a sink, sinks first."""
        depths = [-1] * len(adjacency)
        for node in range(len(adjacency)):
            _resolve_depth(node, adjacency, depths)
        levels: list[list[int]] = [[] for _ in range(max(depths) + 1)]
        for node, depth in enumerate(depths):
            levels[depth].append(node)
        return levels

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


def _resolve_depth(
    start: int,
    adjacency: list[set[int]],
    depths: list[int],
) -> None:
    """Set the longest distance from ``start`` to a sink, and from its subtree."""
    stack = [start]
    while stack:
        node = stack[-1]
        if depths[node] >= 0:
            stack.pop()
            continue
        pending = [child for child in adjacency[node] if depths[child] < 0]
        if pending:
            stack.extend(pending)
            continue
        depths[node] = 1 + max(
            (depths[child] for child in adjacency[node]),
            default=-1,
        )
        stack.pop()
