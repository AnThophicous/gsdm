# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from gsdm.autograd.scatter import Route, Segments


@dataclass(frozen=True, eq=False)
class Relation(Route):
    """Directed edges of one type, with mean-aggregation weights (1 / in-degree of the target)."""

    @classmethod
    def from_edges(cls, edges: Sequence[tuple[int, int]], node_count: int) -> Relation:
        pairs = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        sources, targets = pairs[:, 0], pairs[:, 1]
        if pairs.size and (pairs.min() < 0 or pairs.max() >= node_count):
            raise ValueError("edge refers to a node outside the graph")
        in_degree = np.bincount(targets, minlength=node_count).astype(np.float64)
        weights = 1.0 / in_degree[targets] if targets.size else np.zeros(0)
        return cls(sources, targets, weights, node_count)


class StructuralGraph:
    """Heterogeneous graph consumed by GSDM: node features, typed relations and one segment per member graph."""

    def __init__(
        self,
        node_features: np.ndarray,
        relations: Sequence[Relation],
        segments: Segments | None = None,
    ) -> None:
        features = np.asarray(node_features)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("node_features must be a non-empty 2-D matrix")
        self._validate_relations(relations, features.shape[0])
        self.node_features = features
        self.relations = tuple(relations)
        self.segments = segments or Segments.single(features.shape[0])
        if self.segments.row_count != features.shape[0]:
            raise ValueError("segments must cover every node exactly once")

    @classmethod
    def from_edge_lists(
        cls, node_features: np.ndarray, edge_lists: Sequence[Sequence[tuple[int, int]]]
    ) -> StructuralGraph:
        node_count = np.asarray(node_features).shape[0]
        return cls(node_features, [Relation.from_edges(edges, node_count) for edges in edge_lists])

    @classmethod
    def batch(cls, graphs: Sequence[StructuralGraph]) -> StructuralGraph:
        """Disjoint union: node indices are shifted so each graph keeps its own edges and segment."""
        if not graphs:
            raise ValueError("cannot batch zero graphs")
        first = graphs[0]
        if any(g.feature_size != first.feature_size or g.relation_count != first.relation_count for g in graphs):
            raise ValueError("all graphs in a batch must share feature size and relation count")
        offsets = np.cumsum([0] + [g.node_count for g in graphs[:-1]])
        node_count = sum(g.node_count for g in graphs)
        relations = [
            Relation(
                np.concatenate([g.relations[r].sources + offset for g, offset in zip(graphs, offsets)]),
                np.concatenate([g.relations[r].targets + offset for g, offset in zip(graphs, offsets)]),
                np.concatenate([g.relations[r].weights for g in graphs]),
                node_count,
            )
            for r in range(first.relation_count)
        ]
        features = np.concatenate([g.node_features for g in graphs], axis=0)
        sizes = np.concatenate([g.segments.sizes for g in graphs])
        return cls(features, relations, Segments(sizes))

    @property
    def node_count(self) -> int:
        return self.node_features.shape[0]

    @property
    def feature_size(self) -> int:
        return self.node_features.shape[1]

    @property
    def relation_count(self) -> int:
        return len(self.relations)

    @property
    def graph_count(self) -> int:
        return self.segments.count

    @staticmethod
    def _validate_relations(relations: Sequence[Relation], node_count: int) -> None:
        for relation in relations:
            if relation.node_count != node_count:
                raise ValueError("relation was built for a graph of a different size")
            if relation.edge_count and (relation.sources.max() >= node_count or relation.targets.max() >= node_count):
                raise ValueError("edge refers to a node outside the graph")
