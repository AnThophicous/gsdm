# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class Relation:
    """Directed edges of one type, with mean-aggregation weights (1 / in-degree of the target)."""

    sources: np.ndarray
    targets: np.ndarray
    weights: np.ndarray

    @classmethod
    def from_edges(cls, edges: Sequence[tuple[int, int]], node_count: int) -> Relation:
        pairs = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        sources, targets = pairs[:, 0], pairs[:, 1]
        if pairs.size and (pairs.min() < 0 or pairs.max() >= node_count):
            raise ValueError("edge refers to a node outside the graph")
        in_degree = np.bincount(targets, minlength=node_count).astype(np.float64)
        weights = 1.0 / in_degree[targets] if targets.size else np.zeros(0)
        return cls(sources, targets, weights)

    @property
    def edge_count(self) -> int:
        return int(self.sources.size)


class StructuralGraph:
    """Heterogeneous graph consumed by GSDM: node feature matrix plus typed relations."""

    def __init__(self, node_features: np.ndarray, relations: Sequence[Relation]) -> None:
        features = np.asarray(node_features, dtype=np.float64)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("node_features must be a non-empty 2-D matrix")
        self._validate_relations(relations, features.shape[0])
        self.node_features = features
        self.relations = tuple(relations)

    @classmethod
    def from_edge_lists(
        cls, node_features: np.ndarray, edge_lists: Sequence[Sequence[tuple[int, int]]]
    ) -> StructuralGraph:
        node_count = np.asarray(node_features).shape[0]
        return cls(node_features, [Relation.from_edges(edges, node_count) for edges in edge_lists])

    @property
    def node_count(self) -> int:
        return self.node_features.shape[0]

    @property
    def feature_size(self) -> int:
        return self.node_features.shape[1]

    @property
    def relation_count(self) -> int:
        return len(self.relations)

    @staticmethod
    def _validate_relations(relations: Sequence[Relation], node_count: int) -> None:
        for relation in relations:
            if relation.edge_count and (relation.sources.max() >= node_count or relation.targets.max() >= node_count):
                raise ValueError("edge refers to a node outside the graph")
