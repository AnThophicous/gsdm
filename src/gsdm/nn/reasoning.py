# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import numpy as np

from gsdm.autograd import Tensor, propagate, relu, tanh
from gsdm.graph import StructuralGraph
from gsdm.nn.layers import LayerNorm, Linear
from gsdm.nn.module import Module, Parameter, glorot_uniform


class ReasoningStep(Module):
    """One round of latent reasoning (ADR-0007): typed message passing with a residual update."""

    def __init__(self, hidden_size: int, relation_count: int, rng: np.random.Generator) -> None:
        self.relation_weights = [Parameter(glorot_uniform(rng, hidden_size, hidden_size)) for _ in range(relation_count)]
        self.self_update = Linear(hidden_size, hidden_size, rng)
        self.normalization = LayerNorm(hidden_size)

    def __call__(self, states: Tensor, graph: StructuralGraph) -> Tensor:
        if graph.relation_count != len(self.relation_weights):
            raise ValueError(f"graph has {graph.relation_count} relations, model expects {len(self.relation_weights)}")
        update = self.self_update(states)
        for relation, weight in zip(graph.relations, self.relation_weights):
            if relation.edge_count:
                messages = propagate(states, relation.sources, relation.targets, relation.weights)
                update = update + messages @ weight
        return self.normalization(states + relu(update))


class LatentReasoningEncoder(Module):
    """Encodes a graph by repeating the same reasoning step; the step count can change at inference."""

    def __init__(
        self,
        feature_size: int,
        hidden_size: int,
        relation_count: int,
        reasoning_steps: int,
        rng: np.random.Generator,
    ) -> None:
        if reasoning_steps < 1:
            raise ValueError("reasoning_steps must be at least 1")
        self.input_projection = Linear(feature_size, hidden_size, rng)
        self.step = ReasoningStep(hidden_size, relation_count, rng)
        self.reasoning_steps = reasoning_steps

    def __call__(self, graph: StructuralGraph, reasoning_steps: int | None = None) -> Tensor:
        steps = self.reasoning_steps if reasoning_steps is None else reasoning_steps
        if steps < 1:
            raise ValueError("reasoning_steps must be at least 1")
        states = tanh(self.input_projection(Tensor(graph.node_features)))
        for _ in range(steps):
            states = self.step(states, graph)
        return states
