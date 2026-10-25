# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gsdm.autograd import Segments, Tensor, segment_softmax, segment_sum
from gsdm.nn.layers import Linear
from gsdm.nn.module import Module, Parameter, glorot_uniform


@dataclass(frozen=True, eq=False)
class DecisionOutput:
    """logits: one per decision. attention: (nodes, decisions) weights; argmax over nodes is the evidence node."""

    logits: Tensor
    attention: np.ndarray

    def evidence_node(self, decision_index: int) -> int:
        return int(np.argmax(self.attention[:, decision_index]))


@dataclass(frozen=True, eq=False)
class BatchDecisionOutput:
    """logits: (graphs, decisions). attention: (all nodes, decisions), normalized inside each graph."""

    logits: Tensor
    attention: np.ndarray
    segments: Segments

    def evidence_node(self, graph_index: int, decision_index: int) -> int:
        """Index of the evidence node within its own graph."""
        rows = self.segments.rows(graph_index)
        return int(np.argmax(self.attention[rows, decision_index]))


class AttentionDecisionHead(Module):
    """Per-decision attention pooling over each graph's nodes, then a linear decision (ADR-0008 item 7, ADR-0009 item 2)."""

    def __init__(self, hidden_size: int, decision_count: int, rng: np.random.Generator) -> None:
        self.attention_scores = Linear(hidden_size, decision_count, rng)
        self.decision_weights = Parameter(glorot_uniform(rng, decision_count, hidden_size))
        self.decision_bias = Parameter(np.zeros(decision_count))

    def __call__(self, node_states: Tensor, segments: Segments) -> BatchDecisionOutput:
        attention = segment_softmax(self.attention_scores(node_states), segments)
        evidence = node_states @ self.decision_weights.T
        logits = segment_sum(attention * evidence, segments) + self.decision_bias
        return BatchDecisionOutput(logits=logits, attention=attention.data.copy(), segments=segments)
