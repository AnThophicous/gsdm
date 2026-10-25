# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gsdm.autograd import Tensor, softmax
from gsdm.nn.layers import Linear
from gsdm.nn.module import Module, Parameter, glorot_uniform


@dataclass(frozen=True)
class DecisionOutput:
    """logits: one per decision. attention: (nodes, decisions) weights; argmax over nodes is the evidence node."""

    logits: Tensor
    attention: np.ndarray

    def evidence_node(self, decision_index: int) -> int:
        return int(np.argmax(self.attention[:, decision_index]))


class AttentionDecisionHead(Module):
    """Per-decision attention pooling over nodes followed by a linear decision (ADR-0008, item 7)."""

    def __init__(self, hidden_size: int, decision_count: int, rng: np.random.Generator) -> None:
        self.attention_scores = Linear(hidden_size, decision_count, rng)
        self.decision_weights = Parameter(glorot_uniform(rng, decision_count, hidden_size))
        self.decision_bias = Parameter(np.zeros(decision_count))

    def __call__(self, node_states: Tensor) -> DecisionOutput:
        attention = softmax(self.attention_scores(node_states), axis=0)
        pooled = attention.T @ node_states
        logits = (pooled * self.decision_weights).sum(axis=1) + self.decision_bias
        return DecisionOutput(logits=logits, attention=attention.data.copy())
