# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from gsdm.graph import StructuralGraph
from gsdm.nn.decision import AttentionDecisionHead, BatchDecisionOutput, DecisionOutput
from gsdm.nn.module import Module
from gsdm.nn.reasoning import LatentReasoningEncoder


@dataclass(frozen=True)
class GSDMConfig:
    feature_size: int
    relation_count: int
    decision_count: int
    hidden_size: int = 64
    reasoning_steps: int = 4
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> GSDMConfig:
        return cls(**values)


class GSDModel(Module):
    """Graph-Structural Decision Model: graph in, calibratable decisions out, no text generation."""

    def __init__(self, config: GSDMConfig) -> None:
        rng = np.random.default_rng(config.seed)
        self.config = config
        self.encoder = LatentReasoningEncoder(
            config.feature_size, config.hidden_size, config.relation_count, config.reasoning_steps, rng
        )
        self.head = AttentionDecisionHead(config.hidden_size, config.decision_count, rng)

    def __call__(self, graph: StructuralGraph, reasoning_steps: int | None = None) -> DecisionOutput:
        if graph.graph_count != 1:
            raise ValueError("use forward_batch() for a batch of graphs")
        output = self.forward_batch(graph, reasoning_steps)
        return DecisionOutput(logits=output.logits.reshape(self.config.decision_count), attention=output.attention)

    def forward_batch(self, batch: StructuralGraph, reasoning_steps: int | None = None) -> BatchDecisionOutput:
        if batch.feature_size != self.config.feature_size:
            raise ValueError(f"graph has {batch.feature_size} features, model expects {self.config.feature_size}")
        return self.head(self.encoder(batch, reasoning_steps), batch.segments)
