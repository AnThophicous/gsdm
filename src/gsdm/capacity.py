# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Size and memory of a GSDM configuration, computed analytically without allocating the model (ADR-0009, items 11-13)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from gsdm.nn import GSDMConfig

BYTES_PER_FLOAT32 = 4
VALUES_PER_PARAMETER_IN_TRAINING = 4
ACTIVATIONS_PER_RELATION = 2
ACTIVATIONS_PER_STEP = 10


@dataclass(frozen=True)
class CapacityEstimate:
    parameters: int
    weights_bytes: int
    training_state_bytes: int
    activation_bytes_per_node: int

    def training_bytes(self, nodes_per_batch: int) -> int:
        return self.training_state_bytes + self.activation_bytes_per_node * nodes_per_batch


def parameter_count(config: GSDMConfig) -> int:
    features, hidden, relations, decisions = (
        config.feature_size,
        config.hidden_size,
        config.relation_count,
        config.decision_count,
    )
    input_projection = features * hidden + hidden
    reasoning_step = (relations + 1) * hidden * hidden + 3 * hidden
    decision_head = 2 * decisions * hidden + 2 * decisions
    return input_projection + reasoning_step + decision_head


def estimate(config: GSDMConfig) -> CapacityEstimate:
    """Training state = weight, gradient and two Adam moments per parameter. Activations are an approximation."""
    parameters = parameter_count(config)
    activations_per_step = ACTIVATIONS_PER_RELATION * config.relation_count + ACTIVATIONS_PER_STEP
    return CapacityEstimate(
        parameters=parameters,
        weights_bytes=parameters * BYTES_PER_FLOAT32,
        training_state_bytes=parameters * VALUES_PER_PARAMETER_IN_TRAINING * BYTES_PER_FLOAT32,
        activation_bytes_per_node=config.hidden_size * BYTES_PER_FLOAT32 * activations_per_step * config.reasoning_steps,
    )


def hidden_size_for(target_parameters: int, feature_size: int, relation_count: int, decision_count: int) -> int:
    """Smallest hidden size whose parameter count reaches target_parameters."""
    if target_parameters < 1:
        raise ValueError("target_parameters must be positive")
    a = relation_count + 1
    b = feature_size + 4 + 2 * decision_count
    hidden = max(1, int((-b + math.sqrt(b * b + 4 * a * target_parameters)) / (2 * a)))
    while hidden > 1 and _count(hidden - 1, feature_size, relation_count, decision_count) >= target_parameters:
        hidden -= 1
    while _count(hidden, feature_size, relation_count, decision_count) < target_parameters:
        hidden += 1
    return hidden


def format_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def _count(hidden: int, feature_size: int, relation_count: int, decision_count: int) -> int:
    return parameter_count(GSDMConfig(feature_size, relation_count, decision_count, hidden_size=hidden))
