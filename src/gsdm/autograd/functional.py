# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import numpy as np

from gsdm.autograd.scatter import Route, Segments
from gsdm.autograd.tensor import Tensor


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * values))


def relu(value: Tensor) -> Tensor:
    active = value.data > 0

    def backward(gradient: np.ndarray) -> None:
        value.accumulate_grad(gradient * active)

    return Tensor.from_operation(value.data * active, (value,), backward)


def tanh(value: Tensor) -> Tensor:
    output = np.tanh(value.data)

    def backward(gradient: np.ndarray) -> None:
        value.accumulate_grad(gradient * (1.0 - output**2))

    return Tensor.from_operation(output, (value,), backward)


def sigmoid(value: Tensor) -> Tensor:
    output = stable_sigmoid(value.data)

    def backward(gradient: np.ndarray) -> None:
        value.accumulate_grad(gradient * output * (1.0 - output))

    return Tensor.from_operation(output, (value,), backward)


def softmax(value: Tensor, axis: int) -> Tensor:
    shifted = value.data - value.data.max(axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    output = exponentials / exponentials.sum(axis=axis, keepdims=True)

    def backward(gradient: np.ndarray) -> None:
        weighted_sum = (gradient * output).sum(axis=axis, keepdims=True)
        value.accumulate_grad(output * (gradient - weighted_sum))

    return Tensor.from_operation(output, (value,), backward)


def segment_softmax(value: Tensor, segments: Segments) -> Tensor:
    """Softmax over rows, computed independently inside each segment (column-wise)."""
    shifted = value.data - segments.broadcast(segments.maximum(value.data))
    exponentials = np.exp(shifted)
    output = exponentials / segments.broadcast(segments.sum(exponentials))

    def backward(gradient: np.ndarray) -> None:
        weighted_sum = segments.broadcast(segments.sum(gradient * output))
        value.accumulate_grad(output * (gradient - weighted_sum))

    return Tensor.from_operation(output, (value,), backward)


def segment_sum(value: Tensor, segments: Segments) -> Tensor:
    """Sums the rows of each segment: (rows, ...) -> (segments, ...)."""

    def backward(gradient: np.ndarray) -> None:
        value.accumulate_grad(segments.broadcast(gradient))

    return Tensor.from_operation(segments.sum(value.data), (value,), backward)


def propagate_along(node_states: Tensor, route: Route) -> Tensor:
    """Sparse message passing: out[target] += weight * node_states[source] for each edge of the route."""
    weights = route.weights.astype(node_states.data.dtype, copy=False)[:, None]
    output = route.target_plan.apply(node_states.data[route.sources] * weights)

    def backward(gradient: np.ndarray) -> None:
        node_states.accumulate_grad(route.source_plan.apply(gradient[route.targets] * weights))

    return Tensor.from_operation(output, (node_states,), backward)


def propagate(
    node_states: Tensor,
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> Tensor:
    """Same as propagate_along, for ad-hoc edges; plans are rebuilt on every call."""
    route = Route(np.asarray(sources), np.asarray(targets), np.asarray(weights), node_states.shape[0])
    return propagate_along(node_states, route)


def binary_cross_entropy_with_logits(logits: Tensor, targets: np.ndarray) -> Tensor:
    x = logits.data
    targets = np.asarray(targets, dtype=x.dtype)
    if targets.shape != logits.shape:
        raise ValueError(f"targets shape {targets.shape} does not match logits shape {logits.shape}")
    losses = np.maximum(x, 0.0) - x * targets + np.log1p(np.exp(-np.abs(x)))

    def backward(gradient: np.ndarray) -> None:
        logits.accumulate_grad(gradient * (stable_sigmoid(x) - targets) / x.size)

    return Tensor.from_operation(np.asarray(losses.mean()), (logits,), backward)
