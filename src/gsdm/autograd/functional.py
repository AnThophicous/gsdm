# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import numpy as np

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


def propagate(
    node_states: Tensor,
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> Tensor:
    """Sparse message passing: out[target] += weight * node_states[source] for each edge."""
    output = np.zeros_like(node_states.data)
    np.add.at(output, targets, node_states.data[sources] * weights[:, None])

    def backward(gradient: np.ndarray) -> None:
        source_gradient = np.zeros_like(node_states.data)
        np.add.at(source_gradient, sources, gradient[targets] * weights[:, None])
        node_states.accumulate_grad(source_gradient)

    return Tensor.from_operation(output, (node_states,), backward)


def binary_cross_entropy_with_logits(logits: Tensor, targets: np.ndarray) -> Tensor:
    targets = np.asarray(targets, dtype=np.float64)
    if targets.shape != logits.shape:
        raise ValueError(f"targets shape {targets.shape} does not match logits shape {logits.shape}")
    x = logits.data
    losses = np.maximum(x, 0.0) - x * targets + np.log1p(np.exp(-np.abs(x)))

    def backward(gradient: np.ndarray) -> None:
        logits.accumulate_grad(gradient * (stable_sigmoid(x) - targets) / x.size)

    return Tensor.from_operation(np.asarray(losses.mean()), (logits,), backward)
