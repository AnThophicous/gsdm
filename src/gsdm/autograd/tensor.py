# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np

Backward = Callable[[np.ndarray], None]

_gradient_tracking_enabled = True


@contextmanager
def no_grad() -> Iterator[None]:
    global _gradient_tracking_enabled
    previous = _gradient_tracking_enabled
    _gradient_tracking_enabled = False
    try:
        yield
    finally:
        _gradient_tracking_enabled = previous


class Tensor:
    __array_priority__ = 1000

    def __init__(self, data: np.ndarray | float, requires_grad: bool = False) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.grad: np.ndarray | None = None
        self.requires_grad = requires_grad
        self._parents: tuple[Tensor, ...] = ()
        self._backward: Backward | None = None

    @classmethod
    def from_operation(
        cls, data: np.ndarray, parents: tuple[Tensor, ...], backward: Backward
    ) -> Tensor:
        result = cls(data)
        tracked_parents = tuple(parent for parent in parents if parent.requires_grad)
        if _gradient_tracking_enabled and tracked_parents:
            result.requires_grad = True
            result._parents = tracked_parents
            result._backward = backward
        return result

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def T(self) -> Tensor:
        return transpose(self)

    def accumulate_grad(self, gradient: np.ndarray) -> None:
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad = np.array(gradient, dtype=np.float64, copy=True)
        else:
            self.grad = self.grad + gradient

    def backward(self) -> None:
        if self.data.size != 1:
            raise ValueError("backward() requires a scalar tensor")
        self.grad = np.ones_like(self.data)
        for node in reversed(self._topological_order()):
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)

    def _topological_order(self) -> list[Tensor]:
        order: list[Tensor] = []
        visited: set[int] = set()
        pending: list[tuple[Tensor, bool]] = [(self, False)]
        while pending:
            node, parents_done = pending.pop()
            if parents_done:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            pending.append((node, True))
            pending.extend((parent, False) for parent in node._parents if id(parent) not in visited)
        return order

    def __add__(self, other: Tensor | np.ndarray | float) -> Tensor:
        return add(self, other)

    def __radd__(self, other: np.ndarray | float) -> Tensor:
        return add(other, self)

    def __sub__(self, other: Tensor | np.ndarray | float) -> Tensor:
        return add(self, negate(as_tensor(other)))

    def __rsub__(self, other: np.ndarray | float) -> Tensor:
        return add(other, negate(self))

    def __neg__(self) -> Tensor:
        return negate(self)

    def __mul__(self, other: Tensor | np.ndarray | float) -> Tensor:
        return multiply(self, other)

    def __rmul__(self, other: np.ndarray | float) -> Tensor:
        return multiply(other, self)

    def __truediv__(self, other: Tensor | np.ndarray | float) -> Tensor:
        return divide(self, other)

    def __rtruediv__(self, other: np.ndarray | float) -> Tensor:
        return divide(other, self)

    def __matmul__(self, other: Tensor | np.ndarray) -> Tensor:
        return matmul(self, other)

    def __rmatmul__(self, other: np.ndarray) -> Tensor:
        return matmul(other, self)

    def __pow__(self, exponent: float) -> Tensor:
        return power(self, exponent)

    def sum(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        return reduce_sum(self, axis, keepdims)

    def mean(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        return reduce_mean(self, axis, keepdims)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"


def as_tensor(value: Tensor | np.ndarray | float) -> Tensor:
    return value if isinstance(value, Tensor) else Tensor(value)


def sum_to_shape(gradient: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient


def add(left: Tensor | np.ndarray | float, right: Tensor | np.ndarray | float) -> Tensor:
    a, b = as_tensor(left), as_tensor(right)

    def backward(gradient: np.ndarray) -> None:
        a.accumulate_grad(sum_to_shape(gradient, a.shape))
        b.accumulate_grad(sum_to_shape(gradient, b.shape))

    return Tensor.from_operation(a.data + b.data, (a, b), backward)


def negate(value: Tensor) -> Tensor:
    def backward(gradient: np.ndarray) -> None:
        value.accumulate_grad(-gradient)

    return Tensor.from_operation(-value.data, (value,), backward)


def multiply(left: Tensor | np.ndarray | float, right: Tensor | np.ndarray | float) -> Tensor:
    a, b = as_tensor(left), as_tensor(right)

    def backward(gradient: np.ndarray) -> None:
        a.accumulate_grad(sum_to_shape(gradient * b.data, a.shape))
        b.accumulate_grad(sum_to_shape(gradient * a.data, b.shape))

    return Tensor.from_operation(a.data * b.data, (a, b), backward)


def divide(numerator: Tensor | np.ndarray | float, denominator: Tensor | np.ndarray | float) -> Tensor:
    a, b = as_tensor(numerator), as_tensor(denominator)

    def backward(gradient: np.ndarray) -> None:
        a.accumulate_grad(sum_to_shape(gradient / b.data, a.shape))
        b.accumulate_grad(sum_to_shape(-gradient * a.data / (b.data**2), b.shape))

    return Tensor.from_operation(a.data / b.data, (a, b), backward)


def matmul(left: Tensor | np.ndarray, right: Tensor | np.ndarray) -> Tensor:
    a, b = as_tensor(left), as_tensor(right)
    if a.data.ndim != 2 or b.data.ndim != 2:
        raise ValueError("matmul supports 2-D tensors only")

    def backward(gradient: np.ndarray) -> None:
        a.accumulate_grad(gradient @ b.data.T)
        b.accumulate_grad(a.data.T @ gradient)

    return Tensor.from_operation(a.data @ b.data, (a, b), backward)


def power(base: Tensor, exponent: float) -> Tensor:
    def backward(gradient: np.ndarray) -> None:
        base.accumulate_grad(gradient * exponent * base.data ** (exponent - 1))

    return Tensor.from_operation(base.data**exponent, (base,), backward)


def transpose(value: Tensor) -> Tensor:
    def backward(gradient: np.ndarray) -> None:
        value.accumulate_grad(gradient.T)

    return Tensor.from_operation(value.data.T, (value,), backward)


def reduce_sum(value: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
    def backward(gradient: np.ndarray) -> None:
        if axis is not None and not keepdims:
            gradient = np.expand_dims(gradient, axis)
        value.accumulate_grad(np.broadcast_to(gradient, value.shape))

    return Tensor.from_operation(value.data.sum(axis=axis, keepdims=keepdims), (value,), backward)


def reduce_mean(value: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
    count = value.data.size if axis is None else value.data.shape[axis]
    return reduce_sum(value, axis, keepdims) / float(count)
