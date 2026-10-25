# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Vectorized row scatters and contiguous segments, precomputed once and reused every step (ADR-0009)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


class ScatterPlan:
    """Computes out[index[k]] += values[k] with one sort done up front and np.add.reduceat per call."""

    def __init__(self, index: np.ndarray, size: int) -> None:
        index = np.asarray(index, dtype=np.int64)
        if index.size and (index.min() < 0 or index.max() >= size):
            raise ValueError("scatter index outside the output")
        self.size = size
        self._order = np.argsort(index, kind="stable")
        self._buckets, self._starts = np.unique(index[self._order], return_index=True)

    def apply(self, values: np.ndarray) -> np.ndarray:
        output = np.zeros((self.size, *values.shape[1:]), dtype=values.dtype)
        if self._buckets.size:
            output[self._buckets] = np.add.reduceat(values[self._order], self._starts, axis=0)
        return output


@dataclass(frozen=True, eq=False)
class Route:
    """Weighted directed edges between rows, with scatter plans for both directions built on first use."""

    sources: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    node_count: int

    @property
    def edge_count(self) -> int:
        return int(self.sources.size)

    @cached_property
    def target_plan(self) -> ScatterPlan:
        return ScatterPlan(self.targets, self.node_count)

    @cached_property
    def source_plan(self) -> ScatterPlan:
        return ScatterPlan(self.sources, self.node_count)


class Segments:
    """Partition of rows into contiguous, non-empty segments (one per graph in a batch)."""

    def __init__(self, sizes: np.ndarray) -> None:
        sizes = np.asarray(sizes, dtype=np.int64)
        if sizes.ndim != 1 or sizes.size == 0 or sizes.min() < 1:
            raise ValueError("segments must be a non-empty list of positive sizes")
        self.sizes = sizes
        self.offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        self.ids = np.repeat(np.arange(sizes.size), sizes)

    @classmethod
    def single(cls, row_count: int) -> Segments:
        return cls(np.array([row_count]))

    @property
    def count(self) -> int:
        return int(self.sizes.size)

    @property
    def row_count(self) -> int:
        return int(self.ids.size)

    def sum(self, values: np.ndarray) -> np.ndarray:
        return np.add.reduceat(values, self.offsets, axis=0)

    def maximum(self, values: np.ndarray) -> np.ndarray:
        return np.maximum.reduceat(values, self.offsets, axis=0)

    def broadcast(self, per_segment: np.ndarray) -> np.ndarray:
        return per_segment[self.ids]

    def rows(self, segment: int) -> slice:
        start = int(self.offsets[segment])
        return slice(start, start + int(self.sizes[segment]))
