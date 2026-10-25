# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from gsdm.autograd import binary_cross_entropy_with_logits, no_grad
from gsdm.graph import StructuralGraph
from gsdm.nn import GSDModel
from gsdm.optim import Adam, clip_gradient_norm


@dataclass(frozen=True, eq=False)
class TrainingExample:
    graph: StructuralGraph
    labels: np.ndarray


@dataclass(frozen=True, eq=False)
class Evaluation:
    loss: float
    logits: np.ndarray
    labels: np.ndarray


class Trainer:
    """One optimization step per mini-batch; the whole batch goes through the model as one disjoint graph."""

    def __init__(
        self,
        model: GSDModel,
        optimizer: Adam,
        batch_size: int = 32,
        max_gradient_norm: float = 1.0,
        seed: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_gradient_norm = max_gradient_norm
        self._rng = np.random.default_rng(seed)

    def fit(self, examples: Sequence[TrainingExample], epochs: int) -> list[float]:
        return [self.train_epoch(examples) for _ in range(epochs)]

    def train_epoch(self, examples: Sequence[TrainingExample]) -> float:
        if not examples:
            raise ValueError("cannot train on an empty dataset")
        order = self._rng.permutation(len(examples))
        total_loss = 0.0
        for start in range(0, len(order), self.batch_size):
            batch = [examples[index] for index in order[start : start + self.batch_size]]
            total_loss += self._train_batch(batch) * len(batch)
        return total_loss / len(examples)

    def rng_state(self) -> dict[str, Any]:
        return self._rng.bit_generator.state

    def restore_rng_state(self, state: dict[str, Any]) -> None:
        self._rng.bit_generator.state = state

    def _train_batch(self, batch: Sequence[TrainingExample]) -> float:
        self.optimizer.zero_grad()
        graph, labels = _stack(batch)
        loss = binary_cross_entropy_with_logits(self.model.forward_batch(graph).logits, labels)
        loss.backward()
        clip_gradient_norm(self.optimizer.parameters, self.max_gradient_norm)
        self.optimizer.step()
        return float(loss.data)


def predict_logits(model: GSDModel, graphs: Sequence[StructuralGraph], batch_size: int = 64) -> np.ndarray:
    rows = []
    with no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = StructuralGraph.batch(graphs[start : start + batch_size])
            rows.append(model.forward_batch(batch).logits.data)
    return np.concatenate(rows).astype(np.float64) if rows else np.zeros((0, model.config.decision_count))


def evaluate(model: GSDModel, examples: Sequence[TrainingExample], batch_size: int = 64) -> Evaluation:
    if not examples:
        raise ValueError("cannot evaluate an empty dataset")
    logits = predict_logits(model, [example.graph for example in examples], batch_size)
    labels = np.array([example.labels for example in examples], dtype=np.float64)
    losses = np.maximum(logits, 0.0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
    return Evaluation(loss=float(losses.mean()), logits=logits, labels=labels)


def _stack(batch: Sequence[TrainingExample]) -> tuple[StructuralGraph, np.ndarray]:
    return StructuralGraph.batch([example.graph for example in batch]), np.stack([example.labels for example in batch])
