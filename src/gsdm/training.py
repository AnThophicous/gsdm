# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from gsdm.autograd import binary_cross_entropy_with_logits, no_grad
from gsdm.graph import StructuralGraph
from gsdm.nn import GSDModel
from gsdm.optim import Adam, clip_gradient_norm


@dataclass(frozen=True, eq=False)
class TrainingExample:
    graph: StructuralGraph
    labels: np.ndarray


class Trainer:
    def __init__(
        self,
        model: GSDModel,
        optimizer: Adam,
        batch_size: int = 8,
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

    def _train_batch(self, batch: Sequence[TrainingExample]) -> float:
        self.optimizer.zero_grad()
        batch_loss = 0.0
        for example in batch:
            loss = binary_cross_entropy_with_logits(self.model(example.graph).logits, example.labels) / len(batch)
            loss.backward()
            batch_loss += float(loss.data)
        clip_gradient_norm(self.optimizer.parameters, self.max_gradient_norm)
        self.optimizer.step()
        return batch_loss


def predict_logits(model: GSDModel, graphs: Sequence[StructuralGraph]) -> np.ndarray:
    with no_grad():
        return np.array([model(graph).logits.data for graph in graphs]).reshape(len(graphs), model.config.decision_count)
