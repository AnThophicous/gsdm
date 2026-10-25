# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import numpy as np

from gsdm.autograd import Tensor
from gsdm.nn.module import Module, Parameter, glorot_uniform


class Linear(Module):
    def __init__(self, input_size: int, output_size: int, rng: np.random.Generator) -> None:
        self.weight = Parameter(glorot_uniform(rng, input_size, output_size))
        self.bias = Parameter(np.zeros(output_size))

    def __call__(self, inputs: Tensor) -> Tensor:
        return inputs @ self.weight + self.bias


class LayerNorm(Module):
    def __init__(self, size: int, epsilon: float = 1e-5) -> None:
        self.gain = Parameter(np.ones(size))
        self.bias = Parameter(np.zeros(size))
        self.epsilon = epsilon

    def __call__(self, inputs: Tensor) -> Tensor:
        centered = inputs - inputs.mean(axis=1, keepdims=True)
        variance = (centered * centered).mean(axis=1, keepdims=True)
        normalized = centered * (variance + self.epsilon) ** -0.5
        return normalized * self.gain + self.bias
