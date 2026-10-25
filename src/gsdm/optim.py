# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from gsdm.nn.module import Parameter


class Adam:
    def __init__(
        self,
        parameters: Sequence[Parameter],
        learning_rate: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.beta1, self.beta2 = betas
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.step_count = 0
        self._first_moments = [np.zeros_like(p.data) for p in self.parameters]
        self._second_moments = [np.zeros_like(p.data) for p in self.parameters]

    def step(self) -> None:
        self.step_count += 1
        first_correction = 1.0 - self.beta1**self.step_count
        second_correction = 1.0 - self.beta2**self.step_count
        for parameter, first, second in zip(self.parameters, self._first_moments, self._second_moments):
            if parameter.grad is None:
                continue
            first *= self.beta1
            first += (1.0 - self.beta1) * parameter.grad
            second *= self.beta2
            second += (1.0 - self.beta2) * parameter.grad**2
            update = (first / first_correction) / (np.sqrt(second / second_correction) + self.epsilon)
            parameter.data = parameter.data - self.learning_rate * (update + self.weight_decay * parameter.data)

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "first_moments": [moment.copy() for moment in self._first_moments],
            "second_moments": [moment.copy() for moment in self._second_moments],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        first, second = state["first_moments"], state["second_moments"]
        if len(first) != len(self.parameters) or len(second) != len(self.parameters):
            raise ValueError("optimizer state does not match the parameter list")
        for parameter, first_moment, second_moment in zip(self.parameters, first, second):
            if first_moment.shape != parameter.shape or second_moment.shape != parameter.shape:
                raise ValueError("optimizer state shape does not match its parameter")
        self.step_count = int(state["step_count"])
        self._first_moments = [np.array(m, dtype=p.data.dtype) for m, p in zip(first, self.parameters)]
        self._second_moments = [np.array(m, dtype=p.data.dtype) for m, p in zip(second, self.parameters)]


def clip_gradient_norm(parameters: Sequence[Parameter], max_norm: float) -> float:
    gradients = [p.grad for p in parameters if p.grad is not None]
    total_norm = float(np.sqrt(sum(float((g**2).sum()) for g in gradients)))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad = parameter.grad * scale
    return total_norm
