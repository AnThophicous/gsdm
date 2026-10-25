# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np

from gsdm.autograd import Tensor


class Parameter(Tensor):
    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data, requires_grad=True)


class Module:
    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        for name, value in vars(self).items():
            yield from _named_parameters_of(value, f"{prefix}{name}")

    def parameters(self) -> list[Parameter]:
        return [parameter for _, parameter in self.named_parameters()]

    def parameter_count(self) -> int:
        return sum(parameter.data.size for parameter in self.parameters())

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.grad = None

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: parameter.data.copy() for name, parameter in self.named_parameters()}

    def load_state_dict(self, state: Mapping[str, np.ndarray]) -> None:
        own = dict(self.named_parameters())
        if set(own) != set(state):
            missing, unexpected = sorted(set(own) - set(state)), sorted(set(state) - set(own))
            raise ValueError(f"state mismatch: missing={missing} unexpected={unexpected}")
        for name, parameter in own.items():
            if parameter.shape != state[name].shape:
                raise ValueError(f"shape mismatch for {name}: {parameter.shape} != {state[name].shape}")
        for name, parameter in own.items():
            parameter.data = np.array(state[name], dtype=parameter.data.dtype, copy=True)


def _named_parameters_of(value: object, name: str) -> Iterator[tuple[str, Parameter]]:
    if isinstance(value, Parameter):
        yield name, value
    elif isinstance(value, Module):
        yield from value.named_parameters(f"{name}.")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _named_parameters_of(item, f"{name}.{index}")


def glorot_uniform(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))
