import numpy as np
import pytest

from gsdm.nn import Parameter
from gsdm.optim import Adam


def test_adam_state_round_trip_continues_identically():
    def run(optimizer: Adam, parameter: Parameter, steps: int) -> None:
        for _ in range(steps):
            optimizer.zero_grad()
            (parameter * parameter).sum().backward()
            optimizer.step()

    reference = Parameter(np.array([1.0, -2.0]))
    reference_optimizer = Adam([reference], learning_rate=0.1)
    run(reference_optimizer, reference, 6)

    first = Parameter(np.array([1.0, -2.0]))
    first_optimizer = Adam([first], learning_rate=0.1)
    run(first_optimizer, first, 3)
    second = Parameter(first.data.copy())
    second_optimizer = Adam([second], learning_rate=0.1)
    second_optimizer.load_state_dict(first_optimizer.state_dict())
    run(second_optimizer, second, 3)

    np.testing.assert_array_equal(second.data, reference.data)


def test_adam_state_rejects_mismatched_parameters():
    state = Adam([Parameter(np.zeros(2))]).state_dict()
    with pytest.raises(ValueError):
        Adam([Parameter(np.zeros(3))]).load_state_dict(state)
    with pytest.raises(ValueError):
        Adam([Parameter(np.zeros(2)), Parameter(np.zeros(2))]).load_state_dict(state)
