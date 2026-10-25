import numpy as np
import pytest

from gsdm.autograd import (
    Tensor,
    binary_cross_entropy_with_logits,
    no_grad,
    propagate,
    relu,
    sigmoid,
    softmax,
    tanh,
)
from tests.gradient_check import assert_gradients_match

pytestmark = pytest.mark.usefixtures("float64_precision")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


def leaf(rng: np.random.Generator, *shape: int) -> Tensor:
    return Tensor(rng.normal(size=shape), requires_grad=True)


def test_add_with_broadcasting(rng):
    a, b = leaf(rng, 3, 4), leaf(rng, 4)
    assert_gradients_match(lambda: ((a + b) * (a + b)).sum(), [a, b])


def test_subtract_and_negate(rng):
    a, b = leaf(rng, 2, 3), leaf(rng, 2, 1)
    assert_gradients_match(lambda: ((a - b) * a - (-b)).sum(), [a, b])


def test_multiply_with_broadcasting(rng):
    a, b = leaf(rng, 3, 4), leaf(rng, 1, 4)
    assert_gradients_match(lambda: (a * b * a).sum(), [a, b])


def test_divide(rng):
    a = leaf(rng, 3, 2)
    b = Tensor(rng.uniform(1.0, 2.0, size=(3, 2)), requires_grad=True)
    assert_gradients_match(lambda: (a / b).sum(), [a, b])


def test_matmul(rng):
    a, b = leaf(rng, 3, 4), leaf(rng, 4, 2)
    assert_gradients_match(lambda: ((a @ b) * (a @ b)).sum(), [a, b])


def test_matmul_rejects_non_matrices(rng):
    with pytest.raises(ValueError):
        leaf(rng, 3) @ leaf(rng, 3)


def test_power(rng):
    a = Tensor(rng.uniform(0.5, 2.0, size=(4,)), requires_grad=True)
    assert_gradients_match(lambda: (a**-0.5).sum(), [a])


def test_transpose(rng):
    a, b = leaf(rng, 3, 2), leaf(rng, 3, 4)
    assert_gradients_match(lambda: ((a.T @ b) * (a.T @ b)).sum(), [a, b])


@pytest.mark.parametrize("axis, keepdims", [(None, False), (0, False), (1, True), (1, False)])
def test_sum_and_mean(rng, axis, keepdims):
    a = leaf(rng, 3, 4)
    assert_gradients_match(lambda: (a.sum(axis, keepdims) * a.mean(axis, keepdims)).sum(), [a])


@pytest.mark.parametrize("activation", [relu, tanh, sigmoid])
def test_activations(rng, activation):
    a = leaf(rng, 5, 3)
    weights = rng.normal(size=(5, 3))
    assert_gradients_match(lambda: (activation(a) * weights).sum(), [a])


@pytest.mark.parametrize("axis", [0, 1])
def test_softmax(rng, axis):
    a = leaf(rng, 4, 3)
    weights = rng.normal(size=(4, 3))
    assert_gradients_match(lambda: (softmax(a, axis) * weights).sum(), [a])


def test_softmax_is_stable_for_large_inputs():
    output = softmax(Tensor(np.array([[1000.0], [1001.0]])), axis=0)
    assert np.all(np.isfinite(output.data))
    assert output.data.sum() == pytest.approx(1.0)


def test_propagate_with_repeated_targets(rng):
    states = leaf(rng, 4, 3)
    sources = np.array([0, 1, 2, 2, 3])
    targets = np.array([1, 1, 0, 3, 3])
    edge_weights = np.array([0.5, 0.5, 1.0, 0.5, 0.5])
    weights = rng.normal(size=(4, 3))
    assert_gradients_match(lambda: (propagate(states, sources, targets, edge_weights) * weights).sum(), [states])


def test_propagate_aggregates_messages():
    states = Tensor(np.array([[1.0], [2.0], [4.0]]))
    output = propagate(states, np.array([0, 1]), np.array([2, 2]), np.array([0.5, 0.5]))
    np.testing.assert_allclose(output.data, [[0.0], [0.0], [1.5]])


def test_binary_cross_entropy_gradient(rng):
    logits = leaf(rng, 3)
    targets = np.array([1.0, 0.0, 1.0])
    assert_gradients_match(lambda: binary_cross_entropy_with_logits(logits, targets), [logits])


def test_binary_cross_entropy_matches_reference_and_is_stable():
    logits = Tensor(np.array([0.0, 800.0, -800.0]))
    loss = binary_cross_entropy_with_logits(logits, np.array([1.0, 1.0, 0.0]))
    assert float(loss.data) == pytest.approx(np.log(2.0) / 3.0)


def test_binary_cross_entropy_rejects_shape_mismatch(rng):
    with pytest.raises(ValueError):
        binary_cross_entropy_with_logits(leaf(rng, 3), np.zeros(2))


def test_gradients_accumulate_across_backward_calls(rng):
    a = leaf(rng, 2)
    (a * 2.0).sum().backward()
    (a * 3.0).sum().backward()
    np.testing.assert_allclose(a.grad, [5.0, 5.0])


def test_shared_subexpression_receives_all_gradient_paths(rng):
    a = leaf(rng, 3)
    shared = a * a
    (shared + shared * 2.0).sum().backward()
    np.testing.assert_allclose(a.grad, 6.0 * a.data)


def test_no_grad_disables_graph_construction(rng):
    a = leaf(rng, 2)
    with no_grad():
        result = (a * 2.0).sum()
    assert not result.requires_grad


def test_backward_requires_scalar(rng):
    with pytest.raises(ValueError):
        (leaf(rng, 2) * 2.0).backward()
