import numpy as np
import pytest

from gsdm.autograd import (
    Route,
    ScatterPlan,
    Segments,
    Tensor,
    default_dtype,
    get_default_dtype,
    propagate_along,
    segment_softmax,
    segment_sum,
)
from tests.gradient_check import assert_gradients_match


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(3)


def test_scatter_plan_matches_add_at(rng):
    index = rng.integers(0, 7, size=40)
    values = rng.normal(size=(40, 3))
    expected = np.zeros((9, 3))
    np.add.at(expected, index, values)
    np.testing.assert_allclose(ScatterPlan(index, size=9).apply(values), expected)


def test_scatter_plan_handles_no_rows_and_rejects_bad_index():
    np.testing.assert_array_equal(ScatterPlan(np.zeros(0, dtype=int), size=2).apply(np.zeros((0, 3))), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        ScatterPlan(np.array([0, 5]), size=3)


def test_segments_layout():
    segments = Segments(np.array([2, 1, 3]))
    np.testing.assert_array_equal(segments.offsets, [0, 2, 3])
    np.testing.assert_array_equal(segments.ids, [0, 0, 1, 2, 2, 2])
    assert (segments.count, segments.row_count) == (3, 6)
    assert segments.rows(2) == slice(3, 6)
    with pytest.raises(ValueError):
        Segments(np.array([2, 0]))


def test_segment_softmax_normalizes_each_segment_independently(rng):
    segments = Segments(np.array([2, 3]))
    output = segment_softmax(Tensor(rng.normal(size=(5, 2)) * 50), segments).data
    np.testing.assert_allclose(segments.sum(output), np.ones((2, 2)), rtol=1e-5)


def test_segment_sum_values():
    output = segment_sum(Tensor(np.arange(10.0).reshape(5, 2)), Segments(np.array([2, 3])))
    np.testing.assert_allclose(output.data, [[2.0, 4.0], [18.0, 21.0]])


@pytest.mark.usefixtures("float64_precision")
def test_segment_operation_gradients(rng):
    segments = Segments(np.array([2, 1, 3]))
    values = Tensor(rng.normal(size=(6, 2)), requires_grad=True)
    weights = rng.normal(size=(3, 2))
    assert_gradients_match(lambda: (segment_sum(segment_softmax(values, segments) * values, segments) * weights).sum(), [values])


@pytest.mark.usefixtures("float64_precision")
def test_propagate_along_gradients(rng):
    route = Route(np.array([0, 1, 2, 2, 3]), np.array([1, 1, 0, 3, 3]), np.array([0.5, 0.5, 1.0, 0.5, 0.5]), node_count=4)
    states = Tensor(rng.normal(size=(4, 3)), requires_grad=True)
    weights = rng.normal(size=(4, 3))
    assert_gradients_match(lambda: (propagate_along(states, route) * weights).sum(), [states])


def test_float32_is_the_default_and_float64_is_scoped():
    assert get_default_dtype() is np.float32
    assert Tensor(1.0).data.dtype == np.float32
    with default_dtype(np.float64):
        assert Tensor(1.0).data.dtype == np.float64
    assert Tensor(1.0).data.dtype == np.float32
    with pytest.raises(ValueError):
        with default_dtype(np.int32):
            pass


def test_float32_does_not_silently_upcast_through_operations(rng):
    route = Route(np.array([0, 1]), np.array([1, 0]), np.array([1.0, 1.0]), node_count=2)
    states = Tensor(rng.normal(size=(2, 2)), requires_grad=True)
    output = segment_sum(propagate_along(states, route) * 2.0, Segments.single(2))
    output.sum().backward()
    assert output.data.dtype == np.float32
    assert states.grad.dtype == np.float32
