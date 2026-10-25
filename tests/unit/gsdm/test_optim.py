import numpy as np

from gsdm.nn import Parameter
from gsdm.optim import Adam, clip_gradient_norm


def test_adam_minimizes_a_quadratic():
    target = np.array([3.0, -2.0])
    parameter = Parameter(np.zeros(2))
    optimizer = Adam([parameter], learning_rate=0.1)
    for _ in range(500):
        optimizer.zero_grad()
        difference = parameter - target
        (difference * difference).sum().backward()
        optimizer.step()
    np.testing.assert_allclose(parameter.data, target, atol=1e-3)


def test_adam_skips_parameters_without_gradient():
    parameter = Parameter(np.ones(2))
    Adam([parameter]).step()
    np.testing.assert_array_equal(parameter.data, np.ones(2))


def test_clip_gradient_norm_scales_down_large_gradients():
    first, second = Parameter(np.zeros(2)), Parameter(np.zeros(1))
    first.grad, second.grad = np.array([3.0, 0.0]), np.array([4.0])
    original_norm = clip_gradient_norm([first, second], max_norm=1.0)
    assert original_norm == 5.0
    clipped = np.sqrt((first.grad**2).sum() + (second.grad**2).sum())
    assert abs(clipped - 1.0) < 1e-9


def test_clip_gradient_norm_keeps_small_gradients():
    parameter = Parameter(np.zeros(2))
    parameter.grad = np.array([0.3, 0.4])
    clip_gradient_norm([parameter], max_norm=1.0)
    np.testing.assert_allclose(parameter.grad, [0.3, 0.4])
