from collections.abc import Callable, Sequence

import numpy as np

from gsdm.autograd import Tensor


def assert_gradients_match(
    function: Callable[[], Tensor],
    leaves: Sequence[Tensor],
    step: float = 1e-6,
    tolerance: float = 1e-6,
) -> None:
    for leaf in leaves:
        leaf.grad = None
    function().backward()
    for leaf in leaves:
        analytic = leaf.grad if leaf.grad is not None else np.zeros_like(leaf.data)
        numeric = _numeric_gradient(function, leaf, step)
        scale = max(1.0, float(np.abs(numeric).max()), float(np.abs(analytic).max()))
        error = float(np.abs(analytic - numeric).max()) / scale
        assert error < tolerance, f"gradient mismatch for leaf {leaf.shape}: relative error {error:.2e}"


def _numeric_gradient(function: Callable[[], Tensor], leaf: Tensor, step: float) -> np.ndarray:
    gradient = np.zeros_like(leaf.data)
    for index in np.ndindex(leaf.data.shape):
        original = leaf.data[index]
        leaf.data[index] = original + step
        loss_plus = float(function().data)
        leaf.data[index] = original - step
        loss_minus = float(function().data)
        leaf.data[index] = original
        gradient[index] = (loss_plus - loss_minus) / (2 * step)
    return gradient
