import numpy as np
import pytest

from gsdm.autograd import stable_sigmoid
from gsdm.calibration import calibrated_probabilities, expected_calibration_error, fit_temperatures


def overconfident_dataset(true_temperature: float, size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 4.0, size=size)
    labels = (rng.uniform(size=size) < stable_sigmoid(logits / true_temperature)).astype(float)
    return logits, labels


def test_fit_temperatures_recovers_known_temperature_per_decision():
    first_logits, first_labels = overconfident_dataset(2.5, 20000, seed=1)
    second_logits, second_labels = overconfident_dataset(0.7, 20000, seed=2)
    temperatures = fit_temperatures(
        np.column_stack([first_logits, second_logits]), np.column_stack([first_labels, second_labels])
    )
    np.testing.assert_allclose(temperatures, [2.5, 0.7], rtol=0.08)


def test_calibration_reduces_expected_calibration_error():
    logits, labels = overconfident_dataset(3.0, 20000, seed=3)
    temperatures = fit_temperatures(logits, labels)
    before = expected_calibration_error(stable_sigmoid(logits), labels)
    after = expected_calibration_error(calibrated_probabilities(logits, temperatures[0]), labels)
    assert after < before / 3


def test_expected_calibration_error_of_perfect_and_worst_predictions():
    assert expected_calibration_error(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 0.0
    assert expected_calibration_error(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


def test_expected_calibration_error_weights_bins_by_size():
    probabilities = np.array([0.25, 0.25, 0.25, 0.25])
    labels = np.array([1.0, 0.0, 0.0, 0.0])
    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.0)
    assert expected_calibration_error(np.array([0.9, 0.9]), np.array([1.0, 0.0])) == pytest.approx(0.4)


def test_calibration_rejects_invalid_input():
    with pytest.raises(ValueError):
        fit_temperatures(np.zeros((0, 2)), np.zeros((0, 2)))
    with pytest.raises(ValueError):
        fit_temperatures(np.zeros((3, 2)), np.zeros((3, 1)))
    with pytest.raises(ValueError):
        expected_calibration_error(np.zeros(2), np.zeros(3))
