# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Per-decision temperature scaling and expected calibration error (ADR-0005)."""

from __future__ import annotations

import math

import numpy as np

from gsdm.autograd import stable_sigmoid

MINIMUM_TEMPERATURE = 0.05
MAXIMUM_TEMPERATURE = 20.0
_GOLDEN_RATIO_CONJUGATE = (math.sqrt(5.0) - 1.0) / 2.0


def calibrated_probabilities(logits: np.ndarray, temperatures: np.ndarray) -> np.ndarray:
    return stable_sigmoid(np.asarray(logits) / np.asarray(temperatures))


def fit_temperatures(logits: np.ndarray, labels: np.ndarray, iterations: int = 100) -> np.ndarray:
    logits, labels = _as_matrix(logits), _as_matrix(labels)
    if logits.shape != labels.shape or logits.shape[0] == 0:
        raise ValueError("logits and labels must be non-empty matrices of the same shape")
    return np.array(
        [_fit_single_temperature(logits[:, column], labels[:, column], iterations) for column in range(logits.shape[1])]
    )


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bin_count: int = 10) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    if probabilities.shape != labels.shape or probabilities.size == 0:
        raise ValueError("probabilities and labels must be non-empty and of the same size")
    bins = np.minimum((probabilities * bin_count).astype(int), bin_count - 1)
    error = 0.0
    for bin_index in range(bin_count):
        in_bin = bins == bin_index
        if in_bin.any():
            gap = abs(labels[in_bin].mean() - probabilities[in_bin].mean())
            error += in_bin.mean() * gap
    return float(error)


def _fit_single_temperature(logits: np.ndarray, labels: np.ndarray, iterations: int) -> float:
    def loss_at(log_temperature: float) -> float:
        return _binary_cross_entropy(logits / math.exp(log_temperature), labels)

    low, high = math.log(MINIMUM_TEMPERATURE), math.log(MAXIMUM_TEMPERATURE)
    left = high - _GOLDEN_RATIO_CONJUGATE * (high - low)
    right = low + _GOLDEN_RATIO_CONJUGATE * (high - low)
    left_loss, right_loss = loss_at(left), loss_at(right)
    for _ in range(iterations):
        if left_loss < right_loss:
            high, right, right_loss = right, left, left_loss
            left = high - _GOLDEN_RATIO_CONJUGATE * (high - low)
            left_loss = loss_at(left)
        else:
            low, left, left_loss = left, right, right_loss
            right = low + _GOLDEN_RATIO_CONJUGATE * (high - low)
            right_loss = loss_at(right)
    return math.exp((low + high) / 2.0)


def _binary_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.maximum(logits, 0.0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))))


def _as_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    return matrix.reshape(-1, 1) if matrix.ndim == 1 else matrix
