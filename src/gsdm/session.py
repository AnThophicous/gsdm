# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Epoch loop with validation, best-weight tracking, early stopping and observers (ADR-0009, items 7 and 10)."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from gsdm.autograd import stable_sigmoid
from gsdm.calibration import expected_calibration_error
from gsdm.training import Trainer, TrainingExample, evaluate


@dataclass(frozen=True)
class DecisionQuality:
    """Quality of one decision on the validation set; None when the ratio is undefined (nothing to divide by)."""

    name: str
    precision: float | None
    recall: float | None
    f1: float | None
    positives: int


@dataclass(frozen=True)
class EpochReport:
    epoch: int
    training_loss: float
    validation_loss: float | None
    calibration_error: float | None
    decisions: tuple[DecisionQuality, ...]
    seconds: float
    graphs_per_second: float
    is_best: bool

    @property
    def generalization_gap(self) -> float | None:
        return None if self.validation_loss is None else self.validation_loss - self.training_loss

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"generalization_gap": self.generalization_gap}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> EpochReport:
        fields = {key: value for key, value in values.items() if key != "generalization_gap"}
        fields["decisions"] = tuple(DecisionQuality(**decision) for decision in fields["decisions"])
        return cls(**fields)


class TrainingObserver:
    def on_epoch_end(self, report: EpochReport, session: TrainingSession) -> None:
        pass

    def on_training_end(self, session: TrainingSession) -> None:
        pass


class TrainingSession:
    def __init__(
        self,
        trainer: Trainer,
        training: Sequence[TrainingExample],
        validation: Sequence[TrainingExample] = (),
        decision_names: Sequence[str] | None = None,
        observers: Sequence[TrainingObserver] = (),
        patience: int | None = None,
    ) -> None:
        if not training:
            raise ValueError("cannot train on an empty dataset")
        if patience is not None and patience < 1:
            raise ValueError("patience must be at least 1")
        decision_count = trainer.model.config.decision_count
        self.trainer = trainer
        self.training = training
        self.validation = validation
        self.decision_names = tuple(decision_names or (f"decision-{i}" for i in range(decision_count)))
        if len(self.decision_names) != decision_count:
            raise ValueError("there must be one name per decision")
        self.observers = tuple(observers)
        self.patience = patience
        self.epoch = 0
        self.history: list[EpochReport] = []
        self.best_validation_loss = math.inf
        self.best_state: dict[str, np.ndarray] | None = None
        self.epochs_without_improvement = 0
        self.stopped_early = False
        self._using_best_weights = False

    @property
    def best_epoch(self) -> int | None:
        best = [report.epoch for report in self.history if report.is_best]
        return best[-1] if best else None

    @property
    def is_resumable(self) -> bool:
        return not self._using_best_weights

    def run(self, total_epochs: int) -> list[EpochReport]:
        """Trains until total_epochs have run in total, so a resumed session continues where it stopped."""
        if self._using_best_weights:
            raise RuntimeError("cannot keep training after use_best_weights()")
        while self.epoch < total_epochs and not self.stopped_early:
            self._run_epoch()
        for observer in self.observers:
            observer.on_training_end(self)
        return self.history

    def use_best_weights(self) -> None:
        """Loads the best validation weights into the model. Training cannot continue afterwards."""
        if self.best_state is not None:
            self.trainer.model.load_state_dict(self.best_state)
        self._using_best_weights = True

    def _run_epoch(self) -> None:
        started = time.perf_counter()
        training_loss = self.trainer.train_epoch(self.training)
        seconds = time.perf_counter() - started
        self.epoch += 1
        validation_loss, calibration_error, decisions = self._validate()
        is_best = self._track_best(training_loss if validation_loss is None else validation_loss)
        report = EpochReport(
            epoch=self.epoch,
            training_loss=training_loss,
            validation_loss=validation_loss,
            calibration_error=calibration_error,
            decisions=decisions,
            seconds=seconds,
            graphs_per_second=len(self.training) / seconds if seconds > 0 else math.inf,
            is_best=is_best,
        )
        self.history.append(report)
        for observer in self.observers:
            observer.on_epoch_end(report, self)

    def _validate(self) -> tuple[float | None, float | None, tuple[DecisionQuality, ...]]:
        if not self.validation:
            return None, None, ()
        evaluation = evaluate(self.trainer.model, self.validation)
        calibration_error = expected_calibration_error(stable_sigmoid(evaluation.logits), evaluation.labels)
        return evaluation.loss, calibration_error, decision_quality(evaluation.logits, evaluation.labels, self.decision_names)

    def _track_best(self, loss: float) -> bool:
        if loss < self.best_validation_loss:
            self.best_validation_loss = loss
            self.best_state = self.trainer.model.state_dict()
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        if self.validation and self.patience is not None and self.epochs_without_improvement >= self.patience:
            self.stopped_early = True
        return False


def decision_quality(logits: np.ndarray, labels: np.ndarray, names: Sequence[str]) -> tuple[DecisionQuality, ...]:
    predicted, actual = logits > 0, labels > 0.5
    qualities = []
    for column, name in enumerate(names):
        true_positives = int((predicted[:, column] & actual[:, column]).sum())
        precision = _ratio(true_positives, int(predicted[:, column].sum()))
        recall = _ratio(true_positives, int(actual[:, column].sum()))
        qualities.append(DecisionQuality(name, precision, recall, _f1(precision, recall), int(actual[:, column].sum())))
    return tuple(qualities)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
