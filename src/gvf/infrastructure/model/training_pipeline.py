# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Builds a calibrated GVF model from a JSONL manifest of labeled Python files (ADR-0005, ADR-0008)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gsdm.calibration import calibrated_probabilities, expected_calibration_error, fit_temperatures
from gsdm.optim import Adam
from gsdm.persistence import ModelBundle, save_bundle
from gsdm.training import Trainer, TrainingExample, predict_logits
from gvf.domain import CodeAnalysisError, CodeUnit, VulnerabilityCategory
from gvf.infrastructure.model.detector import CATEGORY_ORDER, model_metadata, new_model
from gvf.infrastructure.python_graph import PythonGraphBuilder


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int = 30
    hidden_size: int = 64
    reasoning_steps: int = 4
    learning_rate: float = 3e-3
    batch_size: int = 8
    validation_fraction: float = 0.2
    seed: int = 0


@dataclass(frozen=True)
class TrainingSummary:
    training_examples: int
    validation_examples: int
    final_training_loss: float
    validation_accuracy: float | None
    calibration_error_before: float | None
    calibration_error_after: float | None
    temperatures: tuple[float, ...]


@dataclass(frozen=True, eq=False)
class LabeledSource:
    code_unit: CodeUnit
    labels: np.ndarray


def load_manifest(manifest_path: Path) -> list[LabeledSource]:
    entries = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            entries.append(_parse_manifest_line(manifest_path.parent, line, line_number))
    return entries


def train_detector(manifest_path: Path, output_path: Path, settings: TrainingSettings) -> TrainingSummary:
    examples = _to_examples(load_manifest(manifest_path))
    if len(examples) < 2:
        raise DatasetError("training needs at least 2 labeled files")
    training, validation = _split(examples, settings.validation_fraction, settings.seed)

    model = new_model(settings.hidden_size, settings.reasoning_steps, settings.seed)
    trainer = Trainer(
        model, Adam(model.parameters(), learning_rate=settings.learning_rate), settings.batch_size, seed=settings.seed
    )
    losses = trainer.fit(training, settings.epochs)

    summary_fields, temperatures = _calibrate(model, validation)
    summary = TrainingSummary(
        training_examples=len(training),
        validation_examples=len(validation),
        final_training_loss=losses[-1] if losses else float("nan"),
        temperatures=tuple(float(t) for t in temperatures),
        **summary_fields,
    )
    metadata = model_metadata() | {"training": asdict(summary) | {"settings": asdict(settings)}}
    save_bundle(output_path, ModelBundle(model=model, temperatures=temperatures, metadata=metadata))
    return summary


def _calibrate(model, validation: list[TrainingExample]) -> tuple[dict[str, float | None], np.ndarray]:
    if not validation:
        empty = {"validation_accuracy": None, "calibration_error_before": None, "calibration_error_after": None}
        return empty, np.ones(len(CATEGORY_ORDER))
    logits = predict_logits(model, [example.graph for example in validation])
    labels = np.array([example.labels for example in validation])
    temperatures = fit_temperatures(logits, labels)
    return {
        "validation_accuracy": float(((logits > 0) == (labels > 0.5)).mean()),
        "calibration_error_before": expected_calibration_error(calibrated_probabilities(logits, 1.0), labels),
        "calibration_error_after": expected_calibration_error(calibrated_probabilities(logits, temperatures), labels),
    }, temperatures


def _parse_manifest_line(base: Path, line: str, line_number: int) -> LabeledSource:
    try:
        entry = json.loads(line)
        path = base / entry["path"]
        codes = entry["labels"]
        categories = {VulnerabilityCategory.from_code(code) for code in codes}
        source = path.read_bytes().decode("utf-8-sig", errors="replace")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as error:
        raise DatasetError(f"manifest line {line_number}: {error}") from error
    labels = np.array([1.0 if category in categories else 0.0 for category in CATEGORY_ORDER])
    return LabeledSource(CodeUnit(path.as_posix(), source), labels)


def _to_examples(sources: list[LabeledSource]) -> list[TrainingExample]:
    builder = PythonGraphBuilder()
    examples = []
    for labeled in sources:
        try:
            graph = builder.build(labeled.code_unit)
        except CodeAnalysisError as error:
            raise DatasetError(f"dataset file cannot be parsed: {error}") from error
        examples.append(TrainingExample(graph.structure, labeled.labels))
    return examples


def _split(
    examples: list[TrainingExample], validation_fraction: float, seed: int
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    if not 0.0 <= validation_fraction < 1.0:
        raise DatasetError("validation_fraction must be within [0, 1)")
    order = np.random.default_rng(seed).permutation(len(examples))
    validation_count = int(round(validation_fraction * len(examples)))
    if validation_fraction > 0:
        validation_count = min(max(1, validation_count), len(examples) - 1)
    validation = [examples[index] for index in order[:validation_count]]
    training = [examples[index] for index in order[validation_count:]]
    return training, validation
