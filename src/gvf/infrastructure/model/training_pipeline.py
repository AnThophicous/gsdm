# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Builds a calibrated GVF model from a JSONL manifest of labeled Python files (ADR-0005, ADR-0008, ADR-0009)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TextIO

import numpy as np

import gsdm
from gsdm.calibration import calibrated_probabilities, expected_calibration_error, fit_temperatures
from gsdm.capacity import estimate, format_bytes
from gsdm.checkpoint import CheckpointWriter, load_checkpoint, read_checkpoint_metadata
from gsdm.diagnostics import diagnose
from gsdm.nn import GSDModel
from gsdm.optim import Adam
from gsdm.persistence import ModelBundle, save_bundle
from gsdm.reporting import ConsoleProgress, DashboardWriter, MetricsLog
from gsdm.session import TrainingObserver, TrainingSession
from gsdm.training import Trainer, TrainingExample, predict_logits
from gvf.domain import CodeAnalysisError, CodeUnit, VulnerabilityCategory
from gvf.infrastructure.model.detector import CATEGORY_ORDER, model_metadata, new_model
from gvf.infrastructure.python_graph import PythonGraphBuilder

CHECKPOINT_FILE = "checkpoint.npz"
METRICS_FILE = "metrics.jsonl"
DASHBOARD_FILE = "dashboard.html"


class DatasetError(ValueError):
    pass


class ResumeError(ValueError):
    pass


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int = 30
    hidden_size: int = 64
    reasoning_steps: int = 4
    learning_rate: float = 3e-3
    batch_size: int = 32
    validation_fraction: float = 0.2
    seed: int = 0
    patience: int = 5
    checkpoint_every: int = 1


@dataclass(frozen=True)
class TrainingSummary:
    training_examples: int
    validation_examples: int
    grouped_validation: bool
    parameters: int
    epochs_run: int
    best_epoch: int | None
    stopped_early: bool
    diagnosis: str
    diagnosis_evidence: str
    final_training_loss: float
    best_validation_loss: float | None
    validation_accuracy: float | None
    calibration_error_before: float | None
    calibration_error_after: float | None
    temperatures: tuple[float, ...]
    run_directory: str


@dataclass(frozen=True, eq=False)
class LabeledSource:
    code_unit: CodeUnit
    labels: np.ndarray
    group: str | None


def run_directory_for(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + ".run")


def load_manifest(manifest_path: Path) -> list[LabeledSource]:
    entries = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            entries.append(_parse_manifest_line(manifest_path.parent, line, line_number))
    return entries


def train_detector(
    manifest_path: Path,
    output_path: Path,
    settings: TrainingSettings,
    progress: TextIO | None = None,
    resume: bool = False,
) -> TrainingSummary:
    run_directory = run_directory_for(output_path)
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if resume:
        settings = _settings_to_resume(run_directory, manifest_digest, settings.epochs)
    sources = load_manifest(manifest_path)
    examples = _to_examples(sources)
    if len(examples) < 2:
        raise DatasetError("training needs at least 2 labeled files")
    groups = [source.group or f"file:{source.code_unit.path}" for source in sources]
    grouped = any(source.group is not None for source in sources)
    training, validation = _grouped_split(examples, groups, settings.validation_fraction, settings.seed)

    model = new_model(settings.hidden_size, settings.reasoning_steps, settings.seed)
    run_directory.mkdir(parents=True, exist_ok=True)
    if not resume:
        (run_directory / METRICS_FILE).unlink(missing_ok=True)
    _announce(progress, model, training, validation, grouped, resume)

    session = TrainingSession(
        Trainer(model, Adam(model.parameters(), learning_rate=settings.learning_rate), settings.batch_size, seed=settings.seed),
        training,
        validation,
        decision_names=[category.code for category in CATEGORY_ORDER],
        observers=_observers(run_directory, model, training, validation, grouped, settings, manifest_digest, progress),
        patience=settings.patience or None,
    )
    if resume:
        load_checkpoint(run_directory / CHECKPOINT_FILE, session)
    session.run(settings.epochs)
    session.use_best_weights()

    calibration, temperatures = _calibrate(model, validation)
    diagnosis = diagnose(session.history)
    summary = TrainingSummary(
        training_examples=len(training),
        validation_examples=len(validation),
        grouped_validation=grouped,
        parameters=model.parameter_count(),
        epochs_run=session.epoch,
        best_epoch=session.best_epoch,
        stopped_early=session.stopped_early,
        diagnosis=diagnosis.status.value,
        diagnosis_evidence=diagnosis.evidence,
        final_training_loss=session.history[-1].training_loss if session.history else float("nan"),
        best_validation_loss=None if not validation else session.best_validation_loss,
        temperatures=tuple(float(t) for t in temperatures),
        run_directory=run_directory.as_posix(),
        **calibration,
    )
    metadata = model_metadata() | {"training": asdict(summary) | {"settings": asdict(settings)}}
    save_bundle(output_path, ModelBundle(model=model, temperatures=temperatures, metadata=metadata))
    return summary


def _settings_to_resume(run_directory: Path, manifest_digest: str, epochs: int) -> TrainingSettings:
    checkpoint = run_directory / CHECKPOINT_FILE
    if not checkpoint.is_file():
        raise ResumeError(f"no checkpoint to resume at {checkpoint.as_posix()}")
    metadata = read_checkpoint_metadata(checkpoint)
    if metadata.get("manifest_sha256") != manifest_digest:
        raise ResumeError("the manifest changed since this run started; resuming would mix two datasets")
    return replace(TrainingSettings(**metadata["settings"]), epochs=epochs)


def _observers(
    run_directory: Path,
    model: GSDModel,
    training: list[TrainingExample],
    validation: list[TrainingExample],
    grouped: bool,
    settings: TrainingSettings,
    manifest_digest: str,
    progress: TextIO | None,
) -> list[TrainingObserver]:
    capacity = estimate(model.config)
    facts = {
        "model": f"GVF on GSDM {gsdm.__version__}",
        "parameters": f"{capacity.parameters:,}",
        "training state": format_bytes(capacity.training_state_bytes),
        "training graphs": f"{len(training):,}",
        "validation graphs": f"{len(validation):,} ({'grouped' if grouped else 'NOT grouped'})",
        "reasoning steps": str(model.config.reasoning_steps),
    }
    observers: list[TrainingObserver] = [
        MetricsLog(run_directory / METRICS_FILE),
        DashboardWriter(run_directory / DASHBOARD_FILE, "GVF training", facts),
        CheckpointWriter(
            run_directory / CHECKPOINT_FILE,
            settings.checkpoint_every,
            {"manifest_sha256": manifest_digest, "settings": asdict(settings)},
        ),
    ]
    if progress is not None:
        observers.insert(0, ConsoleProgress(progress))
    return observers


def _announce(
    progress: TextIO | None,
    model: GSDModel,
    training: list[TrainingExample],
    validation: list[TrainingExample],
    grouped: bool,
    resume: bool,
) -> None:
    if progress is None:
        return
    capacity = estimate(model.config)
    lines = [
        f"GSDM {gsdm.__version__} | {capacity.parameters:,} parameters | weights {format_bytes(capacity.weights_bytes)}"
        f" | training state {format_bytes(capacity.training_state_bytes)}",
        f"{len(training)} training graphs | {len(validation)} validation graphs",
    ]
    if resume:
        lines.append("resuming from checkpoint with the settings stored in it")
    if validation and not grouped:
        lines.append(
            "warning: validation is not grouped; near-duplicate files can sit in both sets and hide memorization. "
            'Add a "group" field to the manifest.'
        )
    missing = categories_without_validation_positives(validation)
    if missing:
        lines.append(
            f"warning: no positive validation example for {', '.join(missing)}; "
            "quality for these categories cannot be measured. Add more groups or change --seed."
        )
    progress.write("\n".join(lines) + "\n")
    progress.flush()


def categories_without_validation_positives(validation: list[TrainingExample]) -> list[str]:
    if not validation:
        return []
    positives = np.array([example.labels for example in validation]).sum(axis=0)
    return [category.code for category, count in zip(CATEGORY_ORDER, positives) if count == 0]


def _calibrate(model: GSDModel, validation: list[TrainingExample]) -> tuple[dict[str, float | None], np.ndarray]:
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
        categories = {VulnerabilityCategory.from_code(code) for code in entry["labels"]}
        group = entry.get("group")
        if group is not None and not isinstance(group, str):
            raise ValueError("group must be a string")
        source = path.read_bytes().decode("utf-8-sig", errors="replace")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as error:
        raise DatasetError(f"manifest line {line_number}: {error}") from error
    labels = np.array([1.0 if category in categories else 0.0 for category in CATEGORY_ORDER])
    return LabeledSource(CodeUnit(path.as_posix(), source), labels, group)


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


def _grouped_split(
    examples: list[TrainingExample], groups: list[str], validation_fraction: float, seed: int
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Whole groups go to validation until it holds about validation_fraction of the examples (ADR-0009, item 6)."""
    if not 0.0 <= validation_fraction < 1.0:
        raise DatasetError("validation_fraction must be within [0, 1)")
    if validation_fraction == 0:
        return list(examples), []
    sizes: dict[str, int] = {}
    for group in groups:
        sizes[group] = sizes.get(group, 0) + 1
    if len(sizes) < 2:
        raise DatasetError("a validation split needs at least 2 groups")
    target = max(1, round(validation_fraction * len(examples)))
    ordered_groups = sorted(sizes)
    chosen: set[str] = set()
    count = 0
    for index in np.random.default_rng(seed).permutation(len(ordered_groups)):
        group = ordered_groups[index]
        if count >= target:
            break
        if count + sizes[group] < len(examples):
            chosen.add(group)
            count += sizes[group]
    training = [example for example, group in zip(examples, groups) if group not in chosen]
    validation = [example for example, group in zip(examples, groups) if group in chosen]
    return training, validation
