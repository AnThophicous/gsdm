# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Composition root: the only place where concrete implementations are wired together."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import gsdm
from gvf.adapters.controllers.analyze_controller import AnalyzeController
from gvf.adapters.presenters.xml_report_presenter import XmlReportPresenter
from gvf.application.use_cases import AnalyzePath
from gvf.domain import CodeSourceNotFoundError, Confidence
from gvf.infrastructure.filesystem.local_code_repository import LocalCodeRepository
from gvf.infrastructure.model.detector import GsdmVulnerabilityDetector
from gvf.infrastructure.model.training_pipeline import TrainingSettings, train_detector

EXIT_CLEAN = 0
EXIT_FINDINGS = 1
EXIT_ERROR = 2


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    try:
        return parsed.handler(parsed)
    except (CodeSourceNotFoundError, ValueError, OSError) as error:
        print(f"gvf: error: {error}", file=sys.stderr)
        return EXIT_ERROR


def _analyze(arguments: argparse.Namespace) -> int:
    threshold = Confidence(arguments.threshold)
    controller = AnalyzeController(
        AnalyzePath(LocalCodeRepository(), GsdmVulnerabilityDetector.from_file(arguments.model)),
        XmlReportPresenter(threshold, architecture_version=gsdm.__version__),
        threshold,
    )
    response = controller.handle(arguments.path)
    sys.stdout.write(response.document)
    return EXIT_FINDINGS if response.has_flagged_findings else EXIT_CLEAN


def _train(arguments: argparse.Namespace) -> int:
    settings = TrainingSettings(
        epochs=arguments.epochs,
        hidden_size=arguments.hidden_size,
        reasoning_steps=arguments.reasoning_steps,
        learning_rate=arguments.learning_rate,
        batch_size=arguments.batch_size,
        validation_fraction=arguments.validation_fraction,
        seed=arguments.seed,
    )
    summary = train_detector(arguments.manifest, arguments.output, settings)
    for name, value in asdict(summary).items():
        print(f"{name}: {value}")
    return EXIT_CLEAN


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gvf", description="GVF - Graph-based Vulnerability Finder (GSDM)")
    commands = parser.add_subparsers(required=True)

    analyze = commands.add_parser("analyze", help="analyze a Python file or directory and print an XML report")
    analyze.add_argument("path")
    analyze.add_argument("--model", type=Path, required=True)
    analyze.add_argument("--threshold", type=float, default=0.5)
    analyze.set_defaults(handler=_analyze)

    defaults = TrainingSettings()
    train = commands.add_parser("train", help="train and calibrate a model from a JSONL manifest")
    train.add_argument("manifest", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=defaults.epochs)
    train.add_argument("--hidden-size", type=int, default=defaults.hidden_size)
    train.add_argument("--reasoning-steps", type=int, default=defaults.reasoning_steps)
    train.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    train.add_argument("--batch-size", type=int, default=defaults.batch_size)
    train.add_argument("--validation-fraction", type=float, default=defaults.validation_fraction)
    train.add_argument("--seed", type=int, default=defaults.seed)
    train.set_defaults(handler=_train)
    return parser
