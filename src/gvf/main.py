# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Composition root: the only place where concrete implementations are wired together."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import gsdm
from gsdm.capacity import estimate, format_bytes, hidden_size_for
from gsdm.nn import GSDMConfig
from gvf.adapters.controllers.analyze_controller import AnalyzeController
from gvf.adapters.presenters.xml_report_presenter import XmlReportPresenter
from gvf.application.use_cases import AnalyzePath
from gvf.domain import CodeSourceNotFoundError, Confidence
from gvf.infrastructure.filesystem.local_code_repository import LocalCodeRepository
from gvf.infrastructure.model.detector import CATEGORY_ORDER, GsdmVulnerabilityDetector
from gvf.infrastructure.model.training_pipeline import TrainingSettings, train_detector
from gvf.infrastructure.python_graph import FEATURE_SIZE, RELATIONS

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
        patience=arguments.patience,
        checkpoint_every=arguments.checkpoint_every,
    )
    summary = train_detector(arguments.manifest, arguments.output, settings, sys.stdout, arguments.resume)
    print()
    for name, value in asdict(summary).items():
        print(f"{name}: {value}")
    return EXIT_CLEAN


def _capacity(arguments: argparse.Namespace) -> int:
    config = GSDMConfig(
        feature_size=FEATURE_SIZE,
        relation_count=len(RELATIONS),
        decision_count=len(CATEGORY_ORDER),
        hidden_size=arguments.hidden_size,
        reasoning_steps=arguments.reasoning_steps,
    )
    if arguments.target_parameters:
        target = int(arguments.target_parameters)
        hidden = hidden_size_for(target, FEATURE_SIZE, len(RELATIONS), len(CATEGORY_ORDER))
        config = replace(config, hidden_size=hidden)
    capacity = estimate(config)
    activations = capacity.activation_bytes_per_node * arguments.nodes_per_batch
    rows = {
        "hidden size": f"{config.hidden_size:,}",
        "reasoning steps": str(config.reasoning_steps),
        "parameters": f"{capacity.parameters:,}",
        "weights (inference, float32)": format_bytes(capacity.weights_bytes),
        "training state (weights, gradients, Adam)": format_bytes(capacity.training_state_bytes),
        f"activations for {arguments.nodes_per_batch:,} nodes per batch": format_bytes(activations),
        "estimated training memory": format_bytes(capacity.training_bytes(arguments.nodes_per_batch)),
    }
    width = max(len(name) for name in rows)
    for name, value in rows.items():
        print(f"{name:<{width}}  {value}")
    return EXIT_CLEAN


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gvf", description=f"GVF - Graph-based Vulnerability Finder (GSDM {gsdm.__version__})")
    commands = parser.add_subparsers(required=True)

    analyze = commands.add_parser("analyze", help="analyze a Python file or directory and print an XML report")
    analyze.add_argument("path")
    analyze.add_argument("--model", type=Path, required=True)
    analyze.add_argument("--threshold", type=float, default=0.5)
    analyze.set_defaults(handler=_analyze)

    defaults = TrainingSettings()
    train = commands.add_parser("train", help="train and calibrate a model from a JSONL manifest")
    train.add_argument("manifest", type=Path)
    train.add_argument("--output", type=Path, required=True, help="model file; logs go to <output>.run/")
    train.add_argument("--epochs", type=int, default=defaults.epochs, help="total epochs, also when resuming")
    train.add_argument("--hidden-size", type=int, default=defaults.hidden_size)
    train.add_argument("--reasoning-steps", type=int, default=defaults.reasoning_steps)
    train.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    train.add_argument("--batch-size", type=int, default=defaults.batch_size)
    train.add_argument("--validation-fraction", type=float, default=defaults.validation_fraction)
    train.add_argument("--seed", type=int, default=defaults.seed)
    train.add_argument("--patience", type=int, default=defaults.patience, help="early stopping patience; 0 disables")
    train.add_argument("--checkpoint-every", type=int, default=defaults.checkpoint_every)
    train.add_argument("--resume", action="store_true", help="continue from <output>.run/checkpoint.npz")
    train.set_defaults(handler=_train)

    capacity = commands.add_parser("capacity", help="estimate parameters and memory before training")
    capacity.add_argument("--hidden-size", type=int, default=defaults.hidden_size)
    capacity.add_argument("--reasoning-steps", type=int, default=defaults.reasoning_steps)
    capacity.add_argument("--nodes-per-batch", type=int, default=20_000)
    capacity.add_argument("--target-parameters", type=float, help="solve the hidden size for this many parameters")
    capacity.set_defaults(handler=_capacity)
    return parser
