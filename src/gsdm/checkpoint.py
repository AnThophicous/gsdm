# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Resumable training checkpoints: a resumed run is bit-identical to an uninterrupted one (ADR-0009, item 10)."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from gsdm.session import EpochReport, TrainingObserver, TrainingSession

_FORMAT_VERSION = 1
_HEADER = "header"


def save_checkpoint(path: Path, session: TrainingSession, metadata: dict[str, Any] | None = None) -> None:
    if not session.is_resumable:
        raise ValueError("the model holds the best weights, not the training weights; save before use_best_weights()")
    trainer = session.trainer
    optimizer_state = trainer.optimizer.state_dict()
    arrays = {f"model/{name}": value for name, value in trainer.model.state_dict().items()}
    if session.best_state is not None:
        arrays |= {f"best/{name}": value for name, value in session.best_state.items()}
    arrays |= {f"adam/first/{i}": m for i, m in enumerate(optimizer_state["first_moments"])}
    arrays |= {f"adam/second/{i}": m for i, m in enumerate(optimizer_state["second_moments"])}
    header = {
        "format_version": _FORMAT_VERSION,
        "config": trainer.model.config.to_dict(),
        "epoch": session.epoch,
        "history": [report.to_dict() for report in session.history],
        "best_validation_loss": None if math.isinf(session.best_validation_loss) else session.best_validation_loss,
        "epochs_without_improvement": session.epochs_without_improvement,
        "stopped_early": session.stopped_early,
        "adam_step_count": optimizer_state["step_count"],
        "rng_state": trainer.rng_state(),
        "metadata": metadata or {},
    }
    arrays[_HEADER] = np.frombuffer(json.dumps(header).encode("utf-8"), dtype=np.uint8)
    _write_atomically(path, arrays)


def read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Reads only the metadata, e.g. to rebuild the session a checkpoint belongs to before loading it."""
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(archive[_HEADER].tobytes().decode("utf-8"))["metadata"]


def load_checkpoint(path: Path, session: TrainingSession) -> dict[str, Any]:
    """Restores model, best weights, optimizer, random state and history into session. Returns the metadata."""
    with np.load(path, allow_pickle=False) as archive:
        header = json.loads(archive[_HEADER].tobytes().decode("utf-8"))
        if header.get("format_version") != _FORMAT_VERSION:
            raise ValueError(f"unsupported checkpoint format version: {header.get('format_version')}")
        if header["config"] != session.trainer.model.config.to_dict():
            raise ValueError("checkpoint was saved for a model with a different configuration")
        model_state = _section(archive, "model/")
        best_state = _section(archive, "best/") or None
        moment_count = len(session.trainer.optimizer.parameters)
        first = [archive[f"adam/first/{i}"] for i in range(moment_count)]
        second = [archive[f"adam/second/{i}"] for i in range(moment_count)]

    session.trainer.model.load_state_dict(model_state)
    session.trainer.optimizer.load_state_dict(
        {"step_count": header["adam_step_count"], "first_moments": first, "second_moments": second}
    )
    session.trainer.restore_rng_state(header["rng_state"])
    session.epoch = header["epoch"]
    session.history = [EpochReport.from_dict(report) for report in header["history"]]
    session.best_state = best_state
    best = header["best_validation_loss"]
    session.best_validation_loss = math.inf if best is None else best
    session.epochs_without_improvement = header["epochs_without_improvement"]
    session.stopped_early = header["stopped_early"]
    return header["metadata"]


class CheckpointWriter(TrainingObserver):
    def __init__(self, path: Path, every: int = 1, metadata: dict[str, Any] | None = None) -> None:
        if every < 1:
            raise ValueError("checkpoint interval must be at least 1 epoch")
        self.path = path
        self.every = every
        self.metadata = metadata or {}

    def on_epoch_end(self, report: EpochReport, session: TrainingSession) -> None:
        if report.epoch % self.every == 0:
            save_checkpoint(self.path, session, self.metadata)

    def on_training_end(self, session: TrainingSession) -> None:
        save_checkpoint(self.path, session, self.metadata)


def _section(archive: Any, prefix: str) -> dict[str, np.ndarray]:
    return {key.removeprefix(prefix): archive[key] for key in archive.files if key.startswith(prefix)}


def _write_atomically(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """An interrupted save never corrupts the previous checkpoint: write a temporary file, then rename."""
    temporary = path.with_name(path.name + ".partial")
    with open(temporary, "wb") as file:
        np.savez(file, **arrays)
    os.replace(temporary, path)
