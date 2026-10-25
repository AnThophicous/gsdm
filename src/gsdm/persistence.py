# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Model files are plain .npz archives read with allow_pickle=False (ADR-0008, item 12)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import gsdm
from gsdm.nn import GSDMConfig, GSDModel

_PARAMETER_PREFIX = "parameter/"
_TEMPERATURES_KEY = "calibration/temperatures"
_HEADER_KEY = "header"
_FORMAT_VERSION = 1


@dataclass
class ModelBundle:
    model: GSDModel
    temperatures: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def save_bundle(path: Path, bundle: ModelBundle) -> None:
    if bundle.temperatures.shape != (bundle.model.config.decision_count,):
        raise ValueError("there must be exactly one temperature per decision")
    header = {
        "format_version": _FORMAT_VERSION,
        "gsdm_version": gsdm.__version__,
        "config": bundle.model.config.to_dict(),
        "metadata": bundle.metadata,
    }
    arrays = {f"{_PARAMETER_PREFIX}{name}": value for name, value in bundle.model.state_dict().items()}
    arrays[_TEMPERATURES_KEY] = bundle.temperatures
    arrays[_HEADER_KEY] = np.frombuffer(json.dumps(header).encode("utf-8"), dtype=np.uint8)
    with open(path, "wb") as file:
        np.savez(file, **arrays)


def load_bundle(path: Path) -> ModelBundle:
    with np.load(path, allow_pickle=False) as archive:
        header = json.loads(archive[_HEADER_KEY].tobytes().decode("utf-8"))
        if header.get("format_version") != _FORMAT_VERSION:
            raise ValueError(f"unsupported model format version: {header.get('format_version')}")
        model = GSDModel(GSDMConfig.from_dict(header["config"]))
        model.load_state_dict(
            {key.removeprefix(_PARAMETER_PREFIX): archive[key] for key in archive.files if key.startswith(_PARAMETER_PREFIX)}
        )
        temperatures = np.array(archive[_TEMPERATURES_KEY], dtype=np.float64)
    return ModelBundle(model=model, temperatures=temperatures, metadata=header["metadata"])
