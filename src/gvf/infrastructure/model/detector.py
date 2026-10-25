# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from pathlib import Path
from typing import Any

from gsdm.autograd import no_grad
from gsdm.calibration import calibrated_probabilities
from gsdm.nn import GSDMConfig, GSDModel
from gsdm.persistence import ModelBundle, load_bundle
from gvf.application.ports import VulnerabilityDetector
from gvf.domain import Assessment, CodeUnit, Confidence, VulnerabilityCategory
from gvf.infrastructure.python_graph import FEATURE_SCHEMA_VERSION, FEATURE_SIZE, RELATIONS, PythonGraphBuilder

CATEGORY_ORDER = tuple(VulnerabilityCategory)


class IncompatibleModelError(ValueError):
    pass


def model_metadata() -> dict[str, Any]:
    return {
        "model": "GVF",
        "language": "python",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_size": FEATURE_SIZE,
        "relations": list(RELATIONS),
        "categories": [category.code for category in CATEGORY_ORDER],
    }


def new_model(hidden_size: int, reasoning_steps: int, seed: int) -> GSDModel:
    return GSDModel(
        GSDMConfig(
            feature_size=FEATURE_SIZE,
            relation_count=len(RELATIONS),
            decision_count=len(CATEGORY_ORDER),
            hidden_size=hidden_size,
            reasoning_steps=reasoning_steps,
            seed=seed,
        )
    )


class GsdmVulnerabilityDetector(VulnerabilityDetector):
    def __init__(self, bundle: ModelBundle, graph_builder: PythonGraphBuilder) -> None:
        _ensure_compatible(bundle.metadata)
        self._bundle = bundle
        self._graph_builder = graph_builder

    @classmethod
    def from_file(cls, path: Path) -> GsdmVulnerabilityDetector:
        return cls(load_bundle(path), PythonGraphBuilder())

    def assess(self, code_unit: CodeUnit) -> tuple[Assessment, ...]:
        graph = self._graph_builder.build(code_unit)
        with no_grad():
            output = self._bundle.model(graph.structure)
        probabilities = calibrated_probabilities(output.logits.data, self._bundle.temperatures)
        return tuple(
            Assessment(
                category=category,
                confidence=Confidence(float(probabilities[index])),
                location=graph.locations[output.evidence_node(index)],
            )
            for index, category in enumerate(CATEGORY_ORDER)
        )


def _ensure_compatible(metadata: dict[str, Any]) -> None:
    expected = model_metadata()
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise IncompatibleModelError(
                f"model was built for {key}={metadata.get(key)!r}, this GVF requires {value!r}"
            )
