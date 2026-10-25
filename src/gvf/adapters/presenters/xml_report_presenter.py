# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""XML report, schema version 1 (ADR-0004)."""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from decimal import ROUND_FLOOR, Decimal

from gvf.application.ports import ReportPresenter
from gvf.domain import AnalysisReport, Assessment, Confidence, FileAnalysis

SCHEMA_VERSION = "1"
_PRECISION = Decimal("0.0001")


class XmlReportPresenter(ReportPresenter):
    def __init__(self, threshold: Confidence, architecture_version: str) -> None:
        if Decimal(str(threshold.value)) != _floor(threshold.value):
            raise ValueError("threshold must have at most 4 decimal places")
        self._threshold = threshold
        self._architecture_version = architecture_version

    def present(self, report: AnalysisReport) -> str:
        root = ElementTree.Element(
            "gvf-report",
            {
                "schema-version": SCHEMA_VERSION,
                "model": "GVF",
                "architecture": "GSDM",
                "architecture-version": self._architecture_version,
                "threshold": _format(self._threshold.value),
            },
        )
        for file in report.files:
            self._append_file(root, file)
        for skipped in report.skipped:
            ElementTree.SubElement(root, "skipped", {"path": skipped.path, "reason": skipped.reason.value})
        ElementTree.indent(root)
        return '<?xml version="1.0" encoding="utf-8"?>\n' + ElementTree.tostring(root, encoding="unicode") + "\n"

    def _append_file(self, root: ElementTree.Element, file: FileAnalysis) -> None:
        file_element = ElementTree.SubElement(root, "file", {"path": file.path})
        for assessment in file.assessments:
            self._append_assessment(file_element, assessment)

    def _append_assessment(self, parent: ElementTree.Element, assessment: Assessment) -> None:
        element = ElementTree.SubElement(
            parent,
            "assessment",
            {
                "category": assessment.category.code,
                "name": assessment.category.slug,
                "confidence": _format(assessment.confidence.value),
                "flagged": "true" if assessment.is_flagged(self._threshold) else "false",
            },
        )
        if assessment.location is not None:
            ElementTree.SubElement(
                element, "location", {"line": str(assessment.location.line), "column": str(assessment.location.column)}
            )


def _floor(value: float) -> Decimal:
    """Flooring keeps the printed confidence consistent with `flagged` for 4-decimal thresholds."""
    return Decimal(str(value)).quantize(_PRECISION, rounding=ROUND_FLOOR)


def _format(value: float) -> str:
    return str(_floor(value))
