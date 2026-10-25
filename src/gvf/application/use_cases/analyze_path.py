# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from gvf.application.ports import CodeRepository, VulnerabilityDetector
from gvf.domain import AnalysisReport, CodeAnalysisError, CodeUnit, FileAnalysis, SkippedFile


class AnalyzePath:
    def __init__(self, repository: CodeRepository, detector: VulnerabilityDetector) -> None:
        self._repository = repository
        self._detector = detector

    def execute(self, path: str) -> AnalysisReport:
        files: list[FileAnalysis] = []
        skipped: list[SkippedFile] = []
        for code_unit in self._repository.load(path):
            try:
                files.append(self._analyze(code_unit))
            except CodeAnalysisError as error:
                skipped.append(SkippedFile(error.path, error.reason))
        return AnalysisReport(files=tuple(files), skipped=tuple(skipped))

    def _analyze(self, code_unit: CodeUnit) -> FileAnalysis:
        return FileAnalysis(path=code_unit.path, assessments=self._detector.assess(code_unit))
