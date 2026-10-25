# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from dataclasses import dataclass

from gvf.application.ports import ReportPresenter
from gvf.application.use_cases import AnalyzePath
from gvf.domain import Confidence


@dataclass(frozen=True)
class AnalysisResponse:
    document: str
    has_flagged_findings: bool


class AnalyzeController:
    def __init__(self, use_case: AnalyzePath, presenter: ReportPresenter, threshold: Confidence) -> None:
        self._use_case = use_case
        self._presenter = presenter
        self._threshold = threshold

    def handle(self, path: str) -> AnalysisResponse:
        report = self._use_case.execute(path)
        return AnalysisResponse(
            document=self._presenter.present(report),
            has_flagged_findings=report.has_flagged_findings(self._threshold),
        )
