# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from abc import ABC, abstractmethod

from gvf.domain import AnalysisReport, Assessment, CodeUnit


class CodeRepository(ABC):
    @abstractmethod
    def load(self, path: str) -> list[CodeUnit]:
        """Return every analyzable unit under path. Raises CodeSourceNotFoundError."""


class VulnerabilityDetector(ABC):
    @abstractmethod
    def assess(self, code_unit: CodeUnit) -> tuple[Assessment, ...]:
        """Return one assessment per category. Raises CodeAnalysisError when the unit cannot be analyzed."""


class ReportPresenter(ABC):
    @abstractmethod
    def present(self, report: AnalysisReport) -> str: ...
