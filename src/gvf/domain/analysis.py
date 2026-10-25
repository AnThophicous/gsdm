# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from gvf.domain.vulnerability import Assessment, Confidence


@dataclass(frozen=True)
class CodeUnit:
    path: str
    source: str


class SkipReason(Enum):
    SYNTAX_ERROR = "syntax-error"


class CodeAnalysisError(Exception):
    def __init__(self, path: str, reason: SkipReason) -> None:
        super().__init__(f"{path}: {reason.value}")
        self.path = path
        self.reason = reason


class CodeSourceNotFoundError(Exception):
    def __init__(self, path: str) -> None:
        super().__init__(f"code source not found: {path}")
        self.path = path


@dataclass(frozen=True)
class FileAnalysis:
    path: str
    assessments: tuple[Assessment, ...]

    def flagged(self, threshold: Confidence) -> tuple[Assessment, ...]:
        return tuple(assessment for assessment in self.assessments if assessment.is_flagged(threshold))


@dataclass(frozen=True)
class SkippedFile:
    path: str
    reason: SkipReason


@dataclass(frozen=True)
class AnalysisReport:
    files: tuple[FileAnalysis, ...]
    skipped: tuple[SkippedFile, ...] = ()

    def has_flagged_findings(self, threshold: Confidence) -> bool:
        return any(file.flagged(threshold) for file in self.files)
