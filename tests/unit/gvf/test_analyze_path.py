import pytest

from gvf.application.ports import CodeRepository, VulnerabilityDetector
from gvf.application.use_cases import AnalyzePath
from gvf.domain import (
    Assessment,
    CodeAnalysisError,
    CodeSourceNotFoundError,
    CodeUnit,
    Confidence,
    SkippedFile,
    SkipReason,
    VulnerabilityCategory,
)


class InMemoryRepository(CodeRepository):
    def __init__(self, units: dict[str, list[CodeUnit]]) -> None:
        self.units = units

    def load(self, path: str) -> list[CodeUnit]:
        if path not in self.units:
            raise CodeSourceNotFoundError(path)
        return self.units[path]


class ScriptedDetector(VulnerabilityDetector):
    def assess(self, code_unit: CodeUnit) -> tuple[Assessment, ...]:
        if code_unit.source == "broken":
            raise CodeAnalysisError(code_unit.path, SkipReason.SYNTAX_ERROR)
        confidence = Confidence(0.9 if "pickle" in code_unit.source else 0.1)
        return (Assessment(VulnerabilityCategory.INSECURE_DESERIALIZATION, confidence),)


def test_analyzes_every_unit_and_records_unanalyzable_ones():
    units = [CodeUnit("a.py", "pickle"), CodeUnit("b.py", "broken"), CodeUnit("c.py", "json")]
    report = AnalyzePath(InMemoryRepository({"project": units}), ScriptedDetector()).execute("project")

    assert [file.path for file in report.files] == ["a.py", "c.py"]
    assert report.files[0].assessments[0].confidence == Confidence(0.9)
    assert report.skipped == (SkippedFile("b.py", SkipReason.SYNTAX_ERROR),)


def test_missing_source_propagates_to_the_caller():
    with pytest.raises(CodeSourceNotFoundError):
        AnalyzePath(InMemoryRepository({}), ScriptedDetector()).execute("nowhere")
