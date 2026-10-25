import xml.etree.ElementTree as ElementTree

import pytest

from gvf.adapters.presenters.xml_report_presenter import XmlReportPresenter
from gvf.domain import (
    AnalysisReport,
    Assessment,
    Confidence,
    FileAnalysis,
    SkippedFile,
    SkipReason,
    SourceLocation,
    VulnerabilityCategory,
)


def present(report: AnalysisReport, threshold: float = 0.5) -> ElementTree.Element:
    document = XmlReportPresenter(Confidence(threshold), architecture_version="0.1.0").present(report)
    assert document.startswith('<?xml version="1.0" encoding="utf-8"?>')
    return ElementTree.fromstring(document.split("\n", 1)[1])


def test_report_follows_schema_v1():
    assessment = Assessment(VulnerabilityCategory.SQL_OR_COMMAND_INJECTION, Confidence(0.91324), SourceLocation(12, 4))
    report = AnalysisReport(
        files=(FileAnalysis("app/db.py", (assessment,)),),
        skipped=(SkippedFile("broken.py", SkipReason.SYNTAX_ERROR),),
    )
    root = present(report)

    assert root.tag == "gvf-report"
    assert root.attrib == {
        "schema-version": "1",
        "model": "GVF",
        "architecture": "GSDM",
        "architecture-version": "0.1.0",
        "threshold": "0.5000",
    }
    element = root.find("file/assessment")
    assert root.find("file").attrib == {"path": "app/db.py"}
    assert element.attrib == {
        "category": "VC-1",
        "name": "sql-or-command-injection",
        "confidence": "0.9132",
        "flagged": "true",
    }
    assert element.find("location").attrib == {"line": "12", "column": "4"}
    assert root.find("skipped").attrib == {"path": "broken.py", "reason": "syntax-error"}


def test_no_element_carries_free_text():
    report = AnalysisReport(files=(FileAnalysis("a.py", (Assessment(VulnerabilityCategory.HARDCODED_SECRET, Confidence(0.2)),)),))
    for element in present(report).iter():
        assert not (element.text or "").strip()


def test_printed_confidence_never_contradicts_flagged():
    just_below = Assessment(VulnerabilityCategory.HARDCODED_SECRET, Confidence(0.49996))
    element = present(AnalysisReport(files=(FileAnalysis("a.py", (just_below,)),))).find("file/assessment")
    assert element.attrib["confidence"] == "0.4999"
    assert element.attrib["flagged"] == "false"


def test_paths_are_escaped():
    report = AnalysisReport(files=(FileAnalysis('we"ird<&>.py', ()),))
    assert present(report).find("file").attrib["path"] == 'we"ird<&>.py'


def test_threshold_must_have_at_most_four_decimals():
    with pytest.raises(ValueError):
        XmlReportPresenter(Confidence(0.12345), architecture_version="0.1.0")
