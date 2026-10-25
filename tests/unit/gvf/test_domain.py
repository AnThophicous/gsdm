import pytest

from gvf.domain import (
    AnalysisReport,
    Assessment,
    Confidence,
    FileAnalysis,
    SourceLocation,
    VulnerabilityCategory,
)


def test_categories_have_stable_codes_and_slugs():
    assert [(c.code, c.slug) for c in VulnerabilityCategory] == [
        ("VC-1", "sql-or-command-injection"),
        ("VC-2", "insecure-deserialization"),
        ("VC-3", "hardcoded-secret"),
    ]
    assert VulnerabilityCategory.from_code("VC-2") is VulnerabilityCategory.INSECURE_DESERIALIZATION
    with pytest.raises(ValueError):
        VulnerabilityCategory.from_code("VC-9")


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan")])
def test_confidence_rejects_values_outside_unit_interval(value):
    with pytest.raises(ValueError):
        Confidence(value)


def test_source_location_rejects_invalid_positions():
    with pytest.raises(ValueError):
        SourceLocation(line=0, column=0)
    with pytest.raises(ValueError):
        SourceLocation(line=1, column=-1)


def test_assessment_is_flagged_at_or_above_threshold():
    threshold = Confidence(0.5)
    category = VulnerabilityCategory.HARDCODED_SECRET
    assert Assessment(category, Confidence(0.5)).is_flagged(threshold)
    assert not Assessment(category, Confidence(0.4999)).is_flagged(threshold)


def test_report_knows_whether_anything_was_flagged():
    threshold = Confidence(0.5)
    low = Assessment(VulnerabilityCategory.SQL_OR_COMMAND_INJECTION, Confidence(0.1))
    high = Assessment(VulnerabilityCategory.SQL_OR_COMMAND_INJECTION, Confidence(0.9))
    clean = FileAnalysis("a.py", (low,))
    dirty = FileAnalysis("b.py", (low, high))
    assert dirty.flagged(threshold) == (high,)
    assert not AnalysisReport(files=(clean,)).has_flagged_findings(threshold)
    assert AnalysisReport(files=(clean, dirty)).has_flagged_findings(threshold)
