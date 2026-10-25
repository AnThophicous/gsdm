import io
import json
import re

import pytest

from gsdm.capacity import estimate, format_bytes, hidden_size_for, parameter_count
from gsdm.diagnostics import LearningStatus, diagnose
from gsdm.nn import GSDMConfig, GSDModel
from gsdm.reporting import ConsoleProgress, DashboardWriter, MetricsLog, format_epoch, render_dashboard
from gsdm.session import DecisionQuality, EpochReport
from tests.unit.gsdm.test_session_and_checkpoint import dataset, new_session


def report(epoch: int, training: float, validation: float | None, f1: float | None = 0.5) -> EpochReport:
    decisions = () if validation is None else (DecisionQuality("VC-1", 0.5, 0.5, f1, 3),)
    return EpochReport(epoch, training, validation, 0.1 if validation else None, decisions, 1.0, 100.0, False)


def history(pairs):
    return [report(index + 1, training, validation) for index, (training, validation) in enumerate(pairs)]


def test_learning_when_validation_keeps_improving():
    diagnosis = diagnose(history([(0.9, 0.95), (0.6, 0.7), (0.4, 0.5), (0.3, 0.42)]))
    assert diagnosis.status is LearningStatus.LEARNING
    assert diagnosis.best_epoch == 4


def test_memorizing_when_validation_rises_while_training_falls():
    diagnosis = diagnose(history([(0.9, 0.9), (0.5, 0.6), (0.3, 0.65), (0.15, 0.72), (0.05, 0.85)]))
    assert diagnosis.status is LearningStatus.MEMORIZING
    assert diagnosis.best_epoch == 2
    assert "since epoch 2" in diagnosis.evidence


def test_memorizing_when_training_is_perfect_but_validation_is_not():
    diagnosis = diagnose(history([(0.5, 0.6), (0.1, 0.4), (0.01, 0.39)]))
    assert diagnosis.status is LearningStatus.MEMORIZING


def test_stalled_when_nothing_improves():
    diagnosis = diagnose(history([(0.6931, 0.6932)] * 6))
    assert diagnosis.status is LearningStatus.STALLED


def test_no_validation_and_too_short():
    assert diagnose(history([(0.5, None), (0.4, None)])).status is LearningStatus.NO_VALIDATION
    assert diagnose(history([(0.5, 0.6), (0.4, 0.5)])).status is LearningStatus.TOO_SHORT


def test_epoch_line_marks_missing_values_and_best_epoch():
    line = format_epoch(EpochReport(3, 0.5, None, None, (), 1.0, 12.5, True))
    assert line.split()[:5] == ["3", "0.5000", "-", "-", "-"]
    assert line.endswith("*")


def test_console_metrics_and_dashboard_observers(tmp_path):
    stream = io.StringIO()
    metrics, dashboard = tmp_path / "metrics.jsonl", tmp_path / "dashboard.html"
    observers = [ConsoleProgress(stream), MetricsLog(metrics), DashboardWriter(dashboard, "run <1>", {"model": "test"})]
    new_session(dataset(1, 8), dataset(2, 4), observers=observers).run(3)

    output = stream.getvalue()
    assert "graphs/s" in output and "diagnosis:" in output
    assert len(output.strip().splitlines()) == 2 + 3 + 1
    lines = metrics.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["epoch"] for line in lines] == [1, 2, 3]
    page = dashboard.read_text(encoding="utf-8")
    assert "run &lt;1&gt;" in page and "<svg" in page
    assert 'http-equiv="refresh"' not in page


def test_dashboard_is_self_contained_and_refreshes_while_live():
    page = render_dashboard("t", history([(0.9, 0.95), (0.6, 0.7), (0.4, 0.5)]), ["VC-1"], {}, live=True)
    assert 'http-equiv="refresh"' in page
    assert not re.search(r"""(src|href)=["']https?://""", page)
    assert "<script" not in page
    assert page.count("<svg") == 4


def test_dashboard_handles_empty_history():
    assert "No data yet" in render_dashboard("t", [], ["VC-1"], {})


@pytest.mark.parametrize("relations, hidden", [(1, 4), (6, 16), (3, 33)])
def test_parameter_count_formula_matches_the_real_model(relations, hidden):
    config = GSDMConfig(feature_size=7, relation_count=relations, decision_count=3, hidden_size=hidden)
    assert parameter_count(config) == GSDModel(config).parameter_count()


def test_capacity_estimate_and_hidden_size_solver():
    config = GSDMConfig(feature_size=393, relation_count=6, decision_count=3, hidden_size=64)
    capacity = estimate(config)
    assert capacity.training_state_bytes == capacity.parameters * 16
    assert capacity.training_bytes(1000) > capacity.training_state_bytes
    hidden = hidden_size_for(10_000_000_000, 393, 6, 3)
    assert parameter_count(GSDMConfig(393, 6, 3, hidden_size=hidden)) >= 10_000_000_000
    assert parameter_count(GSDMConfig(393, 6, 3, hidden_size=hidden - 1)) < 10_000_000_000


def test_format_bytes():
    assert format_bytes(512) == "512 B"
    assert format_bytes(160 * 1024**3) == "160.0 GB"
