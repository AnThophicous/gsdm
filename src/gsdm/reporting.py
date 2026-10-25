# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Training logs: one console line and one JSON line per epoch, plus a self-contained HTML dashboard (ADR-0009)."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from gsdm.diagnostics import LearningDiagnosis, LearningStatus, diagnose
from gsdm.session import EpochReport, TrainingObserver, TrainingSession

LIVE_REFRESH_SECONDS = 15


class ConsoleProgress(TrainingObserver):
    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self._header_written = False

    def on_epoch_end(self, report: EpochReport, session: TrainingSession) -> None:
        if not self._header_written:
            self._write(_header(session.decision_names))
            self._header_written = True
        self._write(format_epoch(report))

    def on_training_end(self, session: TrainingSession) -> None:
        diagnosis = diagnose(session.history)
        stopped = " (early stop)" if session.stopped_early else ""
        self._write(f"best epoch: {session.best_epoch}{stopped} | diagnosis: {diagnosis.status.value} - {diagnosis.evidence}")

    def _write(self, line: str) -> None:
        self.stream.write(line + "\n")
        self.stream.flush()


class MetricsLog(TrainingObserver):
    """Appends one JSON object per epoch; a resumed run keeps appending to the same file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def on_epoch_end(self, report: EpochReport, session: TrainingSession) -> None:
        with open(self.path, "a", encoding="utf-8") as file:
            file.write(json.dumps(report.to_dict()) + "\n")


class DashboardWriter(TrainingObserver):
    """Rewrites the dashboard every epoch (auto-refreshing while training runs) and once more at the end."""

    def __init__(self, path: Path, title: str, facts: Mapping[str, str] | None = None) -> None:
        self.path = path
        self.title = title
        self.facts = dict(facts or {})

    def on_epoch_end(self, report: EpochReport, session: TrainingSession) -> None:
        self._write(session, live=True)

    def on_training_end(self, session: TrainingSession) -> None:
        self._write(session, live=False)

    def _write(self, session: TrainingSession, live: bool) -> None:
        document = render_dashboard(self.title, session.history, session.decision_names, self.facts, live)
        self.path.write_text(document, encoding="utf-8")


def format_epoch(report: EpochReport) -> str:
    cells = [
        f"{report.epoch:>5}",
        f"{report.training_loss:>8.4f}",
        _cell(report.validation_loss, "8.4f"),
        _cell(report.generalization_gap, "+8.4f"),
        _cell(report.calibration_error, "6.3f"),
        *(_cell(decision.f1, "7.3f") for decision in report.decisions),
        f"{report.graphs_per_second:>9.1f}",
    ]
    return "  ".join(cells) + ("  *" if report.is_best else "")


def _header(decision_names: Sequence[str]) -> str:
    names = ["epoch", "   train", "     val", "     gap", "   ECE", *(f"{n[:7]:>7}" for n in decision_names), " graphs/s"]
    return "  ".join(names) + "\n" + "  ".join("-" * len(n) for n in names) + "    (F1 per decision on validation, * = best)"


def _cell(value: float | None, spec: str) -> str:
    width = int("".join(c for c in spec.split(".")[0] if c.isdigit()) or 0)
    return f"{'-':>{width}}" if value is None else format(value, spec)


@dataclass(frozen=True)
class _Series:
    label: str
    values: Sequence[float | None]
    color: str


_STATUS_TONE = {
    LearningStatus.LEARNING: "good",
    LearningStatus.MEMORIZING: "bad",
    LearningStatus.STALLED: "warn",
    LearningStatus.NO_VALIDATION: "warn",
    LearningStatus.TOO_SHORT: "neutral",
}


def render_dashboard(
    title: str,
    history: Sequence[EpochReport],
    decision_names: Sequence[str],
    facts: Mapping[str, str],
    live: bool = False,
) -> str:
    diagnosis = diagnose(history)
    epochs = [report.epoch for report in history]
    best = diagnosis.best_epoch
    charts = [
        _chart(
            "Loss",
            "Training should fall. Validation should fall with it; if it rises while training keeps falling, the model is memorizing.",
            epochs,
            [
                _Series("training", [r.training_loss for r in history], "var(--series-1)"),
                _Series("validation", [r.validation_loss for r in history], "var(--series-2)"),
            ],
            best,
        ),
        _chart(
            "Generalization gap",
            "Validation loss minus training loss. A gap that keeps growing means the model does not transfer to unseen code.",
            epochs,
            [_Series("gap", [r.generalization_gap for r in history], "var(--series-3)")],
            best,
            zero_line=True,
        ),
        _chart(
            "Validation F1 per decision",
            "How well each decision is made on code the model never trained on.",
            epochs,
            [
                _Series(name, [r.decisions[i].f1 if r.decisions else None for r in history], f"var(--series-{i % 4 + 1})")
                for i, name in enumerate(decision_names)
            ],
            best,
            fixed_range=(0.0, 1.0),
        ),
        _chart(
            "Calibration error (ECE, before calibration)",
            "Gap between stated confidence and real accuracy on validation. Lower is better.",
            epochs,
            [_Series("ECE", [r.calibration_error for r in history], "var(--series-4)")],
            best,
        ),
    ]
    refresh = f'<meta http-equiv="refresh" content="{LIVE_REFRESH_SECONDS}">' if live else ""
    return _PAGE.format(
        refresh=refresh,
        title=html.escape(title),
        banner=_banner(diagnosis, live),
        facts=_facts(facts, history),
        charts="\n".join(charts),
        table=_decision_table(history),
    )


def _banner(diagnosis: LearningDiagnosis, live: bool) -> str:
    tone = _STATUS_TONE[diagnosis.status]
    state = "training in progress" if live else "training finished"
    return (
        f'<section class="banner {tone}"><strong>{html.escape(diagnosis.status.value)}</strong>'
        f"<span>{html.escape(diagnosis.evidence)}</span><small>{state}</small></section>"
    )


def _facts(facts: Mapping[str, str], history: Sequence[EpochReport]) -> str:
    rows = dict(facts)
    if history:
        rows["epochs run"] = str(history[-1].epoch)
        rows["throughput"] = f"{history[-1].graphs_per_second:.1f} graphs/s"
    items = "".join(f"<div><dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd></div>" for k, v in rows.items())
    return f'<dl class="facts">{items}</dl>'


def _decision_table(history: Sequence[EpochReport]) -> str:
    if not history or not history[-1].decisions:
        return ""
    rows = "".join(
        f"<tr><th>{html.escape(d.name)}</th><td>{_fmt(d.precision)}</td><td>{_fmt(d.recall)}</td>"
        f"<td>{_fmt(d.f1)}</td><td>{d.positives}</td></tr>"
        for d in history[-1].decisions
    )
    return (
        f"<section class='card'><h2>Last epoch on validation</h2><table><thead><tr><th>decision</th>"
        f"<th>precision</th><th>recall</th><th>F1</th><th>positives</th></tr></thead><tbody>{rows}</tbody></table></section>"
    )


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


_WIDTH, _HEIGHT = 480, 220
_LEFT, _RIGHT, _TOP, _BOTTOM = 58, 12, 18, 30


def _chart(
    title: str,
    caption: str,
    epochs: Sequence[int],
    series: Sequence[_Series],
    best_epoch: int | None,
    zero_line: bool = False,
    fixed_range: tuple[float, float] | None = None,
) -> str:
    values = [v for s in series for v in s.values if v is not None]
    if not values:
        body = '<p class="empty">No data yet.</p>'
    else:
        low, high = fixed_range or _padded_range(values, include_zero=zero_line)
        scale = _Scale(epochs[0], max(epochs[-1], epochs[0] + 1), low, high)
        body = _svg(title, epochs, series, scale, best_epoch, zero_line)
    legend = "".join(
        f'<li><span class="swatch" style="background:{s.color}"></span>{html.escape(s.label)}</li>' for s in series
    )
    return (
        f'<section class="card"><h2>{html.escape(title)}</h2><p class="caption">{html.escape(caption)}</p>'
        f'{body}<ul class="legend">{legend}</ul></section>'
    )


@dataclass(frozen=True)
class _Scale:
    first_epoch: int
    last_epoch: int
    low: float
    high: float

    def x(self, epoch: float) -> float:
        span = self.last_epoch - self.first_epoch
        return _LEFT + (epoch - self.first_epoch) / span * (_WIDTH - _LEFT - _RIGHT)

    def y(self, value: float) -> float:
        return _TOP + (self.high - value) / (self.high - self.low) * (_HEIGHT - _TOP - _BOTTOM)


def _svg(
    title: str, epochs: Sequence[int], series: Sequence[_Series], scale: _Scale, best_epoch: int | None, zero_line: bool
) -> str:
    parts = [f'<svg viewBox="0 0 {_WIDTH} {_HEIGHT}" role="img" aria-label="{html.escape(title)}">']
    for step in range(5):
        value = scale.low + (scale.high - scale.low) * step / 4
        y = scale.y(value)
        parts.append(f'<line class="grid" x1="{_LEFT}" x2="{_WIDTH - _RIGHT}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{_LEFT - 6}" y="{y + 4:.1f}" text-anchor="end">{value:.3g}</text>')
    if zero_line and scale.low < 0 < scale.high:
        parts.append(f'<line class="zero" x1="{_LEFT}" x2="{_WIDTH - _RIGHT}" y1="{scale.y(0):.1f}" y2="{scale.y(0):.1f}"/>')
    if best_epoch is not None:
        x = scale.x(best_epoch)
        parts.append(f'<line class="best" x1="{x:.1f}" x2="{x:.1f}" y1="{_TOP}" y2="{_HEIGHT - _BOTTOM}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{_TOP - 4}" text-anchor="middle">best</text>')
    for epoch in sorted({epochs[0], epochs[-1]} | ({best_epoch} if best_epoch else set())):
        parts.append(f'<text class="tick" x="{scale.x(epoch):.1f}" y="{_HEIGHT - 8}" text-anchor="middle">{epoch}</text>')
    for line in series:
        parts.extend(_polylines(epochs, line, scale))
    parts.append("</svg>")
    return "".join(parts)


def _polylines(epochs: Sequence[int], series: _Series, scale: _Scale) -> list[str]:
    runs: list[list[str]] = [[]]
    for epoch, value in zip(epochs, series.values):
        if value is None:
            runs.append([])
        else:
            runs[-1].append(f"{scale.x(epoch):.1f},{scale.y(value):.1f}")
    shapes = []
    for points in filter(None, runs):
        if len(points) == 1:
            x, y = points[0].split(",")
            shapes.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{series.color}"/>')
        else:
            shapes.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{series.color}" stroke-width="2"/>')
    return shapes


def _padded_range(values: Sequence[float], include_zero: bool) -> tuple[float, float]:
    low, high = min(values), max(values)
    if include_zero:
        low, high = min(low, 0.0), max(high, 0.0)
    if high == low:
        return low - 1.0, high + 1.0
    padding = (high - low) * 0.08
    floor = 0.0 if low >= 0 else -float("inf")
    return max(low - padding, floor), high + padding


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{refresh}
<title>{title}</title>
<style>
:root {{
  --bg: #f7f7f5; --surface: #ffffff; --text: #1d1d1b; --muted: #6b6b66; --line: #e2e2dd;
  --series-1: #2f6fde; --series-2: #e0612f; --series-3: #2c9c6a; --series-4: #9a4fd1;
  --good: #1f8a56; --warn: #b7791f; --bad: #c53030; --neutral: #6b6b66;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #141413; --surface: #1e1e1c; --text: #ecece8; --muted: #a2a29c; --line: #34342f;
    --series-1: #6a9bf0; --series-2: #f08a5d; --series-3: #4fc28d; --series-4: #b98ae6;
    --good: #4fc28d; --warn: #e0a84a; --bad: #f07070; --neutral: #a2a29c;
  }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--text); font: 15px/1.5 system-ui, sans-serif; }}
main {{ max-width: 1080px; margin: 0 auto; padding: 24px 16px 48px; }}
h1 {{ font-size: 22px; margin: 0 0 16px; }}
h2 {{ font-size: 16px; margin: 0 0 4px; }}
.banner {{ display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: baseline; padding: 14px 16px;
  border-radius: 10px; background: var(--surface); border-left: 5px solid var(--neutral); margin-bottom: 16px; }}
.banner strong {{ text-transform: uppercase; letter-spacing: .04em; }}
.banner small {{ color: var(--muted); margin-left: auto; }}
.good {{ border-left-color: var(--good); }} .good strong {{ color: var(--good); }}
.warn {{ border-left-color: var(--warn); }} .warn strong {{ color: var(--warn); }}
.bad {{ border-left-color: var(--bad); }} .bad strong {{ color: var(--bad); }}
.facts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 8px; margin: 0 0 16px; }}
.facts div {{ background: var(--surface); border-radius: 10px; padding: 10px 12px; }}
dt {{ color: var(--muted); font-size: 12px; }} dd {{ margin: 0; font-weight: 600; }}
.grid-charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 460px), 1fr)); gap: 16px; }}
.card {{ background: var(--surface); border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
.caption {{ color: var(--muted); font-size: 13px; margin: 0 0 8px; }}
svg {{ width: 100%; height: auto; display: block; }}
svg .grid {{ stroke: var(--line); }} svg .zero {{ stroke: var(--muted); stroke-dasharray: 4 3; }}
svg .best {{ stroke: var(--muted); stroke-dasharray: 2 4; }}
svg .tick {{ fill: var(--muted); font-size: 15px; }}
.legend {{ list-style: none; display: flex; flex-wrap: wrap; gap: 12px; padding: 0; margin: 8px 0 0; font-size: 13px; }}
.swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }}
.empty {{ color: var(--muted); }}
table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--line); }}
th:first-child {{ text-align: left; }}
.table-wrap {{ overflow-x: auto; }}
</style>
</head>
<body>
<main>
<h1>{title}</h1>
{banner}
{facts}
<div class="grid-charts">
{charts}
</div>
<div class="table-wrap">{table}</div>
</main>
</body>
</html>
"""
