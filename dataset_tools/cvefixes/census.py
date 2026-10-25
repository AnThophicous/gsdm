# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Counts how much usable GVF data CVEfixes holds before anything is built (ADR-0010, item 6)."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from dataset_tools.cvefixes.schema import CWE_CATEGORIES, normalize_cwe, verify_schema
from gvf.domain import CodeAnalysisError, CodeUnit
from gvf.infrastructure.python_graph import PythonGraphBuilder

MAX_LINES = 5000
TOP_OTHER_CWES = 15
_FETCH_BATCH = 500
_PYTHON2 = re.compile(r"^\s*print\s+[^\s(=]|^\s*except\s+[\w.]+\s*,\s*\w+\s*:|^\s*exec\s+[\"'\w]", re.MULTILINE)


@dataclass
class CategoryCensus:
    category: str
    file_changes: int = 0
    positives: int = 0
    hard_negatives: int = 0
    repositories: set[str] = field(default_factory=set)
    cves: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "file_changes": self.file_changes,
            "usable_positives": self.positives,
            "usable_hard_negatives": self.hard_negatives,
            "repositories": len(self.repositories),
            "cves": len(self.cves),
        }


@dataclass
class CensusReport:
    languages: dict[str, int]
    python_file_changes_with_cwe: int
    categories: list[CategoryCensus]
    discarded: dict[str, int]
    positive_graph_nodes: dict[str, int]
    with_diff: int
    other_python_cwes: list[tuple[str, int]]
    cwe_mapping: dict[str, str] = field(default_factory=lambda: dict(CWE_CATEGORIES))

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["categories"] = [category.to_dict() for category in self.categories]
        return values


@dataclass(frozen=True)
class _FileChange:
    file_change_id: str
    repositories: frozenset[str]
    cves: frozenset[str]
    categories: frozenset[str]


def run_census(connection: sqlite3.Connection) -> CensusReport:
    verify_schema(connection)
    languages = _language_counts(connection)
    changes, other_cwes, python_total = _python_file_changes(connection)
    categories = {code: CategoryCensus(code) for code in sorted(set(CWE_CATEGORIES.values()))}
    discarded: Counter[str] = Counter()
    nodes: list[int] = []
    with_diff = 0
    seen: set[str] = set()
    builder = PythonGraphBuilder()

    for change, before, after, diff in _with_code(connection, changes):
        for code in change.categories:
            categories[code].file_changes += 1
        reason = _positive_rejection(before, after)
        if reason is None:
            reason = _duplicate_or_none(before, seen)
        if reason is None:
            node_count = _node_count(builder, change.file_change_id, before)
            reason = None if node_count else ("unparseable_python2" if _PYTHON2.search(before) else "unparseable")
        if reason:
            discarded[reason] += 1
            continue
        nodes.append(node_count)
        with_diff += bool(diff and diff.strip())
        hard_negative = _usable_hard_negative(builder, change.file_change_id, after, seen, discarded)
        for code in change.categories:
            census = categories[code]
            census.positives += 1
            census.hard_negatives += hard_negative
            census.repositories |= change.repositories
            census.cves |= change.cves

    return CensusReport(
        languages=languages,
        python_file_changes_with_cwe=python_total,
        categories=list(categories.values()),
        discarded=dict(sorted(discarded.items())),
        positive_graph_nodes=_percentiles(nodes),
        with_diff=with_diff,
        other_python_cwes=other_cwes.most_common(TOP_OTHER_CWES),
    )


def format_report(report: CensusReport) -> str:
    lines = [
        "CVEfixes census for GVF (ADR-0010)",
        "",
        f"Python file changes linked to any CWE: {report.python_file_changes_with_cwe:,}",
        "",
        f"{'category':<9}{'file changes':>14}{'positives':>11}{'hard negatives':>16}{'repos':>8}{'CVEs':>7}",
    ]
    for category in report.categories:
        row = category.to_dict()
        lines.append(
            f"{row['category']:<9}{row['file_changes']:>14,}{row['usable_positives']:>11,}"
            f"{row['usable_hard_negatives']:>16,}{row['repositories']:>8,}{row['cves']:>7,}"
        )
    lines += ["", "discarded:"] + [f"  {reason}: {count:,}" for reason, count in report.discarded.items()]
    lines += ["", f"graph nodes per positive file: {report.positive_graph_nodes}"]
    lines += [f"positives with a diff (usable for localization checks): {report.with_diff:,}"]
    lines += ["", "most common other CWEs in Python (candidates for later categories):"]
    lines += [f"  {cwe}: {count:,}" for cwe, count in report.other_python_cwes]
    lines += ["", "CWE mapping: " + ", ".join(f"{cwe}->{code}" for cwe, code in report.cwe_mapping.items())]
    return "\n".join(lines)


def _language_counts(connection: sqlite3.Connection) -> dict[str, int]:
    rows = connection.execute(
        "SELECT COALESCE(programming_language, '(none)'), COUNT(*) FROM file_change GROUP BY 1 ORDER BY 2 DESC"
    )
    return {language: count for language, count in rows}


def _python_file_changes(connection: sqlite3.Connection) -> tuple[list[_FileChange], Counter[str], int]:
    rows = connection.execute(
        """
        SELECT fc.file_change_id, f.repo_url, f.cve_id, cc.cwe_id
        FROM file_change fc
        JOIN fixes f ON f.hash = fc.hash
        JOIN cwe_classification cc ON cc.cve_id = f.cve_id
        WHERE lower(fc.programming_language) = 'python'
        """
    )
    grouped: dict[str, dict[str, set[str]]] = {}
    for file_change_id, repository, cve, raw_cwe in rows:
        entry = grouped.setdefault(str(file_change_id), {"repositories": set(), "cves": set(), "cwes": set()})
        entry["repositories"].add(repository)
        entry["cves"].add(cve)
        cwe = normalize_cwe(raw_cwe)
        if cwe:
            entry["cwes"].add(cwe)
    other_cwes: Counter[str] = Counter()
    changes = []
    for file_change_id, entry in grouped.items():
        categories = frozenset(CWE_CATEGORIES[cwe] for cwe in entry["cwes"] if cwe in CWE_CATEGORIES)
        other_cwes.update(cwe for cwe in entry["cwes"] if cwe not in CWE_CATEGORIES)
        if categories:
            changes.append(
                _FileChange(file_change_id, frozenset(entry["repositories"]), frozenset(entry["cves"]), categories)
            )
    return changes, other_cwes, len(grouped)


def _with_code(
    connection: sqlite3.Connection, changes: Sequence[_FileChange]
) -> Iterator[tuple[_FileChange, str | None, str | None, str | None]]:
    by_id = {change.file_change_id: change for change in changes}
    identifiers = sorted(by_id)
    for start in range(0, len(identifiers), _FETCH_BATCH):
        batch = identifiers[start : start + _FETCH_BATCH]
        placeholders = ",".join("?" * len(batch))
        rows = connection.execute(
            f"SELECT file_change_id, code_before, code_after, diff FROM file_change WHERE file_change_id IN ({placeholders})",
            batch,
        )
        for file_change_id, before, after, diff in rows:
            yield by_id[str(file_change_id)], before, after, diff


def _positive_rejection(before: str | None, after: str | None) -> str | None:
    if not before or not before.strip():
        return "no_code_before"
    if before.count("\n") + 1 > MAX_LINES:
        return "too_large"
    if after is not None and _normalized(before) == _normalized(after):
        return "unchanged_by_fix"
    return None


def _duplicate_or_none(source: str, seen: set[str]) -> str | None:
    digest = hashlib.sha256(source.encode("utf-8", errors="replace")).hexdigest()
    if digest in seen:
        return "duplicate_content"
    seen.add(digest)
    return None


def _usable_hard_negative(
    builder: PythonGraphBuilder, file_change_id: str, after: str | None, seen: set[str], discarded: Counter[str]
) -> int:
    if not after or not after.strip():
        discarded["after_missing"] += 1
        return 0
    if _duplicate_or_none(after, seen):
        discarded["after_duplicate"] += 1
        return 0
    if not _node_count(builder, file_change_id, after):
        discarded["after_unparseable"] += 1
        return 0
    return 1


def _node_count(builder: PythonGraphBuilder, file_change_id: str, source: str) -> int:
    try:
        return builder.build(CodeUnit(f"cvefixes/{file_change_id}.py", source)).structure.node_count
    except (CodeAnalysisError, RecursionError, MemoryError):
        return 0


def _normalized(source: str) -> str:
    return "\n".join(line.rstrip() for line in source.strip().splitlines())


def _percentiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    array = np.array(values)
    return {
        "count": int(array.size),
        "p50": int(np.percentile(array, 50)),
        "p90": int(np.percentile(array, 90)),
        "p99": int(np.percentile(array, 99)),
        "max": int(array.max()),
    }
