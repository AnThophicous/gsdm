# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The parts of the CVEfixes schema the tools rely on, checked before any query (ADR-0010, item 9)."""

from __future__ import annotations

import sqlite3

REQUIRED_COLUMNS = {
    "fixes": {"cve_id", "hash", "repo_url"},
    "cwe_classification": {"cve_id", "cwe_id"},
    "file_change": {
        "file_change_id",
        "hash",
        "filename",
        "change_type",
        "code_before",
        "code_after",
        "diff",
        "programming_language",
    },
}

CWE_CATEGORIES = {
    "CWE-89": "VC-1",
    "CWE-77": "VC-1",
    "CWE-78": "VC-1",
    "CWE-88": "VC-1",
    "CWE-502": "VC-2",
    "CWE-798": "VC-3",
    "CWE-259": "VC-3",
    "CWE-321": "VC-3",
}


class SchemaError(ValueError):
    pass


def verify_schema(connection: sqlite3.Connection) -> None:
    actual = {table: table_columns(connection, table) for table in REQUIRED_COLUMNS}
    problems = [
        f"{table}: missing {sorted(columns - actual[table])}"
        for table, columns in REQUIRED_COLUMNS.items()
        if not columns <= actual[table]
    ]
    if problems:
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name").fetchall()
        layout = {name: sorted(table_columns(connection, name)) for (name,) in tables}
        raise SchemaError("CVEfixes schema mismatch: " + "; ".join(problems) + f". Actual schema: {layout}")


def table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({_quoted(table)})")}


def normalize_cwe(raw: str | None) -> str | None:
    if not raw:
        return None
    value = raw.strip().upper()
    if value.isdigit():
        value = f"CWE-{value}"
    return value if value.startswith("CWE-") and value[4:].isdigit() else None


def _quoted(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
