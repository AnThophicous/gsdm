import gzip
import io
import json
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from dataset_tools.cvefixes.census import format_report, run_census
from dataset_tools.cvefixes.prepare import PrepareError, import_sql_dump, prepare_from_zip, verify_md5
from dataset_tools.cvefixes.schema import SchemaError, normalize_cwe

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

SQL_BEFORE = 'def find(cursor, uid):\n    cursor.execute("SELECT * FROM u WHERE id = " + uid)\n'
SQL_AFTER = 'def find(cursor, uid):\n    cursor.execute("SELECT * FROM u WHERE id = %s", (uid,))\n'
PICKLE_BEFORE = "import pickle\n\ndef load(blob):\n    return pickle.loads(blob)\n"
PICKLE_AFTER = "import json\n\ndef load(blob):\n    return json.loads(blob)\n"
PYTHON2 = 'print "hello"\nexcept_value = 1\n'


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE fixes (cve_id TEXT, hash TEXT, repo_url TEXT);
        CREATE TABLE cwe_classification (cve_id TEXT, cwe_id TEXT);
        CREATE TABLE file_change (
            file_change_id TEXT, hash TEXT, filename TEXT, change_type TEXT, code_before TEXT,
            code_after TEXT, diff TEXT, programming_language TEXT
        );
        """
    )


def add_fix(connection, file_id, commit, repo, cve, cwes, before, after, language="Python", diff="@@ -1 +1 @@"):
    connection.execute("INSERT INTO fixes VALUES (?, ?, ?)", (cve, commit, repo))
    for cwe in cwes:
        connection.execute("INSERT INTO cwe_classification VALUES (?, ?)", (cve, cwe))
    connection.execute(
        "INSERT INTO file_change VALUES (?, ?, ?, 'ModificationType.MODIFY', ?, ?, ?, ?)",
        (file_id, commit, f"{file_id}.py", before, after, diff, language),
    )


@pytest.fixture
def database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    create_schema(connection)
    add_fix(connection, "1", "c1", "https://github.com/a/app", "CVE-1", ["CWE-89"], SQL_BEFORE, SQL_AFTER)
    add_fix(connection, "2", "c2", "https://github.com/b/lib", "CVE-2", ["CWE-502", "CWE-20"], PICKLE_BEFORE, PICKLE_AFTER)
    add_fix(connection, "3", "c3", "https://github.com/c/fork", "CVE-3", ["CWE-89"], SQL_BEFORE, SQL_AFTER)
    add_fix(connection, "4", "c4", "https://github.com/d/old", "CVE-4", ["CWE-78"], PYTHON2, "print('hello')\n")
    add_fix(connection, "5", "c5", "https://github.com/e/new", "CVE-5", ["CWE-798"], None, 'KEY = "x"\n')
    add_fix(connection, "6", "c6", "https://github.com/f/js", "CVE-6", ["CWE-89"], "x;", "y;", language="JavaScript")
    add_fix(connection, "7", "c7", "https://github.com/g/web", "CVE-7", ["CWE-79"], "a = 1\n", "a = 2\n")
    add_fix(connection, "8", "c8", "https://github.com/h/same", "CVE-8", ["CWE-77"], "a = 1\n", "a = 1   \n")
    add_fix(connection, "9", "c9", "https://github.com/i/info", "CVE-9", ["NVD-CWE-noinfo"], "b = 1\n", "b = 2\n")
    connection.commit()
    return connection


def category(report, code):
    return next(c.to_dict() for c in report.categories if c.category == code)


def test_census_counts_usable_pairs_and_every_discard(database):
    report = run_census(database)

    assert report.languages == {"Python": 8, "JavaScript": 1}
    assert report.python_file_changes_with_cwe == 8
    assert category(report, "VC-1") == {
        "category": "VC-1",
        "file_changes": 4,
        "usable_positives": 1,
        "usable_hard_negatives": 1,
        "repositories": 1,
        "cves": 1,
    }
    assert category(report, "VC-2")["usable_positives"] == 1
    assert category(report, "VC-2")["usable_hard_negatives"] == 1
    assert category(report, "VC-3")["file_changes"] == 1 and category(report, "VC-3")["usable_positives"] == 0
    assert report.discarded == {
        "duplicate_content": 1,
        "no_code_before": 1,
        "unchanged_by_fix": 1,
        "unparseable_python2": 1,
    }
    assert dict(report.other_python_cwes) == {"CWE-79": 1, "CWE-20": 1}
    assert report.positive_graph_nodes["count"] == 2
    assert report.with_diff == 2


def test_report_text_and_json(database):
    report = run_census(database)
    text = format_report(report)
    assert "VC-1" in text and "unparseable_python2" in text and "CWE-89->VC-1" in text
    assert json.loads(json.dumps(report.to_dict()))["categories"][0]["category"] == "VC-1"


def test_census_refuses_an_unexpected_schema():
    connection = sqlite3.connect(":memory:")
    connection.executescript("CREATE TABLE fixes (cve_id TEXT); CREATE TABLE other (x TEXT);")
    with pytest.raises(SchemaError) as raised:
        run_census(connection)
    assert "fixes: missing" in str(raised.value) and "Actual schema" in str(raised.value)


@pytest.mark.parametrize(
    "raw, expected",
    [("CWE-89", "CWE-89"), (" cwe-502 ", "CWE-502"), ("89", "CWE-89"), ("NVD-CWE-Other", None), ("", None), (None, None)],
)
def test_normalize_cwe(raw, expected):
    assert normalize_cwe(raw) == expected


def dump_of(connection: sqlite3.Connection) -> str:
    return "\n".join(connection.iterdump()) + "\n"


def test_prepare_streams_a_gzipped_dump_out_of_a_zip(database, tmp_path):
    archive = tmp_path / "CVEfixes.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("CVEfixes/Data/CVEfixes.sql.gz", gzip.compress(dump_of(database).encode("utf-8")))
        zip_file.writestr("CVEfixes/README.md", "readme")
    output = tmp_path / "CVEfixes.db"
    assert prepare_from_zip(archive, output) > 0

    restored = sqlite3.connect(output)
    assert restored.execute("SELECT COUNT(*) FROM file_change").fetchone() == (9,)
    code = restored.execute("SELECT code_before FROM file_change WHERE file_change_id = '1'").fetchone()[0]
    assert code == SQL_BEFORE
    assert category(run_census(restored), "VC-1")["usable_positives"] == 1


def test_multi_line_string_with_semicolons_survives_import(tmp_path):
    source = sqlite3.connect(":memory:")
    source.execute("CREATE TABLE t (code TEXT)")
    tricky = "a = 1;\nb = 'x;';\r\nc = \"--\";\n"
    source.execute("INSERT INTO t VALUES (?)", (tricky,))
    output = tmp_path / "t.db"
    import_sql_dump(io.StringIO(dump_of(source), newline=""), output)
    assert sqlite3.connect(output).execute("SELECT code FROM t").fetchone()[0] == tricky


def test_prepare_rejects_existing_output_truncated_dumps_and_ambiguous_zips(tmp_path):
    existing = tmp_path / "exists.db"
    existing.write_bytes(b"")
    with pytest.raises(PrepareError):
        import_sql_dump(["CREATE TABLE t (x);\n"], existing)
    with pytest.raises(PrepareError):
        import_sql_dump(["CREATE TABLE t (x);\n", "INSERT INTO t VALUES ('unterminated\n"], tmp_path / "cut.db")
    assert not (tmp_path / "cut.db").exists()
    archive = tmp_path / "two.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("a.sql", "")
        zip_file.writestr("b.sql.gz", b"")
    with pytest.raises(PrepareError):
        prepare_from_zip(archive, tmp_path / "x.db")


def test_md5_mismatch_is_reported(tmp_path):
    file = tmp_path / "f.zip"
    file.write_bytes(b"not the release")
    with pytest.raises(PrepareError):
        verify_md5(file)


def test_command_line_census(database, tmp_path):
    path = tmp_path / "with space" / "CVEfixes.db"
    path.parent.mkdir()
    disk = sqlite3.connect(path)
    database.backup(disk)
    disk.close()
    report = tmp_path / "census.json"
    result = subprocess.run(
        [sys.executable, "-m", "dataset_tools.cvefixes", "census", str(path), "--report", str(report)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        env=_environment(),
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "CVEfixes census" in result.stdout
    assert json.loads(report.read_text(encoding="utf-8"))["with_diff"] == 2
    missing = subprocess.run(
        [sys.executable, "-m", "dataset_tools.cvefixes", "census", str(tmp_path / "none.db")],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        env=_environment(),
        timeout=120,
    )
    assert missing.returncode == 2 and "database not found" in missing.stderr


def _environment() -> dict[str, str]:
    import os

    return {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT)])}
