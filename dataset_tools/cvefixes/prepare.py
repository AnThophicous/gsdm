# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Imports the CVEfixes SQL dump into SQLite, streaming it straight out of the release zip."""

from __future__ import annotations

import gzip
import hashlib
import io
import sqlite3
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import BinaryIO, TextIO

ZENODO_URL = "https://zenodo.org/records/13118970/files/CVEfixes_v1.0.8.zip?download=1"
ZENODO_MD5 = "4586a358977acfa4c60b1a2cdd096221"
_CHUNK = 8 * 1024 * 1024
_PROGRESS_EVERY = 100_000
_DUMP_TRANSACTION_STATEMENTS = {"BEGIN TRANSACTION;", "BEGIN;", "COMMIT;", "END TRANSACTION;"}


class PrepareError(ValueError):
    pass


def md5_of(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as file:
        while chunk := file.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify_md5(path: Path, expected: str = ZENODO_MD5) -> None:
    actual = md5_of(path)
    if actual != expected:
        raise PrepareError(f"{path.name}: MD5 is {actual}, expected {expected}; the download is incomplete or different")


def prepare_from_zip(zip_path: Path, database_path: Path, progress: TextIO | None = None) -> int:
    with zipfile.ZipFile(zip_path) as archive:
        member = _dump_member(archive)
        if progress:
            progress.write(f"importing {member.filename} ({member.file_size / 2**30:.1f} GB uncompressed in zip)\n")
        with archive.open(member) as raw:
            return import_sql_dump(_text_lines(raw, member.filename), database_path, progress)


def prepare_from_dump(dump_path: Path, database_path: Path, progress: TextIO | None = None) -> int:
    with open(dump_path, "rb") as raw:
        return import_sql_dump(_text_lines(raw, dump_path.name), database_path, progress)


def import_sql_dump(lines: Iterable[str], database_path: Path, progress: TextIO | None = None) -> int:
    """Runs every statement of a SQLite dump in one transaction. Returns the number of statements executed."""
    if database_path.exists():
        raise PrepareError(f"{database_path} already exists; remove it first")
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("BEGIN")
        count = 0
        for count, statement in enumerate(_statements(lines), start=1):
            if statement.strip().upper() in _DUMP_TRANSACTION_STATEMENTS:
                continue
            try:
                connection.execute(statement)
            except sqlite3.Error as error:
                raise PrepareError(f"statement {count} failed ({error}): {statement[:200]!r}") from error
            if progress and count % _PROGRESS_EVERY == 0:
                progress.write(f"  {count:,} statements\n")
                progress.flush()
        connection.execute("COMMIT")
    except BaseException:
        connection.close()
        database_path.unlink(missing_ok=True)
        raise
    connection.close()
    return count


def _statements(lines: Iterable[str]) -> Iterator[str]:
    """Groups lines into complete SQL statements; a statement can span many lines inside string literals."""
    buffer: list[str] = []
    for line in lines:
        buffer.append(line)
        if line.rstrip().endswith(";"):
            candidate = "".join(buffer)
            if sqlite3.complete_statement(candidate):
                yield candidate
                buffer = []
    if "".join(buffer).strip():
        raise PrepareError("the dump ends in the middle of a statement")


def _dump_member(archive: zipfile.ZipFile) -> zipfile.ZipInfo:
    candidates = [info for info in archive.infolist() if info.filename.endswith((".sql.gz", ".sql"))]
    if len(candidates) != 1:
        names = [info.filename for info in archive.infolist()][:20]
        raise PrepareError(f"expected exactly one .sql or .sql.gz file in the zip, found {len(candidates)}: {names}")
    return candidates[0]


def _text_lines(raw: BinaryIO, name: str) -> TextIO:
    binary: BinaryIO = gzip.open(raw) if name.endswith(".gz") else raw  # type: ignore[assignment]
    return io.TextIOWrapper(binary, encoding="utf-8", errors="replace", newline="")
