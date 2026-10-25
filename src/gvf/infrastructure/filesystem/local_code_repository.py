# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from pathlib import Path

from gvf.application.ports import CodeRepository
from gvf.domain import CodeSourceNotFoundError, CodeUnit

IGNORED_DIRECTORIES = frozenset(
    {".git", ".hg", ".venv", "venv", "env", "__pycache__", "node_modules", ".tox", ".mypy_cache", ".pytest_cache"}
)


class LocalCodeRepository(CodeRepository):
    def load(self, path: str) -> list[CodeUnit]:
        target = Path(path)
        if target.is_file():
            return [_read(target)]
        if target.is_dir():
            return [_read(file) for file in sorted(target.rglob("*.py")) if _is_analyzable(file, target)]
        raise CodeSourceNotFoundError(path)


def _is_analyzable(file: Path, root: Path) -> bool:
    relative_parts = file.relative_to(root).parts[:-1]
    return file.is_file() and not any(part in IGNORED_DIRECTORIES for part in relative_parts)


def _read(file: Path) -> CodeUnit:
    return CodeUnit(path=file.as_posix(), source=file.read_bytes().decode("utf-8-sig", errors="replace"))
