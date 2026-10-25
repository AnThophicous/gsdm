"""Enforces the Clean Architecture dependency rule (ADR-0003) by inspecting imports."""

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"

FORBIDDEN_IMPORTS = {
    "gvf/domain": ("gvf.application", "gvf.adapters", "gvf.infrastructure", "gvf.main", "gsdm", "numpy"),
    "gvf/application": ("gvf.adapters", "gvf.infrastructure", "gvf.main", "gsdm", "numpy"),
    "gvf/adapters": ("gvf.infrastructure", "gvf.main", "gsdm", "numpy"),
    "gvf/infrastructure": ("gvf.main",),
    "gsdm": ("gvf",),
}


def imported_modules(file: Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def violates(module: str, forbidden: str) -> bool:
    return module == forbidden or module.startswith(forbidden + ".")


@pytest.mark.parametrize("layer", sorted(FORBIDDEN_IMPORTS))
def test_layer_respects_the_dependency_rule(layer):
    files = sorted((SOURCE_ROOT / layer).rglob("*.py"))
    assert files, f"no source files found for {layer}"
    violations = [
        f"{file.relative_to(SOURCE_ROOT).as_posix()} imports {module}"
        for file in files
        for module in imported_modules(file)
        for forbidden in FORBIDDEN_IMPORTS[layer]
        if violates(module, forbidden)
    ]
    assert not violations, "\n".join(violations)


def test_every_source_file_carries_the_mpl_notice():
    missing = [
        file.relative_to(SOURCE_ROOT).as_posix()
        for file in SOURCE_ROOT.rglob("*.py")
        if "Mozilla Public" not in file.read_text(encoding="utf-8")[:200]
    ]
    assert not missing, missing
