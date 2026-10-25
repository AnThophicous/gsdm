import pytest

from gvf.domain import CodeSourceNotFoundError
from gvf.infrastructure.filesystem.local_code_repository import LocalCodeRepository


def test_loads_a_single_file(tmp_path):
    file = tmp_path / "app.py"
    file.write_bytes(b"x = 1\n")
    [unit] = LocalCodeRepository().load(str(file))
    assert unit.source == "x = 1\n"
    assert unit.path.endswith("app.py")


def test_loads_python_files_recursively_in_order_skipping_tooling_directories(tmp_path):
    for relative in ["b.py", "pkg/a.py", "pkg/notes.txt", ".venv/lib/site.py", "pkg/__pycache__/c.py"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pass\n", encoding="utf-8")
    paths = [unit.path for unit in LocalCodeRepository().load(str(tmp_path))]
    assert [path.removeprefix(tmp_path.as_posix() + "/") for path in paths] == ["b.py", "pkg/a.py"]


def test_decodes_utf8_with_bom_and_replaces_invalid_bytes(tmp_path):
    file = tmp_path / "enc.py"
    file.write_bytes(b"\xef\xbb\xbfname = 'caf\xc3\xa9'\nbad = '\xff'\n")
    [unit] = LocalCodeRepository().load(str(file))
    assert unit.source.startswith("name = 'café'")
    assert "�" in unit.source


def test_missing_path_raises(tmp_path):
    with pytest.raises(CodeSourceNotFoundError):
        LocalCodeRepository().load(str(tmp_path / "nothing"))
