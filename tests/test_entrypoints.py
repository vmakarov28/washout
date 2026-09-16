"""The entry points must at least parse.

Every other test imports the washout package, and none imports run.py or the
scripts -- so a patch that left a raw newline inside an f-string in
run.py passed all 47 of them, and the next search the fleet launched
would have died on its first line. Compiling is cheap and catches exactly
that class of breakage before hours of compute are queued behind it.
"""

import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINTS = [ROOT / "run.py"] + sorted((ROOT / "scripts").glob("*.py"))


@pytest.mark.parametrize("path", ENTRYPOINTS, ids=lambda p: p.name)
def test_entry_point_compiles(path, tmp_path):
    py_compile.compile(str(path), cfile=str(tmp_path / "out.pyc"), doraise=True)
