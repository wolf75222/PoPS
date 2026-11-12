"""The registry of the vertical slice (OD31; tranche/note_tranche_verticale.md, section 6; MG6).

``tests/gates/vertical_slice.toml`` lists the pieces of the slice's case file that PoPS does not
run yet, in the order of the file; ``scripts/vertical_slice.py`` checks it against the tree and
against its base (the list only shrinks). Each rule fails on a tree written for it. The test reads
the source tree only.
"""
from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("vertical_slice",
                                              ROOT / "scripts" / "vertical_slice.py")
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

HEADER = """\
schema_version = 1
gate = "vertical-slice"
case = "case.py"
"""
FIRST = """
[[piece]]
id = "first"
statement = "import first"
refusal = "ImportError: no first"
owner = "a lot"
"""
SECOND = """
[[piece]]
id = "second"
statement = "second()"
refusal = "TypeError: no second"
owner = "another lot"
"""


def _tree(root, registry, case="import first\nsecond()\n"):
    (root / "tests" / "gates").mkdir(parents=True)
    (root / gate.REGISTRY).write_text(textwrap.dedent(registry), encoding="utf-8")
    (root / "case.py").write_text(case, encoding="utf-8")
    return root


def test_the_registry_of_the_tree_holds():
    assert gate.check(ROOT) == []
    registry = gate.read(ROOT)
    assert registry["case"] == \
        "examples/surface/tutorials/scalar_advection/02_openmp_explicit_ssprk2.py"
    assert [piece["id"] for piece in registry["piece"]] == ["B-import", "C"]


def test_the_pieces_follow_the_order_of_the_case_file(tmp_path):
    assert gate.check(_tree(tmp_path / "ordered", HEADER + FIRST + SECOND)) == []
    assert gate.check(_tree(tmp_path / "swapped", HEADER + SECOND + FIRST)) == [
        "the pieces are not in the order of the case file: lines [2, 1]"]


def test_a_statement_is_one_line_of_the_case_file(tmp_path):
    with pytest.raises(gate.RegistryError, match="not one"):
        gate.check(_tree(tmp_path, HEADER + FIRST, case="second()\n"))


def test_a_piece_names_its_statement_refusal_and_owner():
    with pytest.raises(gate.RegistryError, match="id, statement, refusal and owner"):
        gate.load(HEADER + FIRST.replace('owner = "a lot"\n', ""))
    with pytest.raises(gate.RegistryError, match="'Type: message'"):
        gate.load(HEADER + FIRST.replace("ImportError: no first", "no first"))
    with pytest.raises(gate.RegistryError, match="listed twice"):
        gate.load(HEADER + FIRST + FIRST)


def test_the_list_only_shrinks():
    base = gate.load(HEADER + FIRST)
    assert gate.grown(gate.load(HEADER), base) == []
    assert gate.grown(gate.load(HEADER + FIRST + SECOND), base) == [
        "[[piece]] second is not in the base"]
