"""N10 phase 1 (ROM-85): one function tells a field leaf from a finite array (lots 1a and 1b).

Lot 1a computed the IR types, the index sets, the support placements and the layout records of a
plan, and only the tests and the manifest reporter read them. Lot 1b retyped the dispatch: every
line of ``python/pops`` that asked « is this a field » or « is this a finite array » by comparing
a ``vtype`` spelling now asks the one function of ``pops._ir.data_model`` (``value_kind``,
``is_field_leaf``, ``is_finite_array``, ``is_finite_index``, ``field_role``), which reads the kind
from the head of the IR type (note §3.4). These tests keep it so:

* no line outside that module compares a field or finite spelling (the dispatch inventory has
  role rows only, R3);
* the Program builder (``python/pops/time``) and the lowering (``python/pops/codegen``) import
  only the kind functions from what carries the type, so no acceptance or emitter reads a
  placement, a layout record or an index set before the lots that own them (A2);
* the layout records stay in the manifest (review of #749, T5).

DM-T1 checks the same thing from the other side: the emitted C++ of the 40 profiles is unchanged.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCES = ROOT / "python" / "pops"
GUARDED = ("codegen", "time")
CARRIERS = frozenset(("pops._ir.arrays", "pops._ir.index_sets", "pops._ir.data_model"))
CARRIER_NAMES = frozenset(("arrays", "index_sets", "data_model"))
KIND_FUNCTIONS = frozenset(("field_role", "is_field_leaf", "is_finite_array", "is_finite_index",
                            "value_kind"))
# The manifest reporter: the compiled plan record computes the field ``data_layout_and_distribution``
# there, outside its contract identity, and nothing reads it back.
REPORTER = "python/pops/codegen/_compiled_artifact.py"
# Reading the records needs no import (``compiled.plan.data_layout[...]``): the files that may
# name them are the record and the report that copies it into the manifest (review of #749, T5).
RECORD_READERS = frozenset((REPORTER, "python/pops/codegen/_inspect_compiled_report.py"))


def _imports(path: Path, sources: Path = SOURCES) -> list[tuple[int, str, tuple[str, ...]]]:
    """``(line, module, names)`` of every import of ``path``; ``names`` empty for ``import m``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rows = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            rows.extend((node.lineno, alias.name, ()) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                package = path.relative_to(sources.parent).with_suffix("").parts[:-node.level]
                module = ".".join((*package, module)) if module else ".".join(package)
            names = tuple(alias.name for alias in node.names)
            rows.append((node.lineno, module, names))
            if module in ("pops._ir",):
                rows.extend((node.lineno, "pops._ir." + name, ()) for name in names
                            if name in CARRIER_NAMES)
    return rows


def _offending(module: str, names: tuple[str, ...]) -> bool:
    if not (module in CARRIERS or any(module.startswith(name + ".") for name in CARRIERS)):
        return False
    return module != "pops._ir.data_model" or not names or not set(names) <= KIND_FUNCTIONS


def test_builder_and_lowering_read_only_the_kind_of_a_value():
    offenders = []
    for package in GUARDED:
        for path in sorted((SOURCES / package).rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if relative == REPORTER:
                continue
            for line, module, names in _imports(path):
                if _offending(module, names):
                    offenders.append("%s:%d imports %s %s" % (relative, line, module, names))
    assert not offenders, (
        "the builder and the lowering ask the kind of a value (lot 1b of N10); placements, "
        "index sets and layout records belong to later lots:\n" + "\n".join(offenders))


def test_no_line_compares_a_field_or_finite_spelling_outside_the_one_function():
    path = ROOT / "scripts" / "n10_dispatch_inventory.py"
    spec = importlib.util.spec_from_file_location("n10_dispatch_inventory_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [row for row in module.rows() if row[2] != "role"]
    assert not rows, (
        "ask pops._ir.data_model (value_kind, is_field_leaf, is_finite_array, is_finite_index, "
        "field_role) instead of comparing a vtype spelling:\n"
        + "\n".join("%s:%d %s" % (file, line, text) for file, line, _, text in rows))
    assert module._kind('if value.vtype in ("state", "rhs", "scalar_field"):') == "field"
    assert module._kind('if sig["vtype"] == "array":') == "finite"
    assert module._kind('if value.get("vtype") != "index_set":') == "finite"
    assert module._kind('if value.vtype == "rhs":') == "role"
    assert module._kind('signature = dict(domain, vtype="array")') is None


def test_no_emitter_or_acceptance_reads_the_layout_records():
    offenders = []
    for package in GUARDED:
        for path in sorted((SOURCES / package).rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if relative in RECORD_READERS:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if "data_layout" in line:
                    offenders.append("%s:%d %s" % (relative, number, line.strip()))
    assert not offenders, (
        "N10 reports the layout records in the manifest only; no emitter, dispatch or "
        "acceptance reads them:\n" + "\n".join(offenders))


def test_the_guard_sees_every_spelling_of_an_import(tmp_path):
    sources = tmp_path / "pops"
    fake = sources / "codegen" / "module.py"
    fake.parent.mkdir(parents=True)
    for text in ("import pops._ir.arrays\n", "from pops._ir import index_sets\n",
                 "from pops._ir.data_model import ir_type\n", "from .._ir import arrays\n",
                 "import pops._ir.data_model\n",
                 "from pops._ir.data_model import is_field_leaf, support_placements\n"):
        fake.write_text(text, encoding="utf-8")
        rows = _imports(fake, sources)
        assert any(_offending(module, names) for _, module, names in rows), (text, rows)
    fake.write_text("from pops._ir.data_model import field_role, is_field_leaf\n",
                    encoding="utf-8")
    assert not any(_offending(module, names) for _, module, names in _imports(fake, sources))


def test_the_one_admitted_reader_only_reports():
    text = (ROOT / REPORTER).read_text(encoding="utf-8")
    assert text.count("pops._ir.data_model") == 2  # the docstring and the one import
    assert text.count("_data_layout(") == 2       # its definition and its one call
    assert "data_layout" not in text[text.index("def _payload(self)"):][:4000].split(
        "def verify", 1)[0]
