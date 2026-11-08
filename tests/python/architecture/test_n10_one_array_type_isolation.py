"""N10 phase 1, lot 1a (ROM-85): nothing reads the one array type yet (review of the note, A2).

Lot 1a computes the IR types, the index sets, the support placements and the layout records of a
plan, and only the tests and the manifest reporter read them. The move of the mesh profiles to
the one type is lot 1b (MG3), where the dispatch of the emitters reads it. Until then, no module
of the Program builder (``python/pops/time``) or of the lowering (``python/pops/codegen``) may
import what carries the type, so no acceptance, dispatch or emitter entry point can depend on it.
DM-T1 checks the same thing from the other side: the emitted C++ of the 40 profiles is unchanged.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCES = ROOT / "python" / "pops"
GUARDED = ("codegen", "time")
CARRIERS = frozenset(("pops._ir.arrays", "pops._ir.index_sets", "pops._ir.data_model"))
CARRIER_NAMES = frozenset(("arrays", "index_sets", "data_model"))
# The one reader lot 1a admits: the compiled plan record computes the manifest field
# ``data_layout_and_distribution`` there, outside its contract identity, and nothing reads it back.
ALLOWED = frozenset(("python/pops/codegen/_compiled_artifact.py",))
# Reading the records needs no import (``compiled.plan.data_layout[...]``): the files that may
# name them are the record and the report that copies it into the manifest (review of #749, T5).
RECORD_READERS = frozenset((
    "python/pops/codegen/_compiled_artifact.py", "python/pops/codegen/_inspect_compiled_report.py",
))


def _imports(path: Path, sources: Path = SOURCES) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rows = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            rows.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                package = path.relative_to(sources.parent).with_suffix("").parts[:-node.level]
                module = ".".join((*package, module)) if module else ".".join(package)
            rows.append((node.lineno, module))
            if module in ("pops._ir",):
                rows.extend((node.lineno, "pops._ir." + alias.name) for alias in node.names
                            if alias.name in CARRIER_NAMES)
    return rows


def test_builder_and_lowering_do_not_read_the_array_type_before_the_retyping():
    offenders = []
    for package in GUARDED:
        for path in sorted((SOURCES / package).rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if relative in ALLOWED:
                continue
            for line, module in _imports(path):
                if module in CARRIERS or any(module.startswith(name + ".") for name in CARRIERS):
                    offenders.append("%s:%d imports %s" % (relative, line, module))
    assert not offenders, (
        "lot 1a of N10 keeps the one array type out of the builder and the lowering until the "
        "retyping (lot 1b, MG3); move the reader to the manifest reporter or wait for 1b:\n"
        + "\n".join(offenders))


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
        "lot 1a of N10 reports the layout records in the manifest only; no emitter, dispatch or "
        "acceptance reads them before the retyping (lot 1b, MG3):\n" + "\n".join(offenders))


def test_the_guard_sees_every_spelling_of_an_import(tmp_path):
    sources = tmp_path / "pops"
    fake = sources / "codegen" / "module.py"
    fake.parent.mkdir(parents=True)
    for text in ("import pops._ir.arrays\n", "from pops._ir import index_sets\n",
                 "from pops._ir.data_model import ir_type\n", "from .._ir import arrays\n"):
        fake.write_text(text, encoding="utf-8")
        modules = [module for _, module in _imports(fake, sources)]
        assert any(module in CARRIERS for module in modules), (text, modules)


def test_the_one_admitted_reader_only_reports():
    text = (ROOT / "python/pops/codegen/_compiled_artifact.py").read_text(encoding="utf-8")
    assert text.count("pops._ir.data_model") == 2  # the docstring and the one import
    assert text.count("_data_layout(") == 2       # its definition and its one call
    assert "data_layout" not in text[text.index("def _payload(self)"):][:4000].split(
        "def verify", 1)[0]
