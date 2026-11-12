"""The vertical slice of the exit of phase 1 (OD31; tranche/note_tranche_verticale.md, section 6).

The case file ``examples/surface/tutorials/scalar_advection/02_openmp_explicit_ssprk2.py`` is
master's tutorial 02 written in the surface, its upwind flux and its SSPRK2 written in the file.
Until every piece of ``tests/gates/vertical_slice.toml`` is lifted, the file stops on the first
one: this test runs it and checks the statement and the refusal that the registry records (no
skip, no xfail: CF14). The file passes the lint SU-T1 and imports only the surface and the library
(S1-a).
"""
from __future__ import annotations

import importlib.util
import runpy
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load("vertical_slice", ROOT / "scripts" / "vertical_slice.py")
lint = _load("surface_lint", ROOT / "scripts" / "surface_lint.py")
REGISTRY = gate.read(ROOT)
CASE = ROOT / REGISTRY["case"]
# S1-a: the surface and the library until MG10, then pops, pops.math and pops.discrete (T10)
ALLOWED = ("pops.surface", "pops.lib", "pops.math", "pops.discrete")


def _run(tmp_path, monkeypatch):
    """Run the case file in a fresh directory; the exception that stops it, or None."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(CASE)])
    try:
        runpy.run_path(str(CASE), run_name="__main__")
    except Exception as error:  # noqa: BLE001 - the refusal is the result
        return error
    return None


def test_the_case_file_passes_the_lint_and_imports_only_the_surface():
    findings = [str(finding) for finding in lint.lint_file(CASE, lint.load_words(),
                                                            name=REGISTRY["case"])]
    assert findings == []
    import ast

    for node in ast.walk(ast.parse(CASE.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        else:
            continue
        for module in modules:
            assert module == "pops" or module.startswith(ALLOWED), module


def test_the_registry_follows_the_case_file():
    assert gate.check(ROOT) == []


def test_the_case_file_stops_on_the_first_piece_of_the_registry(tmp_path, monkeypatch):
    error = _run(tmp_path, monkeypatch)
    pieces = REGISTRY["piece"]
    if not pieces:
        # the exit of phase 1: the file runs to its end and writes master's two records
        assert error is None, "".join(traceback.format_exception(error))
        assert (tmp_path / "results" / "02_openmp_explicit_ssprk2.npz").is_file()
        return
    first = pieces[0]
    assert error is not None, "the case file ran to its end: remove piece %s" % first["id"]
    refusal = "%s: %s" % (type(error).__name__, error)
    frames = [frame for frame in traceback.extract_tb(error.__traceback__)
              if Path(frame.filename).resolve() == CASE.resolve()]
    assert frames, refusal
    line = CASE.read_text(encoding="utf-8").splitlines()[frames[-1].lineno - 1].strip()
    assert (line, refusal[:len(first["refusal"])]) == (first["statement"], first["refusal"]), \
        "piece %s of %s; the file stops with %s" % (first["id"], gate.REGISTRY, refusal)
