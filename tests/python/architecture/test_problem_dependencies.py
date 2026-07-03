"""ADC-553: the pops.problem assembly package owns no runtime data and no forbidden edge.

:class:`pops.problem.Problem` is a pure authoring assembly (Spec 5 sec.6 / sec.15): it CONTAINS
descriptors, computes nothing, and never reaches into the runtime, the codegen driver or the native
extension. This source-only architecture check (no ``import pops``, no ``_pops``) enforces that fence
on every module under ``python/pops/problem/``:

* no ``import _pops`` / ``from pops._pops`` (the runtime is the only layer allowed to touch it) ;
* no ``import numpy`` (an assembly holds no arrays -- it is metadata only) ;
* no module-scope import of ``pops.runtime`` / ``pops.codegen`` (the assembly does not lower or run;
  ``pops.compile`` / ``pops.bind`` do, from the outside) ;
* each file stays within the 500-line budget (the split exists to kill the monolith).

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PROBLEM = REPO_ROOT / "python" / "pops" / "problem"

MAX_LINES = 500


def _problem_modules():
    return sorted(PROBLEM.rglob("*.py"))


def test_problem_package_exists():
    assert (PROBLEM / "__init__.py").exists(), (
        "python/pops/problem/__init__.py must exist: pops.problem is the ADC-553 assembly package "
        "(the split of the old flat case.py).")
    assert not (REPO_ROOT / "python" / "pops" / "case.py").exists(), (
        "python/pops/case.py must be gone: the assembly moved into the pops.problem package "
        "(ADC-553/ADC-526), with pops.Problem as the one public name.")


def test_no_runtime_codegen_or_pops_import():
    """No module-scope import of _pops / numpy / pops.runtime / pops.codegen in pops.problem."""
    forbidden = []
    for path in _problem_modules():
        tree = ast.parse(path.read_text(), str(path))
        for node in tree.body:  # module scope only (col_offset == 0 implied by tree.body)
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                if node.level == 0 and node.module:
                    names = [node.module]
            for name in names:
                if (name == "_pops" or name.startswith("_pops.")
                        or name == "numpy" or name.startswith("numpy.")
                        or name == "pops._pops" or name.startswith("pops._pops")
                        or name == "pops.runtime" or name.startswith("pops.runtime.")
                        or name == "pops.codegen" or name.startswith("pops.codegen.")):
                    forbidden.append("%s imports %s at module scope"
                                     % (path.relative_to(REPO_ROOT).as_posix(), name))
    assert not forbidden, (
        "pops.problem must stay runtime/codegen/_pops/numpy-free at module scope (ADC-553); "
        "found:\n  " + "\n  ".join(forbidden))


def test_problem_modules_within_line_budget():
    over = []
    for path in _problem_modules():
        lines = sum(1 for _ in path.open("rb"))
        if lines > MAX_LINES:
            over.append("%s: %d lines (limit %d)"
                        % (path.relative_to(REPO_ROOT).as_posix(), lines, MAX_LINES))
    assert not over, "pops.problem files exceed the line budget:\n  " + "\n  ".join(over)
