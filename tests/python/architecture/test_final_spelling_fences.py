"""ADC-625 fences: the ONE final spelling everywhere, no migration crutch can return.

Source-only guards (no ``import pops`` / no ``_pops``) that pin the Phase-5 final surface so a
regression fails loud:

* ``pops.Case`` / ``pops.case.`` is gone from the code and docs (only the intentional rename-refusal
  error message and the historical changelog / rationale note may name it);
* the public ``Program.linear_source`` / ``Program.apply`` refuse a free string (the string
  selector survives only in the ``_linear_source`` / ``_apply`` internal seams).

The test reads the source tree only.
"""
import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POPS = REPO_ROOT / "python" / "pops"


def _py_files():
    return sorted(POPS.rglob("*.py"))


# Lines that are ALLOWED to mention pops.Case: the intentional rename-refusal message and the
# historical rationale note. Keyed by (relative path, substring that must be on the line).
_CASE_ALLOWED = {
    "__init__.py": "was renamed to pops.Problem",       # the AttributeError refusal
    "problem/registries.py": "The old",                 # historical "the old pops.case.Case" note
}


def test_no_pops_case_in_the_code():
    offenders = []
    for path in _py_files():
        rel = path.relative_to(POPS).as_posix()
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if "pops.Case" in line or "pops.case." in line:
                allowed = _CASE_ALLOWED.get(rel)
                if allowed and allowed in line:
                    continue
                offenders.append("python/pops/%s:%d: %s" % (rel, lineno, line.strip()))
    assert not offenders, (
        "pops.Case / pops.case. is the removed spelling (ADC-553); use pops.Problem:\n  "
        + "\n  ".join(offenders))


def _program_method(class_name, method_name):
    """Return the source of ``method_name`` on ``class_name`` in the pops.time program mixins."""
    for path in (POPS / "time" / "program_core.py", POPS / "time" / "program_local.py"):
        src = path.read_text()
        tree = ast.parse(src, str(path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method_name:
                        return ast.get_source_segment(src, item) or ""
    return None


def test_public_linear_source_and_apply_refuse_a_free_string():
    # The public route must raise a TypeError on a bare string; the string selector lives only in
    # the _-prefixed internal seams (_linear_source / _apply).
    for method in ("linear_source", "apply"):
        body = _program_method("_ProgramCore", method)
        assert body is not None, "pops.time _ProgramCore must define %s" % method
        assert "isinstance(operator, str)" in body and "TypeError" in body, (
            "public Program.%s must refuse a free string with a TypeError (ADC-625)" % method)
    # The internal seams exist (on _ProgramLocal) and carry the bare-name selector.
    for seam in ("_linear_source", "_apply"):
        assert _program_method("_ProgramLocal", seam) is not None, (
            "pops.time _ProgramLocal must define the internal seam %s (ADC-625)" % seam)
