"""ADC-585: architecture gate quarantining the legacy ModelSpec POD.

ModelSpec (``pops._bootstrap.ModelSpec``) is the flat C++ POD the ``pops.Model(...)`` sugar builds
for the native ``add_block`` bridge; it is NOT the target model representation (a Module / Problem
plus a ModuleManifest).  ADC-585 quarantines it: it is off the ``pops`` root (``pops.ModelSpec``
no longer exists, it lives at ``pops.runtime.ModelSpec``), and the operator-first authoring surface
must not reference it.

These checks are source-only (they do not import ``pops`` / ``_pops``), so they run without a
built native extension.  If a legitimate ModelSpec hit ever appears in one of the scanned trees,
INVESTIGATE it (compile_drivers / codegen must not need the legacy POD) instead of allowlisting it
silently: the failure names the file and line so the reference can be removed at its source.
"""
import ast
import pathlib
import re


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POPS = REPO_ROOT / "python" / "pops"

# The operator-first / codegen trees that must be ModelSpec-free: the target model representation
# lowers through Module / Problem / ModuleManifest, never the legacy native-bridge POD.
QUARANTINED_ROOTS = (
    "case.py",
    "codegen",
    "model",
    "time",
    "solvers",
    "numerics",
    "fields",
    "mesh",
    "ir",
)

# Word-boundary match: only the exact ``ModelSpec`` token, not e.g. a longer identifier.
MODELSPEC_RE = re.compile(r"\bModelSpec\b")


def _read(path):
    return path.read_text(encoding="utf-8")


def _py_files(root):
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _quarantined_files():
    for entry in QUARANTINED_ROOTS:
        path = POPS / entry
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from _py_files(path)


def _rel(path):
    return path.relative_to(REPO_ROOT).as_posix()


def _modelspec_code_references(path):
    """Line numbers where ModelSpec is USED as a symbol (name / attribute / import), not in prose.

    A docstring or comment that mentions the token to EXPLAIN the quarantine is not a usage; the
    gate is that compile_drivers / codegen do not NEED the legacy POD. So the AST is walked for a
    real ``ModelSpec`` name, attribute access, or import alias -- string / comment content is
    ignored (an ``ast.Constant`` string is never inspected).
    """
    tree = ast.parse(_read(path), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "ModelSpec":
            hits.append(node.lineno)
        elif isinstance(node, ast.Attribute) and node.attr == "ModelSpec":
            hits.append(node.lineno)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "ModelSpec" or alias.asname == "ModelSpec":
                    hits.append(node.lineno)
    return sorted(set(hits))


def test_operator_first_and_codegen_trees_do_not_use_modelspec():
    violations = []
    for path in _quarantined_files():
        for lineno in _modelspec_code_references(path):
            violations.append("%s:%d" % (_rel(path), lineno))

    assert not violations, (
        "the operator-first / codegen surface must not USE the legacy ModelSpec POD (ADC-585); the "
        "target representation is a Module / Problem + ModuleManifest. compile_drivers / codegen "
        "must not need it -- investigate and remove each reference at its source (never allowlist "
        "it):\n  " + "\n  ".join(violations)
    )


def test_pops_root_does_not_import_modelspec():
    text = _read(POPS / "__init__.py")
    # A bare comment mentioning ModelSpec is fine; an IMPORT of the name is what ADC-585 forbids.
    import_lines = [line for line in text.splitlines()
                    if MODELSPEC_RE.search(line) and (
                        "import" in line and not line.lstrip().startswith("#"))]
    assert not import_lines, (
        "pops/__init__.py must not import ModelSpec (ADC-585 moved it to pops.runtime); found:\n  "
        + "\n  ".join(import_lines))


def test_pops_runtime_reexports_modelspec():
    text = _read(POPS / "runtime" / "__init__.py")
    imports_it = any(
        MODELSPEC_RE.search(line) and "import" in line and not line.lstrip().startswith("#")
        for line in text.splitlines())
    assert imports_it, (
        "pops/runtime/__init__.py must re-export ModelSpec (the legacy native-bridge POD's "
        "quarantined home, ADC-585)")


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
