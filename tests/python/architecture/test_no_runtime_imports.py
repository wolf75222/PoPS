"""Spec 4 (36.2, adapted): the symbolic layers must not import the runtime at module scope.

The authoring/IR layers (``_ir``, ``model``, ``physics``, ``time``, ``lib``, ``mesh`` and
the Spec-5 central descriptor packages ``numerics`` / ``moments`` / ``diagnostics`` /
``params`` / ``output`` / ``external``) describe a problem; they must stay importable
without the compiled extension or the codegen/runtime machinery. A MODULE-SCOPE import of ``_pops``, ``pops.codegen`` or ``pops.runtime`` would
pull the heavy/native layer in at import time and break that guarantee.

Lazy (in-function / in-method) imports ARE allowed: a builder may import the runtime
when it actually compiles or installs. We therefore flag only top-level imports, detected
precisely with ``ast`` by requiring ``col_offset == 0`` (an import nested in a function or
class body is indented and is left alone).

``pops.output`` is stricter: it owns ConsumerGraph declarations and checkpoint/output providers,
so it may never import runtime at any scope. Runtime consumes output contracts, never conversely.

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POPS = REPO_ROOT / "python" / "pops"

# Symbolic (authoring/IR) layers that must not touch the runtime at module scope.
SYMBOLIC_LAYERS = ("_ir", "model", "physics", "time", "lib", "mesh",
                   "numerics", "moments", "diagnostics", "params", "output", "external",
                   "fields", "linalg")

# Forbidden module-scope import targets (and their dotted sub-modules).
FORBIDDEN_ROOTS = ("_pops", "pops.codegen", "pops.runtime", "pops._pops")


def _is_forbidden(modname):
    return any(modname == r or modname.startswith(r + ".") for r in FORBIDDEN_ROOTS)


def _module_scope_imports(tree):
    """Yield (lineno, target) for every module-scope (col_offset==0) import target."""
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if node.col_offset != 0:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            # Absolute imports only; relative (level>0) cannot target _pops/codegen/runtime.
            if node.level == 0 and node.module:
                yield node.lineno, node.module


def test_no_module_scope_runtime_imports():
    violations = []
    for layer in SYMBOLIC_LAYERS:
        layer_dir = POPS / layer
        if not layer_dir.is_dir():
            continue
        for path in sorted(layer_dir.rglob("*.py")):
            tree = ast.parse(path.read_text(), str(path))
            for lineno, target in _module_scope_imports(tree):
                if _is_forbidden(target):
                    rel = path.relative_to(REPO_ROOT)
                    violations.append("%s:%d imports %s at module scope" % (rel, lineno, target))
    assert not violations, (
        "symbolic layers must not import _pops/codegen/runtime at module scope "
        "(lazy in-function imports are allowed):\n  " + "\n  ".join(violations)
    )


def test_output_has_no_runtime_dependency_at_any_scope():
    violations = []
    for path in sorted((POPS / "output").rglob("*.py")):
        tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Import):
                targets.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    targets.append(node.module)
                elif node.level >= 2:
                    if node.module:
                        targets.append("pops." + node.module)
                    else:
                        targets.extend("pops." + alias.name for alias in node.names)
            for target in targets:
                if _is_forbidden(target):
                    rel = path.relative_to(REPO_ROOT)
                    violations.append("%s:%d imports %s" % (rel, node.lineno, target))
    assert not violations, (
        "pops.output must own consumer declarations without any runtime dependency:\n  "
        + "\n  ".join(violations)
    )
