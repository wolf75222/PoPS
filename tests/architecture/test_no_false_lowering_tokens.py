"""TASK-003: source gates against false legacy lowering routes.

This test is intentionally source-only.  It does not import ``pops`` or ``_pops``.
The goal is not to ban every native ABI symbol name that still exists in C++;
it is to prevent the modern Python layers from reintroducing public legacy
front doors or routing through old DSL modules.
"""
import ast
import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
POPS = REPO_ROOT / "python" / "pops"


def _py_files(*parts):
    root = POPS.joinpath(*parts)
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _defs(path):
    for node in ast.walk(_tree(path)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node


def test_program_has_no_public_legacy_methods():
    """The Program API is operator-first: no public rhs/source/solve_fields selectors."""
    forbidden = {"rhs", "source", "linear_source", "solve_fields", "solve_linear_old"}
    offenders = []
    for path in _py_files("time"):
        for node in _defs(path):
            if isinstance(node, ast.FunctionDef) and node.name in forbidden:
                offenders.append("%s:%d def %s" % (path.relative_to(REPO_ROOT), node.lineno, node.name))
    assert not offenders, (
        "Program/time public legacy methods must not exist; use T.call(operator_handle, ...) "
        "or typed library macros instead:\n%s" % "\n".join(offenders)
    )


def test_runtime_has_no_public_legacy_wiring_methods():
    """Runtime wiring is sim.install(compiled, ...), not public add_equation/install_program."""
    forbidden = {"add_equation", "add_block", "install_program", "set_poisson", "set_param"}
    offenders = []
    for path in _py_files("runtime"):
        for node in _defs(path):
            if isinstance(node, ast.FunctionDef) and node.name in forbidden:
                offenders.append("%s:%d def %s" % (path.relative_to(REPO_ROOT), node.lineno, node.name))
    assert not offenders, (
        "runtime legacy wiring methods must not be public Python methods; keep native ABI calls "
        "behind private seams such as _install_problem_so:\n%s" % "\n".join(offenders)
    )


def test_modern_layers_do_not_import_old_dsl_modules():
    """model/time/codegen/runtime must not import the old public DSL/orchestration modules."""
    forbidden_imports = {
        "pops.dsl",
        "pops.integrate",
        "pops.case",
        "pops.problem",
        "pops.codegen.orchestration",
    }
    offenders = []
    for package in ("model", "time", "codegen", "runtime"):
        for path in _py_files(package):
            tree = _tree(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in forbidden_imports:
                            offenders.append("%s:%d import %s" % (
                                path.relative_to(REPO_ROOT), node.lineno, alias.name))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if node.level:
                        continue
                    if module in forbidden_imports:
                        offenders.append("%s:%d from %s import ..." % (
                            path.relative_to(REPO_ROOT), node.lineno, module))
    assert not offenders, (
        "modern layers must not import old DSL/orchestration modules:\n%s" % "\n".join(offenders)
    )


def test_no_old_module_lowering_escape_hatches():
    """A Module is already canonical IR; no to_dsl/_module_to_model escape hatch."""
    forbidden_names = {"to_dsl", "_module_to_model"}
    offenders = []
    for package in ("model", "codegen"):
        for path in _py_files(package):
            for node in _defs(path):
                if node.name in forbidden_names:
                    offenders.append("%s:%d %s" % (
                        path.relative_to(REPO_ROOT), node.lineno, node.name))
    assert not offenders, (
        "do not revive Module -> old DSL lowering escape hatches:\n%s" % "\n".join(offenders)
    )


def test_no_legacy_private_program_helper_names():
    """The historical private helpers must not become stable architecture names again."""
    forbidden = ("_rhs_legacy", "_legacy_rhs", "_legacy_solve_fields", "_solve_fields")
    offenders = []
    for path in _py_files("time"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append("%s contains %s" % (path.relative_to(REPO_ROOT), token))
    assert not offenders, (
        "legacy Program helper names must stay deleted:\n%s" % "\n".join(offenders)
    )
