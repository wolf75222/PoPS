"""TASK-003: source gates against false legacy lowering routes.

This test is intentionally source-only.  It does not import ``pops`` or ``_pops``.
The goal is not to ban every native ABI symbol name that still exists in C++;
it is to prevent the modern Python layers from reintroducing public legacy
front doors or routing through old DSL modules.
"""
import ast
import io
import pathlib
import tokenize


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


def _code_tokens(path):
    text = path.read_text(encoding="utf-8")
    stream = io.StringIO(text)
    for tok in tokenize.generate_tokens(stream.readline):
        if tok.type in (tokenize.NAME, tokenize.OP):
            yield tok.string, tok.start[0]


def _next_token(tokens, index, offset=1):
    j = index + offset
    return tokens[j][0] if j < len(tokens) else None


def _is_private_native_bridge(path, token):
    rel = str(path.relative_to(REPO_ROOT))
    if token == "install_program":
        return rel in {
            "python/pops/runtime/_system_unified_install.py",
            "python/pops/runtime/_amr_system_program.py",
        }
    if token == "solve_fields":
        return rel == "python/pops/runtime/_system_diagnostics.py"
    return False


def test_no_false_lowering_tokens():
    """TASK-003: executable production Python must not route through legacy lowering tokens.

    This scans the modern production layers named in TASK-003. It intentionally scans Python
    tokens instead of comments so old prose cannot hide a real route, while public docs/examples
    are guarded by ``test_examples_no_skip`` and the docs surface tests.
    """
    offenders = []
    for package in ("model", "time", "codegen", "runtime"):
        for path in _py_files(package):
            tokens = list(_code_tokens(path))
            for i, (token, line) in enumerate(tokens):
                rel = path.relative_to(REPO_ROOT)
                if token in {"to_dsl", "_module_to_model", "_rhs_legacy"}:
                    offenders.append("%s:%d contains executable token %s" % (rel, line, token))
                if token == "dsl" and _next_token(tokens, i) == "." and _next_token(tokens, i, 2) == "Model":
                    offenders.append("%s:%d contains executable token dsl.Model" % (rel, line))
                if (token == "solve_fields" and _next_token(tokens, i) == "("
                        and not _is_private_native_bridge(path, token)):
                    offenders.append("%s:%d contains executable token solve_fields(" % (rel, line))
                if token == "linear_source" and _next_token(tokens, i) == "(":
                    offenders.append("%s:%d contains executable token linear_source(" % (rel, line))
                if token == "P" and _next_token(tokens, i) == "." and _next_token(tokens, i, 2) == "rhs":
                    offenders.append("%s:%d contains executable token P.rhs" % (rel, line))
                if token == "install_program" and not _is_private_native_bridge(path, token):
                    offenders.append("%s:%d contains executable token install_program" % (rel, line))
                if token == "add_equation" and _next_token(tokens, i) == "(":
                    offenders.append("%s:%d contains executable token add_equation(" % (rel, line))
    assert not offenders, (
        "false-lowering / legacy route tokens are forbidden in production Python:\n%s"
        % "\n".join(offenders)
    )


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
