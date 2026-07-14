"""Final ``pops.lib.time`` factories never select operators by free strings.

The factories accept block-qualified state handles and typed operator handles.  This source-only
gate scans the tracked canonical factory modules and rejects literal strings passed to the internal
operator-call, handle-authentication, or apply seams.  Runtime signature and alias rejection live in
``test_program_factories.py`` and ``test_final_public_api.py``.
"""
from __future__ import annotations

import ast
import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
LIB_TIME = REPO_ROOT / "python" / "pops" / "lib" / "time"
CANONICAL_MODULES = (
    "_factory.py",
    "_helpers.py",
    "euler.py",
    "imex.py",
    "multistep.py",
    "predictor_corrector.py",
    "rk.py",
    "ssprk.py",
    "strang.py",
)


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_string(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _violations(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name in {"_call", "operator_handle"} and node.args and _is_string(node.args[0]):
            violations.append(f"{path.name}:{node.lineno} {name}() takes a string selector")
        if name == "apply":
            if node.args and _is_string(node.args[0]):
                violations.append(f"{path.name}:{node.lineno} apply() takes a string selector")
            for keyword in node.keywords:
                if keyword.arg == "operator" and _is_string(keyword.value):
                    violations.append(
                        f"{path.name}:{node.lineno} apply(operator=) takes a string selector")
    return violations


def test_no_free_string_operator_selector_in_final_lib_time() -> None:
    violations = []
    for module in CANONICAL_MODULES:
        path = LIB_TIME / module
        assert path.is_file(), f"missing canonical pops.lib.time module {module!r}"
        violations.extend(_violations(path))
    assert not violations, (
        "pops.lib.time factories must authenticate typed operator handles:\n  "
        + "\n  ".join(violations)
    )
