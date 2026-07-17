from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HISTORY_NATIVE_TESTS = (
    "tests/python/integration/amr/test_amr_history_parity.py",
    "tests/python/integration/amr/test_amr_history_regrid.py",
    "tests/python/integration/io/test_amr_history_checkpoint.py",
    "tests/python/integration/io/test_amr_history_regrid_replay.py",
    "tests/python/integration/io/test_time_history_checkpoint.py",
    "tests/python/unit/time/test_time_history.py",
)
NATIVE_STAGE_CALLS = {"compile", "compile_problem", "install_program"}
PROCESS_RUNNERS = {"main", "run", "run_all", "_run", "_run_all"}


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _native_stages(statements: list[ast.stmt]) -> set[str]:
    return {
        name
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        if (name := _call_name(node)) in NATIVE_STAGE_CALLS
    }


def _catches_runtime_error(handler: ast.ExceptHandler) -> bool:
    caught = handler.type
    if caught is None:
        return True
    names = {node.id for node in ast.walk(caught) if isinstance(node, ast.Name)}
    return bool(names & {"RuntimeError", "Exception", "BaseException"})


def _downgrades_to_success(handler: ast.ExceptHandler) -> bool:
    if any(isinstance(node, ast.Raise) for node in ast.walk(handler)):
        return False
    if any(isinstance(node, ast.Return) for node in ast.walk(handler)):
        return True
    for node in ast.walk(handler):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name == "exit":
            return True
        if name == "print" and any(
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and "skip" in arg.value.lower()
            for arg in node.args
        ):
            return True
    return False


def _has_process_runner(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        for call in (child for child in ast.walk(node) if isinstance(child, ast.Call)):
            if _call_name(call) in PROCESS_RUNNERS:
                return True
    return False


def test_history_native_prerequisites_are_gated_explicitly() -> None:
    for relative in HISTORY_NATIVE_TESTS:
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "missing_native_compile_requirement" in source, relative
        assert "require_native_or_skip" in source, relative


def test_history_native_scripts_remain_process_isolated() -> None:
    for relative in HISTORY_NATIVE_TESTS:
        tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"), filename=relative)
        assert _has_process_runner(tree), relative


def test_history_native_stage_failures_cannot_be_downgraded_to_skips() -> None:
    violations: list[str] = []
    for relative in HISTORY_NATIVE_TESTS:
        tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            stages = _native_stages(node.body)
            if not stages:
                continue
            for handler in node.handlers:
                if _catches_runtime_error(handler) and _downgrades_to_success(handler):
                    violations.append(
                        "%s:%d catches %s failure then returns/skips"
                        % (relative, handler.lineno, "/".join(sorted(stages)))
                    )
    assert not violations, "\n".join(violations)
