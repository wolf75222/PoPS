"""Fail-closed contracts for every release-significant Python subprocess.

The guarded surface is not a hand-maintained sample: it is the union of files selected by the
pytest process collector and Python MPI entrypoints owned by ``test_manifest.toml``.  A literal
``sys.exit(0)`` is forbidden because it can turn an import, compiler, native-runtime, or MPI
capability regression into a false green.  Optional developer runs use the canonical helpers,
which emit ``POPS_SKIP:``; required CI lanes raise instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.python import conftest as process_runner
from tests.python.support import requirements


def _is_literal_success_exit(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr != "exit":
        return False
    if not isinstance(node.func.value, ast.Name) or node.func.value.id not in {"sys", "_sys"}:
        return False
    return not node.args or (isinstance(node.args[0], ast.Constant) and node.args[0].value == 0)


def _calls_requirement_helper(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id in {"require_native_or_skip", "require_mpi_or_skip"}
        for child in ast.walk(node)
    )


def _has_import_time_requirement_helper(tree: ast.Module) -> bool:
    """Independent oracle for scripts whose import may invoke a canonical skip policy."""
    for node in tree.body:
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom),
        ):
            continue
        if _calls_requirement_helper(node):
            return True
    return False


def test_guarded_surface_unifies_process_discovery_and_manifest_mpi_entrypoints() -> None:
    guarded = set(process_runner._guarded_process_test_paths())
    manifest_mpi = set(process_runner._manifest_mpi_entrypoints())
    discovered = {
        path.resolve()
        for path in (process_runner.REPO_ROOT / "tests/python").rglob("test_*.py")
        if process_runner._requires_process_collection(path)
    }
    assert guarded == discovered | manifest_mpi
    assert manifest_mpi <= guarded


def test_import_time_requirement_helpers_are_independently_process_isolated() -> None:
    expected: set[Path] = set()
    for path in (process_runner.REPO_ROOT / "tests/python").rglob("test_*.py"):
        if "architecture" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _has_import_time_requirement_helper(tree):
            expected.add(path.resolve())
    guarded = set(process_runner._guarded_process_test_paths())
    assert expected <= guarded, (
        "canonical import-time skip policies require process isolation: "
        f"{sorted(str(path.relative_to(process_runner.REPO_ROOT)) for path in expected - guarded)}"
    )


def test_guarded_process_scripts_have_no_literal_success_exit() -> None:
    violations: list[str] = []
    for path in process_runner._guarded_process_test_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if _is_literal_success_exit(node):
                violations.append(f"{path.relative_to(process_runner.REPO_ROOT)}:{node.lineno}")
    assert not violations, (
        "release-significant scripts must use require_native_or_skip/require_mpi_or_skip, not "
        f"literal sys.exit(0): {violations}"
    )


def test_guarded_custom_skip_helpers_delegate_to_canonical_policy() -> None:
    violations: list[str] = []
    for path in process_runner._guarded_process_test_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            is_skip_wrapper = node.name.startswith("_skip")
            is_requirement_wrapper = (
                node.name.startswith("_missing_") and "requirement" in node.name
            )
            if not (is_skip_wrapper or is_requirement_wrapper):
                continue
            if not _calls_requirement_helper(node):
                violations.append(f"{path.relative_to(process_runner.REPO_ROOT)}:{node.lineno}")
    assert not violations, f"free-form process skip helpers remain: {violations}"


def test_guarded_process_scripts_do_not_bypass_required_lane_policy() -> None:
    violations: list[str] = []
    for path in process_runner._guarded_process_test_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "skip_process_test"
            ):
                violations.append(f"{path.relative_to(process_runner.REPO_ROOT)}:{node.lineno}")
    assert not violations, (
        "skip_process_test bypasses POPS_REQUIRE_NATIVE_TESTS/POPS_REQUIRE_MPI_TESTS; "
        f"use the matching require-or-skip helper: {violations}"
    )


def test_guarded_process_scripts_do_not_print_free_form_skips() -> None:
    violations: list[str] = []
    for path in process_runner._guarded_process_test_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
                and node.args
            ):
                continue
            if "skip" in ast.unparse(node.args[0]).lower():
                violations.append(f"{path.relative_to(process_runner.REPO_ROOT)}:{node.lineno}")
    assert not violations, (
        "free-form skip output can leave a process green after dropping coverage; use the matching "
        f"require-or-skip helper: {violations}"
    )


def test_process_runner_routes_skips_by_manifest_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        process_runner,
        "require_native_or_skip",
        lambda reason, **_: calls.append(("native", reason)),
    )
    monkeypatch.setattr(
        process_runner,
        "require_mpi_or_skip",
        lambda reason, **_: calls.append(("mpi", reason)),
    )
    mpi_path = next(iter(process_runner._manifest_mpi_entrypoints()))
    native_path = next(
        path
        for path in process_runner._guarded_process_test_paths()
        if path not in process_runner._manifest_mpi_entrypoints()
    )
    process_runner._require_process_or_skip("mpi unavailable", mpi_path)
    process_runner._require_process_or_skip("native unavailable", native_path)
    assert calls == [
        ("mpi", "mpi unavailable"),
        ("native", "native unavailable"),
    ]


@pytest.mark.parametrize(
    ("environment", "helper", "prefix"),
    (
        ("POPS_REQUIRE_NATIVE_TESTS", requirements.require_native_or_skip, "required native"),
        ("POPS_REQUIRE_MPI_TESTS", requirements.require_mpi_or_skip, "required MPI"),
    ),
)
def test_required_process_helpers_raise_instead_of_skipping(
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
    helper,
    prefix: str,
) -> None:
    monkeypatch.setenv(environment, "1")
    with pytest.raises(RuntimeError, match=prefix):
        helper("missing acceptance capability")


@pytest.mark.parametrize(
    "helper",
    (requirements.require_native_or_skip, requirements.require_mpi_or_skip),
)
def test_optional_pytest_style_skip_uses_the_explicit_callback(
    monkeypatch: pytest.MonkeyPatch,
    helper,
) -> None:
    monkeypatch.delenv("POPS_REQUIRE_NATIVE_TESTS", raising=False)
    monkeypatch.delenv("POPS_REQUIRE_MPI_TESTS", raising=False)
    reasons: list[str] = []
    helper("optional developer capability", optional_skip=reasons.append)
    assert reasons == ["optional developer capability"]


@pytest.mark.parametrize(
    "helper",
    (requirements.require_native_or_skip, requirements.require_mpi_or_skip),
)
def test_optional_script_skip_uses_the_canonical_process_marker(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    helper,
) -> None:
    monkeypatch.delenv("POPS_REQUIRE_NATIVE_TESTS", raising=False)
    monkeypatch.delenv("POPS_REQUIRE_MPI_TESTS", raising=False)
    with pytest.raises(SystemExit) as stopped:
        helper("optional developer capability")
    assert stopped.value.code == 0
    assert capsys.readouterr().out == "POPS_SKIP: optional developer capability\n"
