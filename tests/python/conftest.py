from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from functools import cache
from pathlib import Path

import pytest

from tests.python.support.requirements import (
    native_tests_required,
    require_native_or_skip,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PYTHON = REPO_ROOT / "python"
PROCESS_TEST_TIMEOUT = int(os.environ.get("POPS_PY_PROCESS_TIMEOUT", "300"))


class PythonProcessFailure(Exception):
    def __init__(self, path: Path, returncode: int, output: str) -> None:
        self.path = path
        self.returncode = returncode
        self.output = output
        super().__init__(f"{path} exited with status {returncode}")


class PythonProcessFile(pytest.File):
    def collect(self) -> Iterator[pytest.Item]:
        yield PythonProcessItem.from_parent(self, name=self.path.name)


class PythonProcessItem(pytest.Item):
    def runtest(self) -> None:
        env = os.environ.copy()
        env["POPS_PYTEST_PROCESS"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = _process_pythonpath(env.get("PYTHONPATH"))
        result = subprocess.run(
            [sys.executable, str(self.path)],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=_process_timeout(Path(str(self.path))),
        )
        skip_reason = _process_skip_reason(result.stdout)
        if skip_reason:
            require_native_or_skip(skip_reason, optional_skip=pytest.skip)
        if result.returncode == 0:
            legacy_missing = _missing_process_requirement(result.stdout)
            if legacy_missing:
                require_native_or_skip(legacy_missing, optional_skip=pytest.skip)
        if result.returncode != 0:
            missing = _missing_process_requirement_for_environment(result.stdout, env)
            if missing:
                require_native_or_skip(missing, optional_skip=pytest.skip)
            raise PythonProcessFailure(Path(str(self.path)), result.returncode, result.stdout)

    def repr_failure(self, excinfo: pytest.ExceptionInfo[BaseException]) -> str:
        if isinstance(excinfo.value, PythonProcessFailure):
            output = excinfo.value.output.rstrip()
            if len(output) > 16000:
                output = output[-16000:]
            return (
                f"{excinfo.value.path} exited with status {excinfo.value.returncode}\n"
                f"{output}"
            )
        return super().repr_failure(excinfo)

    def reportinfo(self) -> tuple[Path, int, str]:
        return self.path, 0, f"process-isolated Python test: {self.name}"


def pytest_configure(config: pytest.Config) -> None:
    markers = {
        "unit": "small Python-only or pure API test",
        "integration": "requires _pops or multiple subsystems",
        "regression": "historical bug or numerical oracle",
        "architecture": "source-only architecture contract",
        "bindings": "requires the compiled _pops extension",
        "kokkos": "requires a visible Kokkos install",
        "native_loader": "compiles or loads a native shared object",
        "compiler": "requires a C++ compiler",
        "mpi": "requires MPI runtime support",
        "hdf5": "requires h5py/HDF5 support",
        "slow": "too slow for the default PR lane",
        "serial_resource": "uses process-global state or a shared external resource",
        "process_isolated": "runs the whole file in a subprocess instead of importing it as a pytest module",
    }
    for marker, description in markers.items():
        config.addinivalue_line("markers", f"{marker}: {description}")


def pytest_pycollect_makemodule(module_path: Path, parent: pytest.Collector) -> pytest.File | None:
    if module_path.suffix != ".py" or not module_path.name.startswith("test_"):
        return None
    if _requires_process_collection(module_path):
        return PythonProcessFile.from_parent(parent, path=module_path)
    return None


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def source_python_path() -> Path:
    return SOURCE_PYTHON


@pytest.fixture(scope="session")
def kokkos_root() -> Path:
    for name in ("POPS_KOKKOS_ROOT", "Kokkos_ROOT", "KOKKOS_ROOT"):
        value = os.environ.get(name)
        if value:
            root = Path(value)
            if root.exists():
                return root
    require_native_or_skip(
        "Kokkos root is not configured",
        optional_skip=pytest.skip,
    )
    raise AssertionError("pytest.skip unexpectedly returned")


@pytest.fixture(scope="session")
def native_cxx() -> str:
    candidates = [
        os.environ.get("POPS_TEST_CXX"),
        os.environ.get("CXX"),
        shutil.which("c++"),
        shutil.which("g++"),
        shutil.which("clang++"),
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    require_native_or_skip(
        "no C++ compiler available",
        optional_skip=pytest.skip,
    )
    raise AssertionError("pytest.skip unexpectedly returned")


@pytest.fixture(scope="session")
def mpi_available() -> bool:
    mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")
    if not mpiexec:
        return False
    try:
        result = subprocess.run(
            [mpiexec, "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


@pytest.fixture
def isolated_native_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "pops-native-cache"
    cache.mkdir()
    monkeypatch.setenv("POPS_CACHE_DIR", str(cache))
    monkeypatch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
    return cache


@pytest.fixture
def deterministic_seed(monkeypatch: pytest.MonkeyPatch) -> int:
    seed = 12345
    monkeypatch.setenv("PYTHONHASHSEED", str(seed))
    return seed


@pytest.fixture(autouse=True)
def stable_test_cwd(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.chdir(REPO_ROOT)
    yield


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    compiler_missing = _compiler_gate_reason()
    for item in items:
        path = Path(str(item.fspath))
        parts = set(path.parts)
        if "architecture" in parts:
            item.add_marker(pytest.mark.architecture)
        if "unit" in parts:
            item.add_marker(pytest.mark.unit)
        if "integration" in parts:
            item.add_marker(pytest.mark.integration)
        if "native_loader" in parts:
            item.add_marker(pytest.mark.native_loader)
            item.add_marker(pytest.mark.compiler)
        if "bindings" in parts:
            item.add_marker(pytest.mark.bindings)
        if "mpi" in parts:
            item.add_marker(pytest.mark.mpi)
        if "io" in parts:
            item.add_marker(pytest.mark.hdf5)
        if isinstance(item, PythonProcessItem):
            item.add_marker(pytest.mark.process_isolated)
        # A pytest-native test tagged ``compiler`` cannot run without a C++
        # toolchain and the header tree; skip it explicitly rather than let it
        # fail at import. Process-isolated tests gate themselves and report the
        # skip through the POPS_SKIP marker, so they are left alone here.
        if compiler_missing and not isinstance(item, PythonProcessItem):
            if any(m.name == "compiler" for m in item.iter_markers()):
                require_native_or_skip(
                    compiler_missing,
                    optional_skip=lambda reason, target=item: target.add_marker(
                        pytest.mark.skip(reason=reason)
                    ),
                )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
    call: pytest.CallInfo[None],
) -> Iterator[None]:
    """Make every native-gated skip fail closed in required CI lanes.

    This is the final safety net for a legacy direct ``pytest.skip`` or a third-party fixture skip.
    The canonical requirement helper remains the preferred call site, but release coverage cannot
    become green merely because one native test missed that migration.
    """
    outcome = yield
    report = outcome.get_result()
    if (
        report.skipped
        and native_tests_required()
        and _is_native_requirement_skip(report.longrepr)
    ):
        report.outcome = "failed"
        report.longrepr = (
            "POPS_REQUIRE_NATIVE_TESTS=1 forbids skips for native-gated tests: "
            f"{item.nodeid} ({report.longrepr})"
        )


def _is_native_requirement_skip(longrepr: object) -> bool:
    """Recognize only compiler/Kokkos/native-build skip diagnostics.

    Marker membership is deliberately insufficient: a test can combine a native leg with an
    optional MPI/HDF5 leg. Conversely, legacy mixed tests may omit the compiler marker entirely.
    The skip reason is the authority for whether ``POPS_REQUIRE_NATIVE_TESTS`` applies.
    """
    reason = str(longrepr).lower()
    explicit = (
        "c++ compiler",
        "compiler available",
        "compilateur c++",
        "kokkos",
        "pops headers",
        "pops header",
        "en-tetes pops",
        "native extension",
        "_pops module",
        "stale build",
    )
    return any(token in reason for token in explicit) or (
        ".so" in reason and ("build" in reason or "compile" in reason)
    )


@cache
def _process_timeout(path: Path) -> int:
    """Timeout for one process-isolated test file, in seconds.

    A file whose workload legitimately exceeds the global budget (e.g. several
    DSL native compiles in a row) declares a module-level
    ``POPS_PROCESS_TIMEOUT = <seconds>`` constant; the larger of the two
    budgets wins so a global raise via POPS_PY_PROCESS_TIMEOUT still applies.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return PROCESS_TEST_TIMEOUT
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "POPS_PROCESS_TIMEOUT"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, int)
            ):
                return max(PROCESS_TEST_TIMEOUT, node.value.value)
    return PROCESS_TEST_TIMEOUT


@cache
def _process_test_dirs() -> tuple[str, ...]:
    dirs = {str(path.parent) for path in (REPO_ROOT / "tests" / "python").rglob("test_*.py")}
    return tuple(sorted(dirs))


def _process_pythonpath(existing: str | None) -> str:
    """Build a subprocess path without masking an installed native package.

    A source checkout contains ``pops/__init__.py`` but normally not ``pops._pops``. Putting that
    directory ahead of site-packages makes process-isolated integration tests import a package that
    can never load its native extension, even immediately after ``scripts/build_python.sh`` installed
    a coherent wheel. Use the source package only when it carries a compatible extension itself;
    otherwise exercise the freshly installed wheel while keeping the repository/test helpers visible.
    """
    source_usable = _source_python_has_native_extension()
    entries: list[str] = []
    if existing:
        entries.extend(
            part for part in existing.split(os.pathsep)
            if part and (source_usable or Path(part).resolve() != SOURCE_PYTHON.resolve())
        )
    entries.append(str(REPO_ROOT))
    if source_usable:
        entries.append(str(SOURCE_PYTHON))
    entries.extend(_process_test_dirs())
    deduped: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if entry not in seen:
            seen.add(entry)
            deduped.append(entry)
    return os.pathsep.join(deduped)


@cache
def _source_python_has_native_extension() -> bool:
    package = SOURCE_PYTHON / "pops"
    suffixes = (".so", ".dylib", ".pyd")
    return any(
        path.name.startswith("_pops.") and path.suffix in suffixes
        for path in package.iterdir()
    )


@cache
def _compiler_gate_reason() -> str | None:
    """Return why a compiler-gated test cannot run here, or None if it can.

    The gate mirrors the script-mode guard the process tests use: a usable C++
    driver plus the in-repo header tree. Cached because it never changes within
    a run.
    """
    cxx = (
        os.environ.get("POPS_TEST_CXX")
        or os.environ.get("CXX")
        or shutil.which("c++")
        or shutil.which("g++")
        or shutil.which("clang++")
    )
    if not cxx:
        return "no C++ compiler available"
    include = os.environ.get("POPS_INCLUDE") or str(REPO_ROOT / "include")
    if not Path(include).is_dir():
        return f"PoPS headers absent: {include}"
    return None


PROCESS_SKIP_MARKER = "POPS_SKIP:"


def _process_skip_reason(output: str) -> str | None:
    """Return the reason a subprocess test declared itself skipped, if any.

    Script-style tests that gate on a missing requirement (a C++ compiler, the
    header tree, ...) exit 0 after printing a line, which pytest would otherwise
    record as a silent pass. When such a test prints ``POPS_SKIP: <reason>`` the
    subprocess runner reports SKIPPED instead, whatever the exit status.
    """
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith(PROCESS_SKIP_MARKER):
            return stripped[len(PROCESS_SKIP_MARKER):].strip() or "requirement not met"
    return None


def _missing_process_requirement(output: str) -> str | None:
    if "PoPS is Kokkos-only" in output and "POPS_KOKKOS_ROOT" in output:
        return "native compile requires POPS_KOKKOS_ROOT/Kokkos_ROOT"
    if "PoPS is Kokkos-only" in output and "Kokkos_Core.hpp" in output:
        return "native compile requires POPS_KOKKOS_ROOT/Kokkos_ROOT"
    if "Kokkos introuvable" in output:
        return "native compile requires POPS_KOKKOS_ROOT/Kokkos_ROOT"
    if "DO NOT MATCH those with which the _pops module was built" in output:
        return "headers do not match the built _pops module (stale build/overlay)"
    return None


def _missing_process_requirement_for_environment(
    output: str, environment: dict[str, str],
) -> str | None:
    """Keep legacy local skips out of a native-required release lane.

    ``require_native_or_skip`` raises on missing prerequisites when
    ``POPS_REQUIRE_NATIVE_TESTS=1``. The subprocess parent must preserve that failure instead of
    reclassifying its diagnostic text through the older heuristic fallback above.
    """
    if environment.get("POPS_REQUIRE_NATIVE_TESTS") == "1":
        return None
    return _missing_process_requirement(output)


@cache
def _requires_process_collection(path: Path) -> bool:
    if "architecture" in path.parts:
        return False
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return False
    if _has_test_defs(tree) and _has_pytest_main_entrypoint(tree) and "pytest = _SkipModule()" not in text:
        return False
    return (
        "pytest = _SkipModule()" in text
        or _has_import_time_sys_exit(tree)
        or _has_custom_main_entrypoint(tree)
    )


def _has_import_time_sys_exit(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.If) and _is_main_guard(node.test):
            continue
        if _contains_sys_exit(node):
            return True
    return False


def _has_custom_main_entrypoint(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.If) or not _is_main_guard(node.test):
            continue
        body = ast.unparse(node) if hasattr(ast, "unparse") else ""
        if "pytest.main" in body:
            return False
        return any(_is_runner_call(call) for call in ast.walk(node) if isinstance(call, ast.Call))
    return False


def _has_pytest_main_entrypoint(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.If) and _is_main_guard(node.test):
            body = ast.unparse(node) if hasattr(ast, "unparse") else ""
            return "pytest.main" in body
    return False


def _has_test_defs(tree: ast.Module) -> bool:
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name.startswith("test")
        for node in tree.body
    )


def _is_runner_call(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id in {"main", "run", "run_all", "_run", "_run_all"}
    if isinstance(func, ast.Attribute):
        return func.attr in {"main", "run", "run_all", "_run", "_run_all"}
    return False


def _contains_sys_exit(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Attribute) and func.attr == "exit":
            if isinstance(func.value, ast.Name) and func.value.id in {"sys", "_sys"}:
                return True
    return False


def _is_main_guard(test: ast.AST) -> bool:
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )
