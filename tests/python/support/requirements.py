from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable, Mapping
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

#: Line prefix a script-mode test prints to make the conftest subprocess runner
#: report SKIPPED instead of a silent pass. Kept in sync with
#: ``conftest.PROCESS_SKIP_MARKER``.
SKIP_MARKER = "POPS_SKIP:"

#: Gate-only collection asks the process-isolated pytest collector to expose exact ``test_*``
#: nodeids instead of its ordinary whole-file item.  The selected child receives one authenticated
#: function name through ``PROCESS_TEST_FILTER_ENV`` and must emit ``PROCESS_TEST_RESULT_PREFIX``
#: after that function returns.  Keeping both witnesses explicit makes an ignored selector fail
#: closed instead of silently executing a different proof in the same script.
EXACT_PROCESS_NODEIDS_ENV = "POPS_EXACT_PROCESS_NODEIDS"
PROCESS_TEST_FILTER_ENV = "POPS_PROCESS_TEST_FILTER"
PROCESS_TEST_RESULT_PREFIX = "POPS_PROCESS_TEST_OK:"


def run_process_test_cases(cases: Mapping[str, Callable[[], None]]) -> tuple[str, ...]:
    """Run either every script case or one exact gate-selected process case.

    Script-mode tests retain their ordinary all-cases entrypoint for developer execution.  Release
    gates set an exact filter through the process collector; unknown/missing case names are hard
    failures, and the success marker is emitted only after the selected callable returns.
    """

    requested = os.environ.get(PROCESS_TEST_FILTER_ENV)
    if requested is not None:
        case = cases.get(requested)
        if case is None:
            raise RuntimeError(
                "unknown exact process test %r; available cases are %s"
                % (requested, ", ".join(sorted(cases)))
            )
        case()
        print(PROCESS_TEST_RESULT_PREFIX + requested)
        return (requested,)

    executed: list[str] = []
    for name in sorted(cases):
        cases[name]()
        executed.append(name)
    return tuple(executed)


def repo_include() -> str:
    override = os.environ.get("POPS_INCLUDE")
    if override:
        return override
    return str(REPO_ROOT / "include")


def default_cxx() -> str | None:
    """Return a usable C++ driver, honoring POPS_TEST_CXX/CXX first."""
    return (
        os.environ.get("POPS_TEST_CXX")
        or os.environ.get("CXX")
        or shutil.which("c++")
        or shutil.which("g++")
        or shutil.which("clang++")
    )


def missing_compiler_requirement(include: str | os.PathLike[str] | None = None) -> str | None:
    """Return why a compiler-gated flow cannot run here, or None if it can."""
    if default_cxx() is None:
        return "no C++ compiler available"
    target = str(include) if include is not None else repo_include()
    if not Path(target).is_dir():
        return f"PoPS headers absent: {target}"
    return None


def skip_process_test(reason: str, *, code: int = 0) -> None:
    """Declare a script-mode test skipped and exit.

    Prints ``POPS_SKIP: <reason>`` so the conftest subprocess runner reports
    SKIPPED (not a silent pass) whatever the exit status, then exits ``code``.
    """
    print(f"{SKIP_MARKER} {reason}")
    sys.exit(code)


def native_tests_required() -> bool:
    """Return whether missing native prerequisites are release-gate failures."""
    return os.environ.get("POPS_REQUIRE_NATIVE_TESTS") == "1"


def mpi_tests_required() -> bool:
    """Return whether missing MPI prerequisites are release-gate failures."""
    return os.environ.get("POPS_REQUIRE_MPI_TESTS") == "1"


def require_native_or_skip(
    reason: str,
    *,
    optional_skip: Callable[[str], object] | None = None,
) -> None:
    """Fail a required native CI acceptance, otherwise report an explicit optional skip.

    Local source-only runs may legitimately lack a compiler, Kokkos, or an installed extension.
    The Serial native CI lane sets ``POPS_REQUIRE_NATIVE_TESTS=1`` because those prerequisites are
    part of that lane's contract; treating their disappearance (or an import/API regression) as a
    skip would silently remove release coverage. ``optional_skip`` lets pytest-native fixtures and
    tests use this same policy without relying on the process-test ``POPS_SKIP`` protocol.
    """
    if native_tests_required():
        raise RuntimeError(f"required native test unavailable: {reason}")
    if optional_skip is not None:
        optional_skip(reason)
        return
    skip_process_test(reason)


def require_mpi_or_skip(
    reason: str,
    *,
    optional_skip: Callable[[str], object] | None = None,
) -> None:
    """Fail a required MPI acceptance, otherwise report one canonical optional skip.

    The dedicated MPI lane sets ``POPS_REQUIRE_MPI_TESTS=1``.  Its manifest-owned entrypoints must
    therefore never turn a missing native MPI build, launcher, rank count, or parallel-HDF5
    capability into a successful process exit.  Developer runs retain an explicit ``POPS_SKIP:``
    result through the same process protocol as native-only tests.
    """
    if mpi_tests_required():
        raise RuntimeError(f"required MPI test unavailable: {reason}")
    if optional_skip is not None:
        optional_skip(reason)
        return
    skip_process_test(reason)


def kokkos_root() -> Path | None:
    for name in ("POPS_KOKKOS_ROOT", "Kokkos_ROOT", "KOKKOS_ROOT"):
        value = os.environ.get(name)
        if value:
            root = Path(value)
            if root.exists():
                return root
    return None


def missing_native_compile_requirement(
    include: str | os.PathLike[str], cxx: str | None,
) -> str | None:
    if not cxx:
        return "compilateur C++ absent"
    if not Path(include).is_dir():
        return f"en-tetes PoPS absents: {include}"
    if kokkos_root() is None:
        return "Kokkos introuvable (POPS_KOKKOS_ROOT/Kokkos_ROOT/KOKKOS_ROOT)"
    return None
