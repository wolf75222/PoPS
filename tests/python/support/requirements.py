from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

#: Line prefix a script-mode test prints to make the conftest subprocess runner
#: report SKIPPED instead of a silent pass. Kept in sync with
#: ``conftest.PROCESS_SKIP_MARKER``.
SKIP_MARKER = "POPS_SKIP:"


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


def kokkos_root() -> Path | None:
    for name in ("POPS_KOKKOS_ROOT", "Kokkos_ROOT", "KOKKOS_ROOT"):
        value = os.environ.get(name)
        if value:
            root = Path(value)
            if root.exists():
                return root
    return None


def missing_aot_requirement(include: str | os.PathLike[str], cxx: str | None) -> str | None:
    if not cxx:
        return "compilateur C++ absent"
    if not Path(include).is_dir():
        return f"en-tetes PoPS absents: {include}"
    if kokkos_root() is None:
        return "Kokkos introuvable (POPS_KOKKOS_ROOT/Kokkos_ROOT/KOKKOS_ROOT)"
    return None
