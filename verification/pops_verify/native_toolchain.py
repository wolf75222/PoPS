"""Production compiler/native probes. Does not import tests.python.support."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_include() -> str:
    override = os.environ.get("POPS_INCLUDE")
    if override:
        return override
    return str(REPO_ROOT / "include")


def default_cxx() -> str | None:
    return (
        os.environ.get("POPS_TEST_CXX")
        or os.environ.get("CXX")
        or shutil.which("c++")
        or shutil.which("g++")
        or shutil.which("clang++")
    )


def kokkos_root() -> Path | None:
    for name in ("POPS_KOKKOS_ROOT", "Kokkos_ROOT", "KOKKOS_ROOT"):
        value = os.environ.get(name)
        if value:
            root = Path(value)
            if root.exists():
                return root
    return None


def missing_compiler_requirement(include: str | os.PathLike[str] | None = None) -> str | None:
    if default_cxx() is None:
        return "no C++ compiler available"
    target = str(include) if include is not None else repo_include()
    if not Path(target).is_dir():
        return f"PoPS headers absent: {target}"
    return None


def missing_native_compile_requirement(
    include: str | os.PathLike[str] | None = None,
    cxx: str | None = None,
) -> str | None:
    compiler = cxx if cxx is not None else default_cxx()
    headers = str(include) if include is not None else repo_include()
    if not compiler:
        return "no C++ compiler available"
    if not Path(headers).is_dir():
        return f"PoPS headers absent: {headers}"
    if kokkos_root() is None:
        return "Kokkos introuvable (POPS_KOKKOS_ROOT/Kokkos_ROOT/KOKKOS_ROOT)"
    return None


def native_unavailable_reason() -> str | None:
    missing = missing_compiler_requirement()
    if missing:
        return missing
    return missing_native_compile_requirement()
