from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def repo_include() -> str:
    return str(REPO_ROOT / "include")


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
