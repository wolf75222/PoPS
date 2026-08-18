"""Force IF-01 MPI ranks and IF-04 relative-target restart."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from verification.pops_verify.case_authoring import load_sibling_module


def main() -> int:
    failed = False
    print("=== IF-01 ===")
    if01 = load_sibling_module(
        ROOT / "verification/cases/infrastructure/mpi_invariance/run.py"
    )
    try:
        field = if01.run_native(n_cells=32, t_end=0.25)
        print("IF-01 native field", getattr(field, "shape", None))
    except Exception as exc:
        print("IF-01", type(exc).__name__, exc)
        failed = True

    print("=== IF-04 ===")
    if04 = load_sibling_module(
        ROOT / "verification/cases/infrastructure/checkpoint_restart/run.py"
    )
    try:
        result = if04.run_native(n_cells=16, t=0.1)
        print("IF-04 linf", result["linf"], "l2", result["l2"])
        if float(result["linf"]) > 1.0e-3:
            failed = True
    except Exception as exc:
        print("IF-04", type(exc).__name__, exc)
        failed = True
    print("FAILED" if failed else "DONE")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
