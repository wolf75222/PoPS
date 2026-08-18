"""Phase 7 one-node CPU smoke: time public PF wrappers."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from verification.pops_verify.case_authoring import load_sibling_module


CASES = (
    ("PF-01", "verification/cases/performance/multifab_arith/run.py"),
    ("PF-02", "verification/cases/performance/scalar_mg/run.py"),
    ("PF-03", "verification/cases/performance/advection_rhs/run.py"),
    ("PF-04", "verification/cases/performance/euler_step/run.py"),
    ("PF-05", "verification/cases/performance/composite_poisson/run.py"),
    ("PF-06", "verification/cases/performance/ep_step/run.py"),
    ("PF-07", "verification/cases/performance/regrid_cluster/run.py"),
    ("PF-08", "verification/cases/performance/reflux_sync/run.py"),
    ("PF-11", "verification/cases/performance/amr_e2e/run.py"),
)


def main() -> int:
    print("suite=phase7-one-node")
    failed = False
    for name, rel in CASES:
        print(f"=== {name} ===")
        run = load_sibling_module(ROOT / rel)
        try:
            result = run.run_native()
            print(
                name,
                "elapsed_s",
                result.get("elapsed_s"),
                "cells_per_second",
                result.get("cells_per_second"),
            )
        except Exception as exc:
            label = "GATED" if exc.__class__.__name__ == "NativeUnavailable" else "FAILED"
            print(name, label, type(exc).__name__, exc)
            if label == "FAILED":
                failed = True
    print("FAILED" if failed else "DONE")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
