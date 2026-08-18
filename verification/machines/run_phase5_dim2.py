"""Dim2 native smoke for Phase 5 Cartesian cases (GE-03, RB-05, RB-07)."""
from __future__ import annotations

from pathlib import Path
import os
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from verification.pops_verify.case_authoring import load_sibling_module


def _load(rel: str):
    return load_sibling_module(ROOT / rel)


def _as_field(run, result) -> np.ndarray:
    if isinstance(result, dict):
        return run.pack_conserved(result)
    return np.ascontiguousarray(result, dtype=np.float64)


def _run(name: str, rel: str, n: int, t_end: float) -> bool:
    print(f"=== {name} ===")
    try:
        run = _load(rel)
        field = _as_field(run, run.run_native(n, t_end=t_end))
        finite = bool(np.isfinite(field).all())
        print(
            name,
            field.shape,
            float(np.min(field)),
            float(np.max(field)),
            finite,
        )
        expected = (field.shape[0], n, n)
        return finite and field.shape == expected and field.ndim == 3
    except Exception as exc:
        label = "UNAVAILABLE" if exc.__class__.__name__ == "NativeUnavailable" else "FAILED"
        print(name, label, type(exc).__name__, exc)
        return False


def main() -> int:
    dim = os.environ.get("POPS_NATIVE_DIM", "")
    print(f"host-dim={dim!r} suite=phase5-dim2")
    ok = True
    ok = _run("GE-03", "verification/cases/geometry/radial_acoustic/run.py", 16, 0.02) and ok
    ok = _run("RB-05", "verification/cases/robustness/sedov/run.py", 16, 0.01) and ok
    ok = _run("RB-07", "verification/cases/robustness/liska_implosion/run.py", 16, 0.05) and ok
    ok = _run("GE-06", "verification/cases/geometry/diocotron_amr/run.py", 16, 0.02) and ok
    print("DONE" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
