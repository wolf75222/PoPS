"""Phase 6 native smoke: IF-02 threads, IF-03 Serial/OpenMP, IF-08 doctor+dim."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors


def _tr01():
    return load_sibling_module(
        ROOT / "verification/cases/transport/advection_sine/run.py"
    )


def _exact():
    return load_sibling_module(
        ROOT / "verification/cases/transport/advection_sine/exact.py"
    )


def _run_tr01(n: int, t_end: float, threads: int) -> np.ndarray:
    os.environ["OMP_NUM_THREADS"] = str(int(threads))
    os.environ["POPS_NATIVE_DIM"] = "1"
    field = np.asarray(_tr01().run_native(n, t_end=t_end), dtype=np.float64)
    return np.reshape(field, (-1,))


def main() -> int:
    n = 32
    t_end = 0.25
    exact = _exact()
    centers, volumes = exact.uniform_cell_centers(n)
    oracle = exact.exact_sine(centers, t_end)

    print("=== IF-08 doctor ===")
    import pops

    try:
        doctor = pops.doctor
    except AttributeError:
        from pops.runtime.doctor import doctor
    try:
        report = doctor(verbose=False)
        print("doctor_keys", sorted(report)[:12], "n=", len(report))
    except Exception as exc:
        print("doctor UNAVAILABLE", type(exc).__name__, exc)

    print("=== IF-02 / IF-03 threads ===")
    fields = {}
    for threads in (1, 8):
        field = _run_tr01(n, t_end, threads)
        err = reference_errors(field, oracle, volumes)
        fields[threads] = field
        print(f"OMP={threads}", field.shape, float(err.l2), float(err.linf),
              bool(np.isfinite(field).all()))
    delta = reference_errors(fields[1], fields[8], volumes)
    print("IF-02/03 Linf(1,8)", float(delta.linf), "L2", float(delta.l2))

    print("=== IF-08 dim mismatch ===")
    os.environ["POPS_NATIVE_DIM"] = "1"
    ge03 = load_sibling_module(
        ROOT / "verification/cases/geometry/radial_acoustic/run.py"
    )
    try:
        ge03.run_native(8, t_end=0.01)
        print("GE-03 under Dim1 UNEXPECTED_OK")
    except Exception as exc:
        print("GE-03 under Dim1", type(exc).__name__, str(exc)[:160])
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
