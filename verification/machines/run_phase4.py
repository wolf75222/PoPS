"""Phase 4 native campaign: eigenmodes, Strang, collisions, AP, multirate."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors


def load(rel: str):
    return load_sibling_module(ROOT / rel)


def _report(name, field, extra=""):
    arr = np.asarray(field, dtype=np.float64)
    print(
        name,
        arr.shape,
        "min",
        float(np.min(arr)),
        "max",
        float(np.max(arr)),
        "finite",
        bool(np.isfinite(arr).all()),
        extra,
    )


def run_one(name, fn):
    print(f"=== {name} ===")
    try:
        fn()
    except Exception as exc:
        print(name, "FAILED", type(exc).__name__, exc)


def do_cp05():
    run = load("verification/cases/euler_poisson/multifluid_modes/run.py")
    exact = load("verification/cases/euler_poisson/multifluid_modes/exact.py")
    t_end = 0.125
    field = np.asarray(run.run_native(32, t_end=t_end, mode="plus"), dtype=np.float64)
    centers, volumes = exact.uniform_cell_centers(32)
    ref = exact.exact_state(centers, t_end, mode="plus")
    err = reference_errors(field[0], ref[0], volumes)
    _report("CP-05", field, f"n_L2 {err.l2} n_Linf {err.linf}")


def do_cp06():
    run = load("verification/cases/euler_poisson/ion_acoustic/run.py")
    field = np.asarray(run.run_native(32, t_end=0.10, mode="plus"), dtype=np.float64)
    _report("CP-06", field)


def do_tm02():
    run = load("verification/cases/time/noncommuting_strang/run.py")
    field = np.asarray(run.run_native(0.1, t_end=0.2), dtype=np.float64)
    _report("TM-02", field)


def do_tm03():
    run = load("verification/cases/time/collision_relax/run.py")
    field = np.asarray(run.run_native(0.05, t_end=0.2), dtype=np.float64)
    _report("TM-03", field)
    if hasattr(run, "run_native_two_species"):
        pair = np.asarray(run.run_native_two_species(0.05, t_end=0.2), dtype=np.float64)
        _report("TM-03-2s", pair)


def do_tm05():
    run = load("verification/cases/time/ap_limit/run.py")
    for eps in (1.0, 0.1, 0.01):
        field = np.asarray(run.run_native(0.1, t_end=0.2, eps=eps), dtype=np.float64)
        _report(f"TM-05 eps={eps}", field)


def do_tm06():
    run = load("verification/cases/time/multirate/run.py")
    for ratio in (1, 2, 4):
        field = np.asarray(run.run_native(0.25, r=ratio, t_end=0.25), dtype=np.float64)
        _report(f"TM-06 r={ratio}", field)


def do_tm07():
    run = load("verification/cases/time/rk_field_stages/run.py")
    field = np.asarray(run.run_native(16, t_end=0.05), dtype=np.float64)
    _report("TM-07", field)


def main() -> int:
    for name, fn in (
        ("CP-05", do_cp05),
        ("CP-06", do_cp06),
        ("TM-02", do_tm02),
        ("TM-03", do_tm03),
        ("TM-05", do_tm05),
        ("TM-06", do_tm06),
        ("TM-07", do_tm07),
    ):
        run_one(name, fn)
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
