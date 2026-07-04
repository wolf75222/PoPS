#!/usr/bin/env python3
"""ADC-634: MPI parity of the clean pops.compile(layout=AMR) + pops.bind Program route.

The clean AMR Program route (ADC-634) must be MPI-correct: the DISTRIBUTED run (np > 1) must produce
the BIT-IDENTICAL globally-gathered coarse density as the direct AmrSystem.install_program route on
the same ranks -- the clean route only adds the Problem authoring + config derivation, so the
distributed arithmetic is identical. Under MPI the AMR density() accessor is COLLECTIVE (every rank
calls it and it returns the global coarse array), so the comparison is a whole-domain bit-identity.

This is the MPI analogue of test_amr_clean_route_program's acceptance (a): at a fixed np it asserts
   clean route (distributed) == direct install_program route (distributed),   bit-for-bit,
plus coarse-mass conservation. Both routes run at the SAME np, so this is a dist==dist parity at
fixed np (the AMR MPI parity gate), NOT a cross-np comparison.

WHAT NEEDS WHICH RUNNER. Needs an MPI _pops build (pops.n_ranks() > 1) AND a compiler + a visible
Kokkos (POPS_KOKKOS_ROOT) to build the .so. Self-skips (exit 0) when pops.n_ranks() == 1 (a
single-rank / non-MPI build -- run it under `mpirun -np 2 python3 <file>`), when the .so cannot
build, and when pops / numpy is unavailable. Runs in the CI MPI lane. No fake pops -- a leg that
cannot build the .so skips, never fakes the engine.
"""
import os
import sys

try:
    import numpy as np

    import pops
except Exception as exc:  # noqa: BLE001 -- pops/numpy unavailable in this interpreter
    print("skip test_amr_clean_route_program_mpi (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)

# Reuse the DIRECT-route helpers and the CLEAN-route helpers already validated single-rank.
_AMR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "amr")
_AMR_DIR = os.path.abspath(_AMR_DIR)
if _AMR_DIR not in sys.path:
    sys.path.insert(0, _AMR_DIR)
import test_amr_program_parity as parity  # noqa: E402
import test_amr_clean_route_program as clean  # noqa: E402

N = parity.N
NSTEPS = parity.NSTEPS
DT = parity.DT

_fails = 0


def chk(cond, label):
    global _fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        _fails += 1


def _n_ranks():
    """The MPI world size the built _pops reports (1 on a serial / non-MPI build)."""
    fn = getattr(pops, "n_ranks", None)
    try:
        return int(fn()) if callable(fn) else 1
    except Exception:  # noqa: BLE001 -- a non-MPI build may not expose it
        return 1


def test_clean_route_distributed_equals_direct_distributed():
    """At np > 1 the clean pops.compile(layout=AMR)+pops.bind SSPRK2 route produces the BIT-IDENTICAL
    globally-gathered coarse density as the direct install_program route on the same ranks, and
    conserves the coarse mass. Skips single-rank (run under mpirun -np 2)."""
    nr = _n_ranks()
    print("== ADC-634 MPI clean-route parity (np=%d) ==" % nr)
    if nr < 2:
        print("skip (single rank: pops.n_ranks()=%d; run under `mpirun -np 2 python3 <file>` with an "
              "MPI _pops build)" % nr)
        return

    u0 = parity._init_density()
    direct, derr = parity._amr_run(parity._ssprk2_program(), parity._euler_model("adc634_mpi"), u0)
    if direct is None:
        print("skip (%s)" % derr)
        return
    result, cerr = clean._clean_amr_run(parity._ssprk2_program(),
                                        parity._euler_model("adc634_mpi"), u0)
    if result is None:
        print("skip (%s)" % cerr)
        return

    direct_rho, _direct_phi, direct_mass = direct
    clean_rho, _clean_phi, clean_mass = result
    chk(np.array_equal(direct_rho, clean_rho),
        "np=%d: clean-route gathered coarse density is BIT-IDENTICAL to the direct route "
        "(max|diff| = %.3e)" % (nr, float(np.abs(direct_rho - clean_rho).max())))
    chk(np.array_equal(np.array([direct_mass]), np.array([clean_mass])),
        "np=%d: clean-route coarse mass is bit-identical (%.17g vs %.17g)"
        % (nr, direct_mass, clean_mass))
    # Mass conservation (periodic, no boundary flux): the coarse mass equals the seed to round-off.
    m0 = float(u0.mean())  # mean density == coarse mass / area (L=1)
    chk(abs(clean_mass - m0) < 1e-9,
        "np=%d: the clean route conserves the coarse mass (|m - m0| = %.2e)"
        % (nr, abs(clean_mass - m0)))


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print("\n%s test_amr_clean_route_program_mpi (%d check failures)"
          % ("FAIL" if _fails else "PASS", _fails))
    return _fails


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
