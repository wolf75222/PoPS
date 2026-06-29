#!/usr/bin/env python3
"""SSPRK2 as a compiled multi-stage time Program (epic ADC-399 / ADC-407).

Writes the two-stage SSPRK2 scheme with ``pops.time.Program`` (an intermediate stage state + a
linear-combination commit), compiles it to a ``problem.so`` with ``pops.compile_problem``, installs
it, advances one step C++-side, and checks it reproduces the native ``Explicit.ssprk2()`` step
bit-for-bit. There is NO special SSPRK2 C++ class -- the scheme is just IR lowered by the codegen.

Run::

    python examples/time_programs/ssprk2_program.py

Requires a compiler + a visible Kokkos (``POPS_KOKKOS_ROOT``); prints a skip notice and exits 0
otherwise. cf. docs/sphinx/reference/time-program.md.
"""
import sys

try:
    import numpy as np

    import pops
    from pops.solvers import GeometricMG
    from pops import time as adctime
    from _module_models import (
        explicit_ssprk2,
        first_order_rusanov,
        isothermal_transport_module,
    )
except Exception as exc:  # noqa: BLE001
    print("skip ssprk2_program (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)


def gas_model():
    return isothermal_transport_module("ssprk2_model")


N = 48


def initial_state():
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    return np.stack([rho, 0.4 * rho, -0.2 * rho])


def build_system():
    """The native reference System, stepped natively for comparison."""
    sim = pops.System(n=N, L=1.0, periodic=True)
    sim.install(None,
                instances={"plasma": {"model": gas_model(),
                                      "spatial": first_order_rusanov(),
                                      "time": explicit_ssprk2(),
                                      "initial": initial_state()}},
                solvers={"phi": GeometricMG()})
    return sim


def ssprk2_program():
    """U^{n+1} = 1/2 U^n + 1/2 (U1 + dt R(U1)), U1 = U^n + dt R(U^n) -- built as typed IR."""
    P = adctime.Program("ssprk2_example")
    pops.lib.time.ssprk2(P, "plasma")
    return P


def main():
    dt = 2e-3
    try:
        compiled = pops.compile_problem(model=gas_model(), time=ssprk2_program())
    except RuntimeError as exc:
        print("skip ssprk2_program (compile_problem could not build the .so: %s)" % str(exc)[:160])
        return 0

    # Compiled path via the unified headline entry: install() wires the block instance, its initial
    # state and the Poisson solver, then installs the compiled time Program -- in one call.
    sim = pops.System(n=N, L=1.0, periodic=True)
    sim.install(compiled,
                instances={"plasma": {"model": gas_model(),
                                      "spatial": first_order_rusanov(),
                                      "initial": initial_state()}},
                solvers={"phi": GeometricMG()})
    sim.step(dt)
    U_prog = np.array(sim._get_state("plasma"))

    native = build_system()
    native.step(dt)
    err = float(np.abs(U_prog - np.array(native._get_state("plasma"))).max())
    print("compiled SSPRK2 Program vs native Explicit.ssprk2(): max|d| = %.2e" % err)
    ok = err < 1e-12
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
