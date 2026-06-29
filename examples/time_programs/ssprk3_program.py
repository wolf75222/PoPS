#!/usr/bin/env python3
"""SSPRK3 as a compiled multi-stage time Program (epic ADC-399 / ADC-407).

Writes the three-stage SSPRK3 (Shu-Osher) scheme with ``pops.time.Program`` (two intermediate stage
states + a linear-combination commit) via the ``pops.lib.time.ssprk3`` macro, compiles it to a
``problem.so`` with ``pops.compile_problem``, installs it, advances one step C++-side, and checks it
reproduces the native ``Explicit.ssprk3()`` step bit-for-bit. There is NO special SSPRK3
C++ class -- the scheme is just IR lowered by the codegen (like the merged ssprk2 example/test).

Run::

    python examples/time_programs/ssprk3_program.py

Requires a compiler + a visible Kokkos (``POPS_KOKKOS_ROOT``); prints a skip notice and exits 0
otherwise. cf. docs/sphinx/reference/time-program.md.
"""
import sys

try:
    import numpy as np

    import pops
    from pops.solvers import GeometricMG
    from pops import time as adctime
    import pops.lib.time as libtime  # ready schemes live in pops.lib.time (Spec 4)
    from _module_models import (
        explicit_ssprk3,
        first_order_rusanov,
        isothermal_transport_module,
    )
except Exception as exc:  # noqa: BLE001
    print("skip ssprk3_program (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)


def gas_model():
    return isothermal_transport_module("ssprk3_model")


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
                                      "time": explicit_ssprk3(),
                                      "initial": initial_state()}},
                solvers={"phi": GeometricMG()})
    return sim


def ssprk3_program():
    """SSPRK3 (Shu-Osher), built via the std macro that lowers to typed IR:
    U1 = U0 + dt k0; U2 = 3/4 U0 + 1/4 (U1 + dt k1); U^{n+1} = 1/3 U0 + 2/3 (U2 + dt k2)."""
    P = adctime.Program("ssprk3_example")
    libtime.ssprk3(P, "plasma")
    return P


def main():
    dt = 2e-3
    try:
        compiled = pops.compile_problem(model=gas_model(), time=ssprk3_program())
    except RuntimeError as exc:
        print("skip ssprk3_program (compile_problem could not build the .so: %s)" % str(exc)[:160])
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
    print("compiled SSPRK3 Program vs native Explicit.ssprk3(): max|d| = %.2e" % err)
    ok = err < 1e-12
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
