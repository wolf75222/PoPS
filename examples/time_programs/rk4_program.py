#!/usr/bin/env python3
"""Classic RK4 as a compiled multi-stage time Program (epic ADC-399 / ADC-407).

Writes the four-stage RK4 scheme with ``pops.time.Program`` (three intermediate stage states + a
linear-combination commit), compiles it to a ``problem.so``, installs it, advances one step
C++-side, and checks it against an offline stage-by-stage reference built from the same runtime
primitives. There is NO special RK4 C++ class -- the scheme is just IR lowered by the codegen.

Run::

    python examples/time_programs/rk4_program.py

Requires a compiler + a visible Kokkos (``POPS_KOKKOS_ROOT``); prints a skip notice and exits 0
otherwise. cf. docs/sphinx/reference/time-program.md.
"""
import sys

try:
    import numpy as np

    import pops
    from pops.solvers import GeometricMG
    from pops import time as adctime
    from _module_models import explicit_euler, first_order_rusanov, isothermal_transport_module
except Exception as exc:  # noqa: BLE001
    print("skip rk4_program (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)


def gas_model():
    return isothermal_transport_module("rk4_model")


N = 48


def initial_state():
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    return np.stack([rho, 0.4 * rho, -0.2 * rho])


def build_system():
    """The native reference System, evaluated one RHS stage at a time."""
    sim = pops.System(n=N, L=1.0, periodic=True)
    sim.install(None,
                instances={"plasma": {"model": gas_model(),
                                      "spatial": first_order_rusanov(),
                                      "time": explicit_euler(),
                                      "initial": initial_state()}},
                solvers={"phi": GeometricMG()})
    return sim


def rk4_program():
    P = adctime.Program("rk4_example")
    pops.lib.time.rk4(P, "plasma")
    return P


def offline_rhs(ref, U):
    ref._set_state("plasma", U)
    ref.solve_fields()
    return np.array(ref._eval_rhs("plasma"))


def main():
    dt = 2e-3
    try:
        compiled = pops.compile_problem(model=gas_model(), time=rk4_program())
    except RuntimeError as exc:
        print("skip rk4_program (compile_problem could not build the .so: %s)" % str(exc)[:160])
        return 0

    # Compiled path via the unified headline entry: install() wires the block instance, its initial
    # state and the Poisson solver, then installs the compiled time Program -- in one call.
    sim = pops.System(n=N, L=1.0, periodic=True)
    sim.install(compiled,
                instances={"plasma": {"model": gas_model(),
                                      "spatial": first_order_rusanov(),
                                      "initial": initial_state()}},
                solvers={"phi": GeometricMG()})
    U0 = np.array(sim._get_state("plasma"))
    sim.step(dt)
    U_prog = np.array(sim._get_state("plasma"))

    ref = build_system()
    k1 = offline_rhs(ref, U0)
    k2 = offline_rhs(ref, U0 + 0.5 * dt * k1)
    k3 = offline_rhs(ref, U0 + 0.5 * dt * k2)
    k4 = offline_rhs(ref, U0 + dt * k3)
    U_ref = U0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    err = float(np.abs(U_prog - U_ref).max())
    print("compiled RK4 Program vs offline stage reference: max|d| = %.2e" % err)
    ok = err < 1e-12
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
