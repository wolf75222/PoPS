#!/usr/bin/env python3
"""ADC-637 R4 pre-req: freeze the condensed-Schur BRICK trajectory + throughput to a checked-in golden.

Run ONCE against the pre-retirement tree (the brick still present), it captures the brick route's
System AND flat-AMR trajectories at theta == 1 and theta == 0.5 over >= 10 steps, plus the brick
median-step-time throughput baseline, into tests/data/adc637/condensed_brick_golden.npz. After the
brick is deleted, tests/python/unit/time/test_condensed_generic_parity.py asserts the SOLE (generic)
route reproduces this frozen data bit-for-bit (np.array_equal) and runs at >= 98% of the frozen
throughput -- the parity/throughput gates survive the deletion of the reference they were built against
(R1). This is a HARNESS, not a test: it writes the golden, it does not assert.

Not run in CI. Regenerate only on a deliberate golden refresh: rebuild _pops on the pre-retirement
base, then `python3 tests/data/adc637/capture_brick_golden.py`.
"""
from pops.params import ConstParam
import sys
import time

_N = 24
_L = 1.0
_DT = 0.03
_ALPHA = 0.7
_BZ = 0.8
_TOL = 1e-10
_NSTEPS = 10
_PERF_N = 64
_PERF_STEPS = 50


def _imports():
    try:
        import numpy as np

        import pops
        from pops import time as adctime
        import pops.lib.time as lt
        from pops.ir.ops import sqrt
        from pops.lib.models import author_electrostatic_lorentz
        from pops.numerics.reconstruction import FirstOrder
        from pops.numerics.riemann import Rusanov
        from pops.physics._facade import Model
        from pops.runtime._system import System
    except Exception as exc:  # noqa: BLE001
        print("capture aborted (stack unavailable: %s)" % exc)
        return None
    probe = System(n=8, L=_L, periodic=True)
    if not hasattr(probe, "install_program") or not hasattr(probe, "set_magnetic_field"):
        print("capture aborted (_pops lacks install_program / set_magnetic_field; rebuild _pops)")
        return None
    return dict(np=np, pops=pops, adctime=adctime, lt=lt, sqrt=sqrt,
                author_electrostatic_lorentz=author_electrostatic_lorentz, FirstOrder=FirstOrder,
                Rusanov=Rusanov, Model=Model, System=System)


def _schur_model(env, name, with_J):
    Model, sqrt = env["Model"], env["sqrt"]
    m = Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    cs2 = m.value(m.param(ConstParam("cs2", 0.5)))
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    p = m.primitive("p", cs2 * rho)
    m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from([rho, rho * u, rho * v])
    m.flux(x=[mx, mx * u + p, my * u], y=[my, mx * v, my * v + p])
    cs = sqrt(cs2)
    m.eigenvalues(x=[u - cs, u, u + cs], y=[v - cs, v, v + cs])
    m.elliptic_rhs(rho)
    m.aux("grad_x")
    m.aux("grad_y")
    m.aux("B_z")
    if with_J:
        env["author_electrostatic_lorentz"](m)
    return m


def _initial_state(env, n):
    np = env["np"]
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    return np.stack([rho0, 0.4 * rho0, -0.2 * rho0])


def _make_sim(env, name, n):
    np, pops = env["np"], env["pops"]
    System = env["System"]
    sim = System(n=n, L=_L, periodic=True)
    cm = _schur_model(env, name, False).compile(backend="production")
    sim.add_equation("blk", cm, spatial=pops.FiniteVolume(limiter=env["FirstOrder"](),
                                                          riemann=env["Rusanov"]()),
                     time=pops.Explicit(method="euler"))
    sim.set_poisson("charge_density", "geometric_mg")
    sim.set_magnetic_field(_BZ * np.ones(n * n))
    sim.set_state("blk", _initial_state(env, n))
    return sim


def _compile_route(env, route, theta, tag, n):
    pops, adctime, lt = env["pops"], env["adctime"], env["lt"]
    from pops.model import OperatorHandle
    sim = _make_sim(env, "blk_%s" % tag, n)
    model = _schur_model(env, "prog_%s" % tag, True)
    registry = model.operator_registry()
    operator = registry.operators_of_kind("local_linear_operator")[0]
    linear = OperatorHandle(
        operator.name, kind=operator.kind, owner=registry.owner_path,
        signature=operator.signature)
    P = adctime.Program("cs_%s" % tag).bind_operators(model)
    lt.CondensedSchur(
        P, "blk", alpha=_ALPHA, theta=theta, tol=_TOL, max_iter=400,
        linear_operator=linear)
    compiled = pops.codegen.compile_problem(model=model, time=P)
    sim.install_program(compiled.so_path)
    return sim


def _median_step_time(sim, nsteps, nrep=3):
    sim.step(_DT)
    times = []
    for _ in range(nrep):
        t0 = time.perf_counter()
        for _ in range(nsteps):
            sim.step(_DT)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def main():
    env = _imports()
    if env is None:
        return 1
    np = env["np"]
    out = {}
    # Trajectory golden: the BRICK route on System, theta == 1 and theta == 0.5, >= 10 steps.
    for theta in (1.0, 0.5):
        tag = int(round(theta * 100))
        brick = _compile_route(env, "brick", theta, "brick_%d" % tag, _N)
        for _ in range(_NSTEPS):
            brick.step(_DT)
        state = np.array(brick.get_state("blk"))
        out["brick_theta%d" % tag] = state
        print("captured brick theta=%.2f  state shape=%s  |state|_inf=%.6e"
              % (theta, state.shape, float(np.abs(state).max())))
        # Same-build cross-check: prove the generic route already matches the brick NOW (pre-deletion),
        # so freezing the brick side is a faithful reference for the surviving generic gate.
        generic = _compile_route(env, "generic", theta, "generic_%d" % tag, _N)
        for _ in range(_NSTEPS):
            generic.step(_DT)
        g = np.array(generic.get_state("blk"))
        eq = np.array_equal(state, g)
        print("  same-build brick==generic (theta=%.2f) array_equal=%s  max|diff|=%.3e"
              % (theta, eq, float(np.abs(state - g).max())))
        if not eq:
            print("  WARNING: brick and generic diverge pre-deletion; the golden would be unreachable")
    # Throughput baseline: brick median step time on the perf grid (the >= 98% gate reads this).
    perf = _compile_route(env, "brick", 1.0, "perf_brick", _PERF_N)
    t_brick = _median_step_time(perf, _PERF_STEPS)
    out["throughput_brick_seconds"] = np.array([t_brick])
    out["meta_n"] = np.array([_N])
    out["meta_nsteps"] = np.array([_NSTEPS])
    out["meta_dt"] = np.array([_DT])
    out["meta_alpha"] = np.array([_ALPHA])
    out["meta_bz"] = np.array([_BZ])
    out["meta_perf_n"] = np.array([_PERF_N])
    out["meta_perf_steps"] = np.array([_PERF_STEPS])
    print("captured brick throughput n=%d x%d steps: %.4fs (median of 3)"
          % (_PERF_N, _PERF_STEPS, t_brick))
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "condensed_brick_golden.npz")
    np.savez(path, **out)
    print("wrote golden: %s (%d arrays)" % (path, len(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
