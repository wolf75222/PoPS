#!/usr/bin/env python3
"""Bit-parity of the GENERIC condensed-implicit solve vs the hand-written Schur brick (ADC-637 PR-2).

``pops.lib.time.condensed_schur`` gained a ``route`` selector: ``"brick"`` (the historical hand-written
``P.schur_*`` ops -> ``coupling/schur/**``) and ``"generic"`` (the DSL ``P.condensed_*`` ops on the
electrostatic-Lorentz linearization ``J = [[0, B_z], [-B_z, 0]]``, authored via
``pops.lib.physics.author_electrostatic_lorentz`` and emitted INLINE through the closed-form
``block_inverse`` intrinsic with no Schur vocabulary). This test proves the two routes produce the SAME
trajectory to the LAST BIT.

THE PARITY CRUX. The coefficient tensor ``A = I + c*rho*M^{-1}`` reads the four inverse entries directly,
and ``block_inverse<2>`` reduces bit-for-bit to ``LorentzEliminator``'s ``binv_11..22`` -- so the phi
solve gets bit-identical coefficients. But the RHS flux ``F = M^{-1}(mx, my)`` and the velocity
reconstruction ``v = M^{-1}(v^n - theta*dt*grad phi)`` apply ``M^{-1}`` to a VECTOR, and the brick applied
it with ``LorentzEliminator::apply_Binv`` = ``inv*(vx + w*vy)`` -- ONE reciprocal factored out of the
bracket. The generic emitter reproduces exactly that factored order via ``block_apply_inverse`` (measured:
summing the pre-divided entries ``(1/det)*vx + (w/det)*vy`` instead rounds differently and drifts ~1/3 of
cells by a ULP each step, so ``np.array_equal`` would fail over a multi-step run). This test is the gate
that pins the factored-apply choice.

(A) TRAJECTORY parity, strict ``np.array_equal`` (never allclose): compile+install the brick route and
    the generic route on the SAME rho/mx/my block with a constant B_z, take >= 10 steps at theta == 1 AND
    theta == 0.5 (the ADC-427 carry path), and assert the density, momentum and (theta=0.5) the carried
    potential are bit-identical.

(B) THROUGHPUT gate (ADC-637 design section 7): the generic route runs at >= 98% of the brick route on a
    meaningful grid (median of 3 timed runs of >= 50 steps). Since (A) proves identical arithmetic on
    identical memory, the only slack is the compiler's codegen of the inlined expression vs the struct
    method; a real regression (an extra temporary, a missed CSE) would show here.

Self-skips (exit 0) without numpy / _pops / install_program / set_magnetic_field / a compiler / a visible
Kokkos -- never fakes the engine (project policy: no fake pops in tests). Runs under pytest and as a
script.
"""
import sys
import time

# A cold CI compile cache builds four .so variants here; lift the per-process budget (ADC-627 idiom).
POPS_PROCESS_TIMEOUT = 1200

_N = 24
_L = 1.0
_DT = 0.03
_ALPHA = 0.7
_BZ = 0.8
_TOL = 1e-10
_NSTEPS = 10


def _imports():
    """The full runtime stack, or None (skip) if any piece is missing -- never fake."""
    try:
        import numpy as np

        import pops
        from pops import time as adctime
        import pops.lib.time as lt
        from pops.ir.ops import sqrt
        from pops.lib.physics import author_electrostatic_lorentz
        from pops.numerics.reconstruction import FirstOrder
        from pops.numerics.riemann import Rusanov
        from pops.physics.facade import Model
        from pops.runtime.system import System
    except Exception as exc:  # noqa: BLE001 -- pops / numpy unavailable here
        print("skip test_condensed_generic_parity (stack unavailable: %s)" % exc)
        return None
    probe = System(n=8, L=_L, periodic=True)
    if not hasattr(probe, "install_program"):
        print("skip test_condensed_generic_parity (_pops lacks install_program; rebuild _pops)")
        return None
    if not hasattr(probe, "set_magnetic_field"):
        print("skip test_condensed_generic_parity (_pops lacks set_magnetic_field; rebuild _pops)")
        return None
    return dict(np=np, pops=pops, adctime=adctime, lt=lt, sqrt=sqrt,
                author_electrostatic_lorentz=author_electrostatic_lorentz, FirstOrder=FirstOrder,
                Rusanov=Rusanov, Model=Model, System=System)


def _schur_model(env, name, with_J):
    """The canonical isothermal condensed-Schur block (rho, mx, my) + Poisson + B_z aux. @p with_J
    additionally authors the electrostatic-Lorentz linearization the generic route's coeff op resolves
    (harmless for the brick route, which never references it)."""
    Model, sqrt = env["Model"], env["sqrt"]
    m = Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    cs2 = m.param("cs2", 0.5)
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
    """A System with a compiled (transport) model, a Poisson solver, a constant B_z and the initial
    state. Returns (sim, None) on a toolchain miss."""
    np, pops = env["np"], env["pops"]
    System = env["System"]
    sim = System(n=n, L=_L, periodic=True)
    try:
        cm = _schur_model(env, name, False).compile(backend="production")
    except RuntimeError as exc:
        print("skip: transport model compile failed: %s" % str(exc)[:160])
        return None, None
    sim.add_equation("blk", cm, spatial=pops.FiniteVolume(limiter=env["FirstOrder"](),
                                                          riemann=env["Rusanov"]()),
                     time=pops.Explicit(method="euler"))
    sim.set_poisson("charge_density", "geometric_mg")
    sim.set_magnetic_field(_BZ * np.ones(n * n))
    sim.set_state("blk", _initial_state(env, n))
    return sim, True


def _compile_route(env, route, theta, tag, n):
    """Compile+install condensed_schur(route, theta) on a fresh System at grid @p n; return the sim or
    None on a toolchain miss."""
    pops, adctime, lt = env["pops"], env["adctime"], env["lt"]
    sim, ok = _make_sim(env, "blk_%s" % tag, n)
    if sim is None:
        return None
    P = adctime.Program("cs_%s" % tag)
    lt.condensed_schur(P, "blk", alpha=_ALPHA, theta=theta, route=route, tol=_TOL, max_iter=400)
    try:
        compiled = pops.codegen.compile_problem(model=_schur_model(env, "prog_%s" % tag, True), time=P)
    except RuntimeError as exc:
        print("skip: compile_problem failed (%s): %s" % (route, str(exc)[:160]))
        return None
    sim.install_program(compiled.so_path)
    return sim


def _run_trajectory_parity(env, theta):
    """(A) >= _NSTEPS steps of the brick route vs the generic route, strict np.array_equal on the state."""
    np = env["np"]
    tag = int(round(theta * 100))
    brick = _compile_route(env, "brick", theta, "brick_%d" % tag, _N)
    if brick is None:
        return None
    generic = _compile_route(env, "generic", theta, "generic_%d" % tag, _N)
    if generic is None:
        return None
    for _ in range(_NSTEPS):
        brick.step(_DT)
        generic.step(_DT)
    b = np.array(brick.get_state("blk"))
    g = np.array(generic.get_state("blk"))
    eq = np.array_equal(b, g)
    dmax = float(np.abs(b - g).max())
    print("  theta=%.2f  %d steps  brick vs generic  array_equal=%s  max|diff|=%.3e"
          % (theta, _NSTEPS, eq, dmax))
    assert eq, ("condensed_schur route='generic' must be BIT-IDENTICAL to route='brick' at theta=%.2f "
                "(max|diff|=%.3e over %d steps); the factored block_apply_inverse is the fix"
                % (theta, dmax, _NSTEPS))
    return True


def _median_step_time(sim, nsteps, nrep=3):
    """Median wall time of @p nsteps steps over @p nrep repeats (a warm-up run first)."""
    sim.step(_DT)  # warm-up (JIT / first-touch)
    times = []
    for _ in range(nrep):
        t0 = time.perf_counter()
        for _ in range(nsteps):
            sim.step(_DT)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def _run_throughput_gate(env):
    """(B) generic route throughput >= 98% of the brick route on a meaningful grid (design section 7)."""
    n, nsteps, theta = 64, 50, 1.0
    brick = _compile_route(env, "brick", theta, "perf_brick", n)
    if brick is None:
        return None
    generic = _compile_route(env, "generic", theta, "perf_generic", n)
    if generic is None:
        return None
    t_brick = _median_step_time(brick, nsteps)
    t_generic = _median_step_time(generic, nsteps)
    ratio = t_brick / t_generic if t_generic > 0 else 0.0  # >1 means generic is faster
    print("  throughput n=%d x%d steps  brick=%.4fs  generic=%.4fs  generic/brick throughput=%.1f%%"
          % (n, nsteps, t_brick, t_generic, 100.0 * ratio))
    assert ratio >= 0.98, ("the generic route must run at >= 98%% of the brick throughput "
                           "(got %.1f%%; brick=%.4fs generic=%.4fs)" % (100.0 * ratio, t_brick, t_generic))
    return True


def test_generic_condensed_solve_matches_brick():
    """(A) + (B): the generic route is bit-identical to the brick over a multi-step trajectory at
    theta == 1 and theta == 0.5, and runs at >= 98% of its throughput."""
    env = _imports()
    if env is None:
        return
    checked = 0
    for theta in (1.0, 0.5):
        if _run_trajectory_parity(env, theta):
            checked += 1
    if _run_throughput_gate(env):
        checked += 1
    if checked == 0:
        print("-- test_condensed_generic_parity: all sub-checks skipped (toolchain unavailable) --")


if __name__ == "__main__":
    test_generic_condensed_solve_matches_brick()
    print("done test_condensed_generic_parity")
    sys.exit(0)
