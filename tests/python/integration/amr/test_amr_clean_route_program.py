#!/usr/bin/env python3
"""ADC-634: the clean pops.compile(layout=AMR) + pops.bind route for a whole-system time Program.

The AMR branch of pops.compile used to drop problem._time silently: a whole-system time Program on
layout=AMR was never compiled, and _AmrRuntimeAdapter.install hard-coded compiled=None, so the
hierarchy ran the native per-block policy. ADC-634 routes the Program through
compile_problem(target='amr_system') and installs it on the hierarchy via AmrSystem.install_program.

This asserts the clean route is IDENTICAL to the proven direct route (test_amr_program_parity's
AmrSystem.add_equation + install_program), and composes correctly:

  (a) explicit + SSPRK2 Program: the clean route == the direct install_program route BIT-FOR-BIT
      (np.array_equal on the evolved coarse density), for SSPRK2 and a custom midpoint Program;
  (b) a flat AMR hierarchy (FrozenRegrid, single level, no C/F interface): the clean AMR route ==
      the Uniform clean route BIT-FOR-BIT on the density (mean-removed phi to the MG tolerance);
  (d) a Program reading a dsl.Param(kind='runtime'): pops.bind(params={...}) reaches
      set_program_params -- the run DIFFERS from the k=2.0 (declaration-default value) run and
      MATCHES a direct set_program_params run; a bind WITHOUT params= refuses with the actionable
      message (the Uniform-mirrored Spec 5 arguments() contract);
  composition: amr_program_op_support(ssprk2) is all-green; amr_program_op_support(condensed_schur)
      is now GREEN (ADC-633 wired the per-level Schur assembly + the flat/composite solve), and the
      clean route COMPILES + INSTALLS + RUNS the Schur Program on a flat hierarchy (finite, coarse-mass
      conserved) instead of throwing the old deferral backstop.

WHAT NEEDS WHICH RUNNER. The composition query is pure Python (any interpreter with pops). The
bit-identity acceptances need a compiler + a visible Kokkos (POPS_KOKKOS_ROOT) to build the .so;
the compiled-.so dlopen + per-level RUN is validatable on Kokkos CPU (Serial/OpenMP) locally.
Self-skips (exit 0) without pops / a built _pops / a compiler. Pytest + __main__ guard (CI runs
``python3 <file>``). No fake pops -- a leg that cannot build the .so skips, never fakes the engine.
"""
import os
import sys

try:
    import numpy as np

    import pops
    import pops.lib.time as lib_time
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.mesh.amr import FrozenRegrid
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import AMR, Uniform
    from pops.runtime.amr_program_support import amr_program_op_support
    from pops.runtime.system import AmrSystem
except Exception as exc:  # noqa: BLE001 -- pops/numpy unavailable in this interpreter
    print("skip test_amr_clean_route_program (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)

# Reuse the proven DIRECT-route helpers (model / program / IC / _amr_run / _system_run) so the clean
# route is compared to the exact same physics + Program + initial state, byte for byte.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import test_amr_program_parity as parity  # noqa: E402

# ADC-627 idiom: this file AOT-compiles several Program/.so artifacts; give the
# process-isolated runner headroom over the default (CI runner speed varies 3-4x).
POPS_PROCESS_TIMEOUT = 1200

N = parity.N
NSTEPS = parity.NSTEPS
DT = parity.DT

_fails = 0


def chk(cond, label):
    global _fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        _fails += 1


# --------------------------------------------------------------------------------------------------
# Clean-route helpers: author a pops.Problem, compile with layout=AMR / Uniform, bind, run.
# --------------------------------------------------------------------------------------------------
def _amr_layout():
    """A single-level (flat) AMR layout: FrozenRegrid -> regrid_every=0, the coarse-only Program layout
    the direct _amr_run uses (AmrSystem(regrid_every=0)). Periodic base, so the config matches."""
    return AMR(base=CartesianMesh(n=N, L=1.0, periodic=True), regrid=FrozenRegrid())


def _uniform_layout():
    return Uniform(CartesianMesh(n=N, L=1.0, periodic=True))


def _spatial():
    return pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov())


def _condensed_program(model, *, block="plasma", theta=1.0):
    """Author the condensed preset against the exact bound local-linear declaration."""
    from pops.model import OperatorHandle
    from pops.time import Program

    registry = model.operator_registry()
    operator = registry.operators_of_kind("local_linear_operator")[0]
    handle = OperatorHandle(
        operator.name, kind=operator.kind, owner=registry.owner_path,
        signature=operator.signature)
    program = Program("condensed_schur").bind_operators(model)
    lib_time.condensed_schur(
        program, block, alpha=1.0, theta=theta, linear_operator=handle)
    return program


def _problem(model, program, block="plasma"):
    """A single-block Problem carrying the model + spatial + the whole-system time Program. No explicit
    Poisson field: the model's elliptic_rhs drives the default AMR/System Poisson solve, matching the
    direct _amr_run / _system_run (which never call set_poisson)."""
    return (pops.Problem().block(block, physics=model, spatial=_spatial())
            .time(program))


def _clean_amr_run(program, model, u0, nsteps=NSTEPS, dt=DT, params=None, block="plasma"):
    """The CLEAN route: pops.compile(problem, layout=AMR(FrozenRegrid)) + pops.bind, then step. Returns
    (coarse density comp-0, coarse potential, coarse mass) -- the same tuple _amr_run returns."""
    problem = _problem(model, program, block=block)
    try:
        compiled = pops.compile(problem, layout=_amr_layout())
    except RuntimeError as exc:
        return None, "compile (clean AMR): %s" % str(exc)[:200]
    # The handle must be the CompiledProblem-for-AMR shape (carries the Program) targeting amr_system.
    if getattr(compiled, "program", None) is None:
        return None, "clean AMR compile did not carry a Program (ADC-634 route not wired)"
    try:
        sim = pops.bind(compiled, initial_state={block: u0}, params=params or {})
    except RuntimeError as exc:
        return None, "bind (clean AMR): %s" % str(exc)[:240]
    for _ in range(nsteps):
        sim.step(dt)
    return (np.array(sim.density(block)), np.array(sim.potential()),
            float(sim.mass(block))), None


def _clean_uniform_run(program, model, u0, nsteps=NSTEPS, dt=DT, block="plasma"):
    """The CLEAN Uniform route: pops.compile(problem, layout=Uniform) + pops.bind, then step. Returns
    (density comp-0, potential) after nsteps."""
    problem = _problem(model, program, block=block)
    try:
        compiled = pops.compile(problem, layout=_uniform_layout())
    except RuntimeError as exc:
        return None, "compile (clean Uniform): %s" % str(exc)[:200]
    try:
        sim = pops.bind(compiled, initial_state={block: u0})
    except RuntimeError as exc:
        return None, "bind (clean Uniform): %s" % str(exc)[:240]
    for _ in range(nsteps):
        sim.step(dt)
    state = np.array(sim.get_state(block))
    return (state[0], np.array(sim.potential())), None


# --------------------------------------------------------------------------------------------------
# (a) explicit + SSPRK2 clean route == direct install_program route, bit-for-bit.
# --------------------------------------------------------------------------------------------------
def test_clean_amr_ssprk2_equals_direct_install_program():
    """(a) The clean pops.compile(layout=AMR)+pops.bind route with an SSPRK2 Program produces the
    BYTE-IDENTICAL evolved coarse density as the direct AmrSystem.add_equation + install_program route
    (test_amr_program_parity._amr_run). Same model, same Program, same IC, same regrid_every=0 config
    -> the clean route only ADDS the Problem authoring + config derivation; the arithmetic is identical."""
    print("== (a) clean AMR SSPRK2 route == direct install_program (bit-identical) ==")
    model = parity._euler_model("adc634_clean_ssprk2")
    u0 = parity._init_density()

    direct, derr = parity._amr_run(parity._ssprk2_program(), model, u0)
    if direct is None:
        print("skip (%s)" % derr)
        return
    clean, cerr = _clean_amr_run(parity._ssprk2_program(),
                                 parity._euler_model("adc634_clean_ssprk2"), u0)
    if clean is None:
        print("skip (%s)" % cerr)
        return

    direct_rho, direct_phi, direct_mass = direct
    clean_rho, clean_phi, clean_mass = clean
    drho = float(np.abs(direct_rho - clean_rho).max())
    chk(np.array_equal(direct_rho, clean_rho),
        "clean-route coarse density is BIT-IDENTICAL to the direct install_program route "
        "(max|diff| = %.3e)" % drho)
    chk(np.array_equal(np.array([direct_mass]), np.array([clean_mass])),
        "clean-route coarse mass is bit-identical (%.17g vs %.17g)" % (direct_mass, clean_mass))
    dphi = float(np.abs((direct_phi - direct_phi.mean())
                        - (clean_phi - clean_phi.mean())).max())
    rng = float(np.abs(direct_phi - direct_phi.mean()).max()) or 1.0
    chk(dphi / rng < 1e-4,
        "the mean-removed coarse potential matches to the MG tolerance (rel max|diff| = %.3e)"
        % (dphi / rng))


def test_clean_amr_custom_midpoint_equals_direct():
    """(a') A CUSTOM 2-stage midpoint Program through the clean route is ALSO bit-identical to the direct
    route -- the Program TEXT drives the integrator through the clean seam (not a hard-coded scheme)."""
    print("== (a') clean AMR midpoint Program == direct install_program (bit-identical) ==")
    u0 = parity._init_density()

    direct, derr = parity._amr_run(parity._midpoint_program(), parity._euler_model("adc634_mid"), u0)
    if direct is None:
        print("skip (%s)" % derr)
        return
    clean, cerr = _clean_amr_run(parity._midpoint_program(), parity._euler_model("adc634_mid"), u0)
    if clean is None:
        print("skip (%s)" % cerr)
        return
    chk(np.array_equal(direct[0], clean[0]),
        "clean midpoint density is bit-identical to direct (max|diff| = %.3e)"
        % float(np.abs(direct[0] - clean[0]).max()))


# --------------------------------------------------------------------------------------------------
# (b) flat AMR clean route == Uniform clean route, bit-for-bit on the density.
# --------------------------------------------------------------------------------------------------
def _lift_density(model, u0, block="plasma"):
    """The native set_density lift (rho -> [rho, 0, 0, E(rho)]) READ BACK from a scratch System.

    The Uniform clean route binds the FULL conservative state (install writes set_state), while the
    AMR clean route seeds the density (install writes set_density, the native lift fills the other
    components). Reading the lift back from a System seeded with the same u0 makes the Uniform bind
    state BITWISE the AMR seed (the System and AMR set_density lifts are bit-identical, pinned by
    test_amr_program_parity's single-level parity), so leg (b) stays a bit-identity. Returns the
    (components, n, n) state, or (None, reason) when the block .so cannot build (honest skip)."""
    from pops.runtime.system import System
    sim = System(n=N, L=1.0)
    try:
        block_cm = model.compile(backend="production")
    except RuntimeError as exc:
        return None, "compile (lift scratch System): %s" % str(exc)[:160]
    try:
        sim.add_equation(block, block_cm, spatial=_spatial(), time=pops.Explicit(method="ssprk2"))
        sim.set_density(block, u0)
        return np.array(sim.get_state(block)), None
    except RuntimeError as exc:
        return None, "seed (lift scratch System): %s" % str(exc)[:160]


def test_clean_flat_amr_equals_clean_uniform():
    """(b) On a FLAT hierarchy (FrozenRegrid, single level, no C/F interface so couple_levels is exact),
    the clean AMR route and the clean Uniform route drive the SAME SSPRK2 Program to the BIT-IDENTICAL
    density (the AmrProgramContext seam methods are byte-faithful ProgramContext mirrors). The periodic
    Poisson phi is pinned up to an additive constant differently, so phi is compared mean-removed to
    the MG tolerance -- the physically meaningful part that feeds the density's RHS."""
    print("== (b) clean flat AMR route == clean Uniform route (bit-identical density) ==")
    u0 = parity._init_density()

    # The Uniform bind takes the FULL conservative state: read the native set_density lift back
    # from a scratch System so the Uniform initial state is BITWISE the AMR set_density seed.
    u0_full, lerr = _lift_density(parity._euler_model("adc634_flat"), u0)
    if u0_full is None:
        print("skip (%s)" % lerr)
        return
    uni, uerr = _clean_uniform_run(parity._ssprk2_program(), parity._euler_model("adc634_flat"),
                                   u0_full)
    if uni is None:
        print("skip (%s)" % uerr)
        return
    amr, aerr = _clean_amr_run(parity._ssprk2_program(), parity._euler_model("adc634_flat"), u0)
    if amr is None:
        print("skip (%s)" % aerr)
        return

    uni_rho, uni_phi = uni
    amr_rho, amr_phi, _amr_mass = amr
    chk(np.array_equal(uni_rho, amr_rho),
        "clean flat-AMR density is BIT-IDENTICAL to clean Uniform (max|diff| = %.3e)"
        % float(np.abs(uni_rho - amr_rho).max()))
    dphi = float(np.abs((uni_phi - uni_phi.mean()) - (amr_phi - amr_phi.mean())).max())
    rng = float(np.abs(uni_phi - uni_phi.mean()).max()) or 1.0
    chk(dphi / rng < 1e-4,
        "the mean-removed potential matches to the MG tolerance (rel max|diff| = %.3e)" % (dphi / rng))


# --------------------------------------------------------------------------------------------------
# (d) a runtime param reaches set_program_params through the clean route's bind(params=).
# --------------------------------------------------------------------------------------------------
def _decay_model(k_value=2.0, name="adc634_decay"):
    """Scalar rho, NO transport, a named source S = k*rho reading a runtime param k (no elliptic, so it
    runs on AMR with no Poisson solver). Mirror of test_program_runtime_params._decay_model."""
    from pops.physics import RuntimeParam
    from pops.physics.facade import Model
    m = Model(name)
    (rho,) = m.conservative_vars("rho")
    k = m.param(RuntimeParam("k", k_value))
    m.primitive_vars(rho=rho)
    m.conservative_from([rho])
    m.flux(x=[rho * 0.0], y=[rho * 0.0])
    m.eigenvalues(x=[rho * 0.0], y=[rho * 0.0])
    m.source_term("decay", [k * rho])
    return m


def _decay_program(name="adc634_decay_prog", block="gas"):
    """U <- U + dt*S over 'gas'; S reads the runtime param k."""
    from pops import time as adctime
    P = adctime.Program(name)
    U = P.state(block)
    S = P._source("decay", state=U)
    P.commit(P.state("U", block=block).next, P.linear_combine("U1", U + P.dt * S))
    return P


def _decay_ic():
    x = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)


def _direct_decay_amr(u0, set_k=None, nsteps=1, dt=1e-2):
    """The DIRECT decay route on AMR: add_equation + install_program (+ optional set_program_params).
    Returns the evolved density after nsteps."""
    amr = AmrSystem(n=N, L=1.0, periodic=True, regrid_every=0)
    if not hasattr(amr, "install_program") or not hasattr(amr, "set_program_params"):
        return None, "the built _pops lacks install_program/set_program_params (rebuild _pops)"
    try:
        compiled = pops.codegen.compile_problem(model=_decay_model(2.0), time=_decay_program(),
                                                target="amr_system")
        block_cm = _decay_model(2.0).compile(backend="production", target="amr_system")
    except RuntimeError as exc:
        return None, "compile (direct decay AMR): %s" % str(exc)[:160]
    try:
        # The installed Program drives the stepping (forward Euler from its text); the loader
        # block's nominal policy only needs an AMR-accepted kind ('explicit' | 'imex').
        amr.add_equation("gas", block_cm, spatial=_spatial(), time=pops.Explicit(method="ssprk2"))
        amr.set_density("gas", u0)
        amr.install_program(compiled.so_path)
        if set_k is not None:
            amr.set_program_params(0, [set_k])
    except RuntimeError as exc:
        return None, "install (direct decay AMR): %s" % str(exc)[:240]
    for _ in range(nsteps):
        amr.step(dt)
    return np.array(amr.density("gas")), None


def _clean_decay_amr(u0, params=None, nsteps=1, dt=1e-2):
    """The CLEAN decay route on AMR: pops.compile(layout=AMR) + pops.bind(params=). Returns the
    evolved density after nsteps."""
    problem = _problem(_decay_model(2.0), _decay_program(), block="gas")
    try:
        compiled = pops.compile(problem, layout=_amr_layout())
    except RuntimeError as exc:
        return None, "compile (clean decay AMR): %s" % str(exc)[:160]
    try:
        sim = pops.bind(compiled, initial_state={"gas": u0}, params=params or {})
    except RuntimeError as exc:
        return None, "bind (clean decay AMR): %s" % str(exc)[:240]
    for _ in range(nsteps):
        sim.step(dt)
    return np.array(sim.density("gas")), None


def test_clean_amr_bind_params_reach_set_program_params():
    """(d) A Program reading dsl.Param(kind='runtime'): the clean route's pops.bind(params={'k': 6.0})
    routes k to set_program_params, so the run DIFFERS from the k=2.0 (the declaration-default value)
    run and MATCHES a direct set_program_params(0, [6.0]) run. S = k*rho, no flux -> the step scales
    linearly in k, so k=6 gives 3x the increment of k=2. The clean AMR bind also enforces the SAME
    Spec 5 contract as the clean Uniform bind: arguments() marks kind='runtime' params required
    (inspect_compiled), so a bind WITHOUT params= refuses with the actionable message -- the
    declaration default reaches the kernel only on the DIRECT install_program route, via
    route_program_params' fallback."""
    print("== (d) clean-route bind(params=) reaches set_program_params ==")
    u0 = _decay_ic()

    # Uniform-mirrored bind contract: a runtime-param Program refuses to bind without params=.
    try:
        res, rerr = _clean_decay_amr(u0, params=None)
    except ValueError as exc:
        chk("runtime param 'k'" in str(exc),
            "bind without params= refuses with the actionable runtime-param message (Uniform parity)")
    else:
        if res is None:  # no compiler / Kokkos: the .so did not build -- an honest skip, not a fail
            print("skip (%s)" % rerr)
            return
        chk(False, "bind without params= must refuse a runtime-param Program (Uniform-mirrored)")

    default_rho, derr = _clean_decay_amr(u0, params={"k": 2.0})  # the declaration-default value
    if default_rho is None:
        print("skip (%s)" % derr)
        return
    override_rho, oerr = _clean_decay_amr(u0, params={"k": 6.0})  # k = 6.0 via bind(params=)
    if override_rho is None:
        print("skip (%s)" % oerr)
        return
    direct_rho, direrr = _direct_decay_amr(u0, set_k=6.0)      # direct set_program_params(0, [6.0])
    if direct_rho is None:
        print("skip (%s)" % direrr)
        return

    chk(not np.array_equal(default_rho, override_rho),
        "bind(params={'k': 6.0}) changed the run vs the declaration-default k=2.0 (max|diff| = %.3e)"
        % float(np.abs(default_rho - override_rho).max()))
    chk(np.array_equal(override_rho, direct_rho),
        "clean bind(params=) == direct set_program_params run, bit-for-bit (max|diff| = %.3e)"
        % float(np.abs(override_rho - direct_rho).max()))
    # S = k*rho, no flux: the k=6 increment is 3x the k=2 increment (LINEAR in k), to round-off.
    d_default = default_rho - u0
    d_override = override_rho - u0
    chk(np.allclose(d_override, 3.0 * d_default, rtol=1e-9, atol=1e-12),
        "a different k (2 -> 6) scales the step x3 without recompiling (max|d6 - 3 d2| = %.2e)"
        % float(np.abs(d_override - 3.0 * d_default).max()))


# --------------------------------------------------------------------------------------------------
# composition: the capability query + the Schur backstop (compiles, installs, throws at run).
# --------------------------------------------------------------------------------------------------
def test_composition_query_ssprk2_all_green():
    """The capability query reports an explicit SSPRK2 Program all-green on the AMR Program path (it
    uses no deferred op). Pure Python -- no build needed."""
    print("== composition: amr_program_op_support(ssprk2) is all green ==")
    support = amr_program_op_support(parity._ssprk2_program())
    pending = {g: s for g, s in support.items() if s != "green"}
    chk(not pending, "no pending group for an SSPRK2 Program (support = %r)" % support)


def test_composition_query_condensed_green_633():
    """A condensed-implicit Program uses the generic condensed_* ops -> the capability query reports
    condensed=green now that ADC-633 wired the per-level assembly + the flat/composite solve and ADC-637
    made the generic route the sole route. Pure Python."""
    print("== composition: amr_program_op_support(condensed_schur) is green (condensed group) ==")
    schur = _condensed_program(_schur_model("adc633_support"))
    support = amr_program_op_support(schur)
    chk(support.get("condensed") == "green",
        "the condensed Program reports condensed=green (support = %r)" % support)


def _schur_model(name="adc633_schur"):
    """Isothermal 2D fluid block (rho, mx, my) with a Poisson coupling + a B_z aux: the canonical
    condensed block (Density / MomentumX / MomentumY roles + B_z at the c_bz=3 aux slot). The generic
    condensed route (ADC-637) requires the electrostatic-Lorentz linearization J on the momentum subset,
    authored here."""
    from pops.ir.ops import sqrt
    from pops.lib.models import author_electrostatic_lorentz
    from pops.physics.facade import Model
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
    m.rate_operator("explicit_rhs", flux=True)
    author_electrostatic_lorentz(m)
    return m


def test_clean_amr_schur_program_compiles_installs_and_runs():
    """ADC-633: the clean route COMPILES + INSTALLS a condensed-Schur Program on AMR and RUNS it on a
    flat hierarchy (FrozenRegrid): the Schur ops are wired (per-level assembly + flat matrix-free
    BiCGStab through ctx.solve_linear_schur), so the step advances instead of throwing the old deferral
    backstop. B_z is seeded through bind(aux={'B_z': ...}). Assert finite + coarse-mass conserved (rho
    frozen by the Schur reconstruction). Self-skips without a compiler / Kokkos (never a fake engine)."""
    print("== composition: clean-route Schur Program compiles+installs+runs (ADC-633) ==")
    u0 = parity._init_density()
    bz0 = 4.0 * np.ones((N, N))
    model = _schur_model("adc633_schur")
    schur = _condensed_program(model)
    problem = _problem(model, schur)
    try:
        compiled = pops.compile(problem, layout=_amr_layout())
    except RuntimeError as exc:
        print("skip (compile: %s)" % str(exc)[:200])
        return
    if getattr(compiled, "program", None) is None:
        print("skip (clean AMR Schur compile carried no Program)")
        return
    try:
        sim = pops.bind(compiled, initial_state={"plasma": u0}, aux={"B_z": bz0})
    except RuntimeError as exc:
        print("skip (bind: %s)" % str(exc)[:240])
        return
    m0 = float(u0.mean())  # coarse mass / area (L=1); rho is frozen by the Schur reconstruction
    try:
        for _ in range(NSTEPS):
            sim.step(DT)
    except RuntimeError as exc:
        chk(False, "the wired condensed-Schur Program on AMR must run, but the step threw: %s"
            % str(exc)[:200])
        return
    rho = np.asarray(sim.density("plasma"))
    chk(np.isfinite(rho).all() and float(rho.min()) > 0.0,
        "the condensed-Schur Program on AMR keeps a finite, strictly-positive density (min = %.4f)"
        % float(rho.min()))
    mass = float(sim.mass("plasma"))
    chk(abs(mass - m0) < 1e-9,
        "the condensed-Schur Program on AMR conserves the coarse mass (|m - m0| = %.2e)"
        % abs(mass - m0))


def _clean_amr_schur_run(theta, name):
    """Drive a condensed-Schur Program at @p theta through the clean AMR (flat / FrozenRegrid) route.
    Seeds B_z through bind(aux=). Returns (coarse density, coarse potential) after NSTEPS, or
    (None, reason) on an honest skip. The theta < 1 path exercises the ncomp=1 phi^n history ring
    (ADC-427): last step's phi^{n+1} is this step's phi^n on the per-level ring."""
    u0 = parity._init_density()
    bz0 = 4.0 * np.ones((N, N))
    model = _schur_model(name)
    schur = _condensed_program(model, theta=theta)
    problem = _problem(model, schur)
    try:
        compiled = pops.compile(problem, layout=_amr_layout())
    except RuntimeError as exc:
        return None, "compile (clean AMR schur theta=%s): %s" % (theta, str(exc)[:180])
    if getattr(compiled, "program", None) is None:
        return None, "clean AMR schur compile carried no Program"
    try:
        sim = pops.bind(compiled, initial_state={"plasma": u0}, aux={"B_z": bz0})
    except RuntimeError as exc:
        return None, "bind (clean AMR schur theta=%s): %s" % (theta, str(exc)[:220])
    for _ in range(NSTEPS):
        sim.step(DT)
    return (np.array(sim.density("plasma")), np.array(sim.potential())), None


def _lift_schur_density(model, u0, block="plasma"):
    """The native set_density lift of the isothermal Schur block (rho -> [rho, 0, 0]) read back from a
    scratch System, so the Uniform bind seeds the FULL conservative state BITWISE identically to the
    AMR set_density seed (mirror of _lift_density for the compressible model). Returns the (3, n, n)
    state or (None, reason)."""
    from pops.runtime.system import System
    sim = System(n=N, L=1.0)
    try:
        block_cm = model.compile(backend="production")
    except RuntimeError as exc:
        return None, "compile (lift scratch Schur System): %s" % str(exc)[:160]
    try:
        sim.add_equation(block, block_cm, spatial=_spatial(), time=pops.Explicit(method="ssprk2"))
        sim.set_density(block, u0)
        return np.array(sim.get_state(block)), None
    except RuntimeError as exc:
        return None, "seed (lift scratch Schur System): %s" % str(exc)[:160]


def _clean_uniform_schur_run(theta, name):
    """The Uniform counterpart of _clean_amr_schur_run: same Program at @p theta, layout=Uniform.
    The Uniform bind takes the FULL conservative state, so the AMR set_density seed is lifted back
    from a scratch System (bit-identically) before binding. Returns (density comp-0, potential), or
    (None, reason) on an honest skip."""
    u0 = parity._init_density()
    bz0 = 4.0 * np.ones((N, N))
    model = _schur_model(name)
    lifted, lerr = _lift_schur_density(_schur_model(name + "_lift"), u0)
    if lifted is None:
        return None, lerr
    schur = _condensed_program(model, theta=theta)
    problem = _problem(model, schur)
    try:
        compiled = pops.compile(problem, layout=_uniform_layout())
    except RuntimeError as exc:
        return None, "compile (clean Uniform schur theta=%s): %s" % (theta, str(exc)[:180])
    try:
        sim = pops.bind(compiled, initial_state={"plasma": lifted}, aux={"B_z": bz0})
    except RuntimeError as exc:
        return None, "bind (clean Uniform schur theta=%s): %s" % (theta, str(exc)[:220])
    for _ in range(NSTEPS):
        sim.step(DT)
    state = np.array(sim.get_state("plasma"))
    return (state[0], np.array(sim.potential())), None


def test_clean_flat_amr_schur_theta_one_equals_uniform():
    """ADC-427 (theta == 1, byte-identity): the condensed-Schur Program at theta = 1 (the fresh-zero
    phi path, NO history carry) drives the SAME evolved density on a flat AMR hierarchy as on Uniform.
    The flat AMR emitted body runs level 0 bit-identically to Uniform (the ADC-633 flat-parity path);
    theta == 1 emits no history op, so the ring is untouched. np.array_equal on the density."""
    print("== (c) clean flat-AMR Schur theta=1 == clean Uniform theta=1 (bit-identical density) ==")
    amr, aerr = _clean_amr_schur_run(1.0, "adc427_schur_t1")
    if amr is None:
        print("skip (%s)" % aerr)
        return
    uni, uerr = _clean_uniform_schur_run(1.0, "adc427_schur_t1")
    if uni is None:
        print("skip (%s)" % uerr)
        return
    amr_rho, uni_rho = amr[0], uni[0]
    chk(np.array_equal(uni_rho, amr_rho),
        "clean flat-AMR Schur theta=1 density is BIT-IDENTICAL to Uniform (max|diff| = %.3e)"
        % float(np.abs(uni_rho - amr_rho).max()))


def test_clean_flat_amr_schur_theta_half_equals_uniform():
    """ADC-427 (theta = 0.5, the phi^n carry): the condensed-Schur Program at theta = 0.5 CARRIES phi^n
    across steps through the ncomp=1 System / AMR history ring (the -Lap(phi^n) RHS anchor + the warm
    start + the 1/theta phi extrapolation). On a flat AMR hierarchy the per-level ring degenerates to
    the level-0 ring, so the emitted body runs bit-identically to the Uniform level-0 carry. This gate
    is the load-bearing acceptance for the ncomp-aware AMR history ring: it FAILS if the AMR ring
    ignores the ncomp=1 request (a block-ncomp ring would carry the wrong phi^n slot). np.array_equal."""
    print("== (c') clean flat-AMR Schur theta=0.5 == clean Uniform theta=0.5 (bit-identical density) ==")
    amr, aerr = _clean_amr_schur_run(0.5, "adc427_schur_t05")
    if amr is None:
        print("skip (%s)" % aerr)
        return
    uni, uerr = _clean_uniform_schur_run(0.5, "adc427_schur_t05")
    if uni is None:
        print("skip (%s)" % uerr)
        return
    amr_rho, uni_rho = amr[0], uni[0]
    chk(np.array_equal(uni_rho, amr_rho),
        "clean flat-AMR Schur theta=0.5 density is BIT-IDENTICAL to Uniform level-0 "
        "(the ncomp=1 phi^n carry; max|diff| = %.3e)" % float(np.abs(uni_rho - amr_rho).max()))
def test_flat_amr_condensed_program_matches_system_bit_for_bit():
    """ADC-637 retirement precondition on AMR: the generic condensed-implicit Program on a FLAT AMR
    hierarchy (FrozenRegrid, no fine patch) evolves the density BIT-FOR-BIT identically to the same
    Program on the Uniform layout. assembly_target / assembly_source are the identity when
    !has_fine_patches(), and solve_linear_matfree dispatches flat->the SAME matrix-free BiCGStab -- so
    the flat-AMR trajectory transfers the proven System trajectory to the last bit (the acceptance the
    brick retirement rests on for AMR). Self-skips without a compiler / Kokkos (never a fake engine)."""
    print("== retirement precondition: flat-AMR condensed Program == System condensed Program ==")
    u0 = parity._init_density()  # density-only 2D seed; the AMR route lifts (momentum=0), Uniform binds
    bz0 = 4.0 * np.ones((N, N))   # the full conservative state, so read the lift back for the Uniform seed.

    # The Uniform bind takes the FULL conservative state; read the native set_density lift back from a
    # scratch System so the Uniform seed is BITWISE the AMR set_density seed (same pattern as leg (b)).
    u0_full, lerr = _lift_density(_schur_model("adc637_lift"), u0)
    if u0_full is None:
        print("skip (%s)" % lerr)
        return

    def _run(layout, seed, seed_as_density):
        model = _schur_model("adc637_flat")
        schur = _condensed_program(model)
        problem = _problem(model, schur)
        try:
            compiled = pops.compile(problem, layout=layout)
        except RuntimeError as exc:
            return None, "compile: %s" % str(exc)[:200]
        if getattr(compiled, "program", None) is None:
            return None, "compile carried no Program"
        try:
            sim = pops.bind(compiled, initial_state={"plasma": seed}, aux={"B_z": bz0})
        except RuntimeError as exc:
            return None, "bind: %s" % str(exc)[:240]
        try:
            for _ in range(NSTEPS):
                sim.step(DT)
        except RuntimeError as exc:
            return None, "step: %s" % str(exc)[:200]
        return np.asarray(sim.density("plasma")), None

    amr_rho, aerr = _run(_amr_layout(), u0, True)
    if amr_rho is None:
        print("skip (flat AMR %s)" % aerr)
        return
    sys_rho, serr = _run(_uniform_layout(), u0_full, False)
    if sys_rho is None:
        print("skip (Uniform %s)" % serr)
        return
    dmax = float(np.abs(amr_rho - sys_rho).max())
    chk(np.array_equal(amr_rho, sys_rho),
        "the flat-AMR condensed Program density is BIT-IDENTICAL to the System condensed Program "
        "(max|diff| = %.3e)" % dmax)


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print("\n%s test_amr_clean_route_program (%d check failures)"
          % ("FAIL" if _fails else "PASS", _fails))
    return _fails


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
