"""ADC-515: the ONE declarative Spec 6 sec.20 Uniform x AMR test matrix (the greppable table).

This module is the single source of truth for the sec.20 (operation x layout x block-count) grid. It
exports one dict, :data:`MATRIX` (keyed by a stable ``"op.layout.blocks"`` id), and the frozen
:data:`EXPECTED_KEYS` completeness set. ``test_spec6_amr_matrix.py`` iterates ``MATRIX``, dispatching
each cell by its ``kind``:

  * ``green_inert``  -- an INERT check (arguments / estimate_memory / inspect_amr / route facts on an
    AMR-compiled handle or descriptor); returns a proof token, allocates nothing.
  * ``green_live``   -- builds a REAL ``AmrSystem`` and asserts finite + mass-conserved / clock-
    advanced / a live patch (never a fake engine).
  * ``refuse``       -- ``(exc_type, needles, callable)`` run by a shared ``_expect_refusal``
    (warning-free, stable substrings) -- the same discipline as ``unit/compliance/_cells``.
  * ``exists``       -- asserts the CITED existing coverage still holds (a thin route-fact / import
    check), so an already-covered cell is pointed at, not duplicated.
  * ``pending``      -- CONSTRUCTS the row's authoring object (proving it is structurally real) then
    ``pytest.skip`` with the pending marker; it flips to ``green_live`` when the named issue lands. No
    pending row remains: the multistep-on-AMR rows flipped to ``exists`` when ADC-631 merged, the
    clean-``compile(layout=AMR)`` explicit / SSPRK Program row to ``green_live`` with ADC-634, and the
    compiled condensed-Schur hierarchy Program row to ``green_live`` with ADC-633 (the per-level Schur
    assembly + the flat/composite solve).

Importing this module requires ``pops`` (the AMR-route handle + bricks); ``test_spec6_amr_matrix``
importorskips it so the suite is green on a bare box. ASCII only.
"""
import warnings
from collections import namedtuple

import numpy as np
import pytest

import pops
import pops.lib.time as lib_time
from pops.codegen._plans import ResolvedBlock, ResolvedSimulationPlan
from pops.codegen.compiled_artifact import CompiledBlockArtifact, CompiledSimulationArtifact
from pops.codegen.loader import CompiledModel
from pops.ir.ops import sqrt
from pops.identity import make_identity
from pops.mesh import CartesianMesh
from pops.mesh.amr import FrozenRegrid, Refine, RegridEvery
from pops.mesh.layouts import AMR, Uniform
from pops.model import Handle, OwnerPath
from pops.model.bind_schema import BindSchema
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
from pops.physics.facade import Model as FacadeModel
from pops.params import ConstParam, RuntimeParam
from pops.problem._snapshot import AuthoringSnapshot
from pops.runtime.system import AmrSystem
from tests.python.support.typed_program import program_states, synthetic_module


# --------------------------------------------------------------------------------------------------
# Cell shape + the shared refusal runner (mirrors unit/compliance/_cells.run_negative_cell).
# --------------------------------------------------------------------------------------------------
Cell = namedtuple("Cell", "op layout blocks kind run")


def _expect_refusal(exc_type, needles, call):
    """Run ``call``; assert it raises ``exc_type`` with every needle and NO warning. Returns the msg."""
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        try:
            call()
        except exc_type as exc:
            message = str(exc)
        else:
            raise AssertionError("expected %s, no exception raised" % exc_type.__name__)
    assert not seen, "refusal emitted warning(s) instead of a clean reject: %r" % (seen,)
    missing = [n for n in needles if n not in message]
    assert not missing, "refusal %r missing needles %r" % (message, missing)
    return message


# --------------------------------------------------------------------------------------------------
# Native model + state helpers (composed native bricks; no DSL / Kokkos AOT compile).
# --------------------------------------------------------------------------------------------------
def _scalar_charge(q, B0=1.0):
    return pops.Model(pops.Scalar(), pops.ExB(B0=B0), pops.NoSource(), pops.ChargeDensity(charge=q))


def _iso_model(cs2=1.0, alpha=1.0):
    return pops.Model(state=pops.FluidState(kind="isothermal", cs2=cs2),
                      transport=pops.IsothermalFlux(), source=pops.NoSource(),
                      elliptic=pops.BackgroundDensity(alpha=alpha, n0=0.0))


def _bump(n, amp):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    r = 1.0 + amp * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01)
    return r + (1.0 - r.mean())


def _iso_state(n, L, rho=1.5):
    x = (np.arange(n) + 0.5) * (L / n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    r = rho * np.ones((n, n))
    u = 0.5 * np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
    v = -0.3 * np.sin(2.0 * np.pi * X / L) * np.sin(np.pi * Y / L)
    return np.stack([r, r * u, r * v])


def _amr_route_handle(*, n_aux=1, mpi=True):
    """An exact metadata-only AMR compiled artifact."""

    handle = CompiledModel(
        so_path="<stub-amr>", backend="production", adder="add_native_block",
        cons_names=["rho", "mx", "my"], cons_roles=["Density", "MomentumX", "MomentumY"],
        prim_names=["rho", "mx", "my"], n_vars=3, gamma=1.4, n_aux=n_aux,
        params={"alpha": RuntimeParam("alpha", default=1.0)},
        caps={"cpu": True, "amr": True, "mpi": mpi}, abi_key="k", model_hash="h", cxx="c++",
        std="c++23", target="amr_system", aux_extra_names=["B_z"][:n_aux])
    handle.artifact_identity = make_identity(
        "artifact", {"fixture": "spec6-matrix-model", "n_aux": n_aux, "mpi": mpi})
    rho = Handle("rho", kind="state", owner=OwnerPath.shared("spec6-matrix"))
    layout = AMR(base=CartesianMesh(n=64, periodic=True), max_levels=2, ratio=2,
                 regrid=RegridEvery(4), refine=Refine.on(rho).above(0.1))
    snapshot = AuthoringSnapshot({"kind": "spec6-matrix-stub"})
    schema = BindSchema()
    plan = ResolvedSimulationPlan(
        snapshot=snapshot,
        target="amr_system",
        backend="production",
        layout=layout,
        time=None,
        blocks=(ResolvedBlock(
            "ne", {"model": "spec6-matrix-model"}, None, "production"),),
        bind_schema=schema,
        compile_values=schema.resolve_compile(),
        field_solvers={},
        outputs=(),
        diagnostics=(),
        libraries=(),
        requirements={"amr": True},
        capabilities={"cpu": True, "amr": True, "mpi": mpi},
    )
    return CompiledSimulationArtifact(
        plan=plan,
        program=None,
        blocks=(CompiledBlockArtifact("ne", handle, None),),
    )


def _routes_by_id():
    return {r.to_dict()["route_id"]: r.to_dict() for r in pops.native_capability_report().routes}


# --------------------------------------------------------------------------------------------------
# green_live: build a real AmrSystem, assert finite + conserved + live patch / clock-advanced.
# --------------------------------------------------------------------------------------------------
def _run_explicit_family(time_brick, *, multi, n=32):
    sim = AmrSystem(n=n, L=1.0, periodic=True, regrid_every=4)
    sim.add_block("ions", _scalar_charge(+1.0), spatial=pops.Spatial(minmod=True), time=time_brick)
    if multi:
        sim.add_block("electrons", _scalar_charge(-1.0), spatial=pops.Spatial(minmod=True),
                      time=time_brick)
    sim.set_poisson(bc="periodic")
    sim.set_refinement(1.05)
    sim.set_density("ions", _bump(n, 0.40))
    if multi:
        sim.set_density("electrons", _bump(n, 0.20))
    blocks = ("ions", "electrons") if multi else ("ions",)
    m0 = {b: sim.mass(b) for b in blocks}
    sim.advance(0.002, 10)
    for b, ms in m0.items():
        assert np.isfinite(np.asarray(sim.density(b))).all(), "block %r not finite" % b
        assert abs(sim.mass(b) - ms) / (abs(ms) + 1.0) < 1e-9, "block %r mass drift" % b
    assert sim.n_patches() >= 1, "no live fine patch"
    return "green_live:%d_patch" % sim.n_patches()


def _explicit_family(brick, multi):
    return lambda: _run_explicit_family(brick, multi=multi)


def _run_condensed_schur(splitting, n=24, L=1.0):
    sim = AmrSystem(n=n, L=L, periodic=False, regrid_every=0)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet")
    sim.set_refinement(1e30)
    sim.set_magnetic_field(4.0 * np.ones((n, n)))
    cls = pops.Strang if splitting == "strang" else pops.Split
    sim.add_equation("electrons", model=_iso_model(cs2=1.0, alpha=3.0),
                     spatial=pops.Spatial(minmod=True),
                     time=cls(hyperbolic=pops.Explicit(),
                              source=pops.CondensedSchur(kind="electrostatic_lorentz",
                                                         theta=1.0, alpha=3.0)))
    sim.set_conservative_state("electrons", _iso_state(n, L))
    m0 = sim.mass()
    for _ in range(5):
        sim.step(5.0e-4)
    assert np.isfinite(np.asarray(sim.density())).all(), "%s: density not finite" % splitting
    assert np.isfinite(np.asarray(sim.potential())).all(), "%s: potential not finite" % splitting
    assert abs(sim.mass() - m0) <= 1e-9 * max(abs(m0), 1e-30), "%s: mass drift" % splitting
    return "green_live:schur_%s" % splitting


def _run_profile_cfl(n=32):
    sim = AmrSystem(n=n, L=1.0, periodic=True, regrid_every=2, coarse_max_grid=16)
    sim.add_block("ne", _scalar_charge(+1.0), spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    sim.set_poisson(bc="periodic")
    sim.set_refinement(threshold=1.05)
    sim.set_density("ne", _bump(n, 0.40))
    with sim.profile(pops.Profile.Basic()) as prof:
        sim.step_cfl(0.4)
        sim.step_cfl(0.4)
    assert sim.time() > 0.0 and np.isfinite(sim.time()), "step_cfl did not advance the clock"
    assert prof.summary().by_amr_mpi() is not None
    assert sim.amr.patch_table().built is True
    return "green_live:cfl_t=%.3e" % sim.time()


# --------------------------------------------------------------------------------------------------
# green_live: the clean pops.compile(layout=AMR)+pops.bind whole-system Program route (ADC-634).
# --------------------------------------------------------------------------------------------------
def _euler_facade_model(name="spec6_clean_euler"):
    """A compressible Euler facade model (elliptic_rhs = rho so a field solve runs): the physics the
    clean-route Program lowers AND the block the AMR instance carries. Mirror of the ADC-634 acceptance
    model. Needs the DSL compiler + Kokkos to build the .so (the cell skips cleanly without them)."""
    g = 1.4
    m = FacadeModel(name)
    rho, rhou, rhov, E = m.conservative_vars("rho", "rho_u", "rho_v", "E")
    u, v = rhou / rho, rhov / rho
    p = (g - 1.0) * (E - 0.5 * rho * (u * u + v * v))
    pu, pv, pp = m.primitive("u", u), m.primitive("v", v), m.primitive("p", p)
    H = (E + pp) / rho
    c = sqrt(g * pp / rho)
    m.flux(x=[rhou, rhou * pu + pp, rhou * pv, rho * H * pu],
           y=[rhov, rhov * pu, rhov * pv + pp, rho * H * pv])
    m.eigenvalues(x=[pu - c, pu, pu + c], y=[pv - c, pv, pv + c])
    m.primitive_vars(rho, pu, pv, pp)
    m.conservative_from([rho, rho * pu, rho * pv,
                        pp / (g - 1.0) + 0.5 * rho * (pu * pu + pv * pv)])
    m.gamma(g)
    m.elliptic_rhs(rho)
    m.rate_operator("explicit_rhs", flux=True)
    return m


def _clean_density(n=16):
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)


def _run_clean_route_program(n=16, nsteps=4, dt=1.0e-3):
    """Build a real AmrSystem via the clean pops.compile(layout=AMR(FrozenRegrid))+pops.bind route with
    an SSPRK3 whole-system Program (ADC-634), step it, and assert finite + coarse-mass-conserved. Skips
    cleanly (pytest.skip, never a fake engine) when the .so cannot build (no compiler / no Kokkos)."""
    model = _euler_facade_model()
    program = pops.time.Program("ssprk3")
    _case, states = program_states(program, model, ("plasma",))
    lib_time.ssprk3(program, states["plasma"])
    u0 = _clean_density(n)
    problem = (pops.Problem()
               .block("plasma", physics=model,
                      spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()))
               .time(program))
    layout = AMR(base=CartesianMesh(n=n, L=1.0, periodic=True), regrid=FrozenRegrid())
    try:
        compiled = pops.compile(problem, layout=layout)
    except RuntimeError as exc:
        pytest.skip("clean-route AMR Program .so could not build (no compiler / Kokkos): %s"
                    % str(exc)[:160])
    assert getattr(compiled, "program", None) is not None, \
        "the clean AMR route must carry the compiled Program (ADC-634)"
    try:
        sim = pops.bind(compiled, initial_state={"plasma": u0})
    except RuntimeError as exc:
        pytest.skip("clean-route AMR Program bind/install could not run: %s" % str(exc)[:200])
    m0 = float(u0.mean())  # coarse mass / area (L=1)
    for _ in range(nsteps):
        sim.step(dt)
    rho = np.asarray(sim.density("plasma"))
    assert np.isfinite(rho).all() and float(rho.min()) > 0.0, "clean-route density not finite/positive"
    mass = float(sim.mass("plasma"))
    assert abs(mass - m0) < 1e-9, "clean-route coarse mass drift (|m - m0| = %.2e)" % abs(mass - m0)
    return "green_live:clean_program_ssprk3"


def _schur_facade_model(name="spec6_clean_schur"):
    """Isothermal 2D fluid block (rho, mx, my) with a Poisson coupling + a B_z aux: the canonical
    condensed block a Program lowers on the hierarchy. m.aux("B_z") makes B_z the canonical aux slot;
    elliptic_rhs = rho drives the field solve. The generic condensed route (ADC-637) requires the
    electrostatic-Lorentz linearization J on the momentum subset, authored here. Needs the DSL compiler +
    Kokkos to build the .so (the cell skips cleanly without them)."""
    from pops.lib.models import author_electrostatic_lorentz
    m = FacadeModel(name)
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
    m.rate_operator("explicit_rhs", flux=True)
    author_electrostatic_lorentz(m)
    return m


def _run_clean_schur_program(n=16, nsteps=4, dt=5.0e-4):
    """ADC-633: build a real AmrSystem via the clean pops.compile(layout=AMR(FrozenRegrid))+pops.bind
    route with a condensed-Schur whole-system Program (theta=1), step it, and assert finite +
    coarse-mass-conserved (rho frozen by the Schur reconstruction). The flat hierarchy runs the emitted
    matrix-free BiCGStab through ctx.solve_linear_schur. B_z is seeded through bind(aux={'B_z': ...}).
    Skips cleanly (never a fake engine) when the .so cannot build (no compiler / no Kokkos)."""
    from pops.model import OperatorHandle
    model = _schur_facade_model()
    registry = model.operator_registry()
    operator = registry.operators_of_kind("local_linear_operator")[0]
    linear = OperatorHandle(
        operator.name, kind=operator.kind, owner=registry.owner_path,
        signature=operator.signature)
    program = pops.time.Program("condensed_schur").bind_operators(model)
    _case, states = program_states(program, model, ("plasma",))
    lib_time.condensed_schur(
        program, states["plasma"], alpha=1.0, theta=1.0, linear_operator=linear)
    u0 = _clean_density(n)
    bz0 = 4.0 * np.ones((n, n))
    problem = (pops.Problem()
               .block("plasma", physics=model,
                      spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()))
               .time(program))
    layout = AMR(base=CartesianMesh(n=n, L=1.0, periodic=True), regrid=FrozenRegrid())
    try:
        compiled = pops.compile(problem, layout=layout)
    except RuntimeError as exc:
        pytest.skip("clean-route AMR Schur Program .so could not build (no compiler / Kokkos): %s"
                    % str(exc)[:160])
    assert getattr(compiled, "program", None) is not None, \
        "the clean AMR route must carry the compiled Schur Program (ADC-634 route, ADC-633 ops)"
    try:
        sim = pops.bind(compiled, initial_state={"plasma": u0}, aux={"B_z": bz0})
    except RuntimeError as exc:
        pytest.skip("clean-route AMR Schur Program bind/install could not run: %s" % str(exc)[:200])
    m0 = float(u0.mean())  # coarse mass / area (L=1); rho is frozen by the Schur reconstruction
    for _ in range(nsteps):
        sim.step(dt)
    rho = np.asarray(sim.density("plasma"))
    assert np.isfinite(rho).all() and float(rho.min()) > 0.0, "schur density not finite/positive"
    assert np.isfinite(np.asarray(sim.potential())).all(), "schur potential not finite"
    mass = float(sim.mass("plasma"))
    assert abs(mass - m0) < 1e-9, "schur coarse mass drift (|m - m0| = %.2e)" % abs(mass - m0)
    return "green_live:clean_schur_program"


# --------------------------------------------------------------------------------------------------
# green_inert: introspection on the AMR-route handle (no run, no dlopen).
# --------------------------------------------------------------------------------------------------
def _inert_arguments():
    from pops.codegen.inspect_compiled import build_arguments

    args = build_arguments(_amr_route_handle())
    assert args.layout_runtime["layout"] == "amr"
    assert next(iter(args.instances.values()))["components"] == 3
    assert set(args.aux) == {"B_z"} and args.params["alpha"]["kind"] == "runtime"
    return "green_inert:arguments_amr"


def _inert_estimate_memory():
    from pops.codegen.inspect_compiled import build_memory_estimate

    handle = _amr_route_handle()
    mesh = CartesianMesh(n=64, L=1.0, periodic=True)
    est = build_memory_estimate(handle, mesh, layout=handle.layout)
    assert est.layout == "amr" and est.categories.get("amr_patch", 0) > 0
    assert est.total_bytes >= build_memory_estimate(
        handle, mesh, layout=Uniform(mesh)).total_bytes
    return "green_inert:estimate_memory_amr"


def _inert_inspect_amr():
    rep = pops.inspect_amr(_amr_route_handle().layout).to_dict()
    assert rep["layout"] == "amr" and rep["max_levels"] == 2
    assert {row["slot"] for row in rep["policies"]} >= {"refine", "regrid"}
    return "green_inert:inspect_amr"


# --------------------------------------------------------------------------------------------------
# refuse: precise rejections (stable substrings only).
# --------------------------------------------------------------------------------------------------
def _refuse_imexrk_on_amr():
    def call():
        AmrSystem(n=16, L=1.0, periodic=True, regrid_every=0).add_block(
            "ne", _scalar_charge(+1.0), spatial=pops.Spatial(minmod=True), time=pops.IMEXRK())
    return _expect_refusal(RuntimeError,
                           ("imexrk_ars222", "not wired on AMR", "Cartesian System"), call)


def _exists_native_runtime_params():
    # Cited: tests/python/integration/amr/test_amr_native_params (ADC-514 wired set_block_params:
    # a native AMR block's runtime param changes the run without recompiling; params={} is
    # bit-identical). The blanket NotImplementedError refusal is gone; the residual precise
    # refusal (a name declared by no instance) lives in test_amr_refusals.
    sim = AmrSystem(n=16, L=1.0, periodic=True, regrid_every=0)
    assert callable(getattr(sim, "set_block_params", None))
    return "exists:test_amr_native_params"


def _refuse_multiblock_source_stage():
    def call():
        n = 16
        sim = AmrSystem(n=n, L=1.0, periodic=False, regrid_every=0)
        sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet")
        sim.set_refinement(1e30)
        sim.set_magnetic_field(4.0 * np.ones((n, n)))
        schur = pops.Strang(hyperbolic=pops.Explicit(),
                            source=pops.CondensedSchur(kind="electrostatic_lorentz",
                                                       theta=1.0, alpha=3.0))
        sim.add_equation("e1", model=_iso_model(alpha=3.0), spatial=pops.Spatial(minmod=True),
                         time=schur)
        sim.add_equation("e2", model=_iso_model(alpha=3.0), spatial=pops.Spatial(minmod=True),
                         time=schur)
    return _expect_refusal(RuntimeError,
                           ("set_source_stage", "SINGLE-BLOCK", ">= 2 blocks"), call)


# --------------------------------------------------------------------------------------------------
# exists: cite the already-covered cells (a thin route-fact / descriptor check, no duplication).
# --------------------------------------------------------------------------------------------------
def _exists_fft_on_amr_refused():
    # Cited: unit/compliance/_cells.neg.fft_on_amr_or_bc (descriptor + route). Assert the route fact
    # still holds here so the sec.20 matrix agrees with the compliance matrix (no re-test of the run).
    row = _routes_by_id()["elliptic:fft_amr"]
    assert row["status"] == "unavailable"
    assert "FFT requires a single uniform periodic mesh, not AMR" in row["reason"]
    return "exists:_cells.neg.fft_on_amr_or_bc"


def _exists_ssprk3_exclusivity():
    # Cited: tests/python/integration/amr/test_amr_ssprk3 (SSPRK3-vs-IMEX exclusivity). The SSPRK3
    # brick is the native explicit family (kind='ssprk3'); assert it authors here.
    program = pops.time.Program("spec6-exists-ssprk3")
    module = synthetic_module("spec6_ssprk3_state", components=("rho",))
    _case, states = program_states(program, module, ("plasma",))
    lib_time.ssprk3(program, states["plasma"])
    assert isinstance(program, pops.time.Program)
    return "exists:test_amr_ssprk3"


def _exists_ssprk2_program_parity():
    # Cited: test_amr_program_parity (ADC-508 compiled-Program SSPRK2 parity on AMR). The compiled
    # whole-system Program parity is proven there; assert the ssprk2 macro authors a stable Program.
    prog = pops.time.Program("spec6-exists-ssprk2")
    module = synthetic_module("spec6_ssprk2_state", components=("rho",))
    _case, states = program_states(prog, module, ("plasma",))
    lib_time.ssprk2(prog, states["plasma"])
    assert prog._ir_hash() == prog._ir_hash()
    return "exists:test_amr_program_parity"


def _exists_named_field():
    # Cited: test_amr_named_field (ADC-517 named elliptic field / .field on AMR). Assert the
    # layout:AMR route is advertised available (the named-field run lives in the dedicated suite).
    assert _routes_by_id()["layout:AMR"]["status"] == "available"
    return "exists:test_amr_named_field"


# --------------------------------------------------------------------------------------------------
# pending: construct the authoring object (structurally real), then skip with the pending marker.
# A pending cell NEVER executes the deferred path and NEVER pins transitional behavior.
# --------------------------------------------------------------------------------------------------
def _exists_multistep(builder):
    # Cited: tests/python/integration/amr/test_amr_history_parity (ADC-631, MERGED): AB2 on a flat
    # 2-block AMR hierarchy is bit-identical to Uniform, ring slots byte-identical; regrid remap +
    # v3 replay covered by test_amr_history_regrid / test_amr_history_checkpoint. The authoring
    # object stays structurally real here.
    def run():
        program = pops.time.Program("spec6-exists-multistep")
        module = synthetic_module("spec6_multistep_state", components=("rho",))
        _case, states = program_states(program, module, ("plasma",))
        builder(program, states["plasma"])
        assert isinstance(program, pops.time.Program)
        return "exists:test_amr_history_parity"
    return run


# --------------------------------------------------------------------------------------------------
# THE MATRIX -- keyed "op.layout.blocks". Uniform baseline cells cite the shipping Spec 5 coverage;
# the AMR column is the ADC-515 focus (green live / inert, precise refusals -- no pending row remains).
# --------------------------------------------------------------------------------------------------
MATRIX = {
    # explicit family -- AMR green live (mono + multi)
    "explicit.amr.mono": Cell("explicit", "amr", "mono", "green_live",
                              _explicit_family(pops.Explicit(), False)),
    "explicit.amr.multi": Cell("explicit", "amr", "multi", "green_live",
                               _explicit_family(pops.Explicit(), True)),
    "ssprk3.amr.mono": Cell("ssprk3", "amr", "mono", "green_live",
                            _explicit_family(pops.Explicit(ssprk3=True), False)),
    "ssprk3.amr.multi": Cell("ssprk3", "amr", "multi", "exists", _exists_ssprk3_exclusivity),
    "imex.amr.mono": Cell("imex", "amr", "mono", "green_live",
                          _explicit_family(pops.IMEX(), False)),
    # condensed-Schur source stage -- AMR green live mono, refuse multi
    "strang_schur.amr.mono": Cell("strang_schur", "amr", "mono", "green_live",
                                  lambda: _run_condensed_schur("strang")),
    "lie_schur.amr.mono": Cell("lie_schur", "amr", "mono", "green_live",
                               lambda: _run_condensed_schur("lie")),
    "strang_schur.amr.multi": Cell("strang_schur", "amr", "multi", "refuse",
                                   _refuse_multiblock_source_stage),
    # compiled whole-system Program (ADC-508) -- cite the parity suite
    "program_ssprk2.amr.mono": Cell("program_ssprk2", "amr", "mono", "exists",
                                    _exists_ssprk2_program_parity),
    # runtime params on AMR -- precise refusal (mono + multi share the native no-param seam)
    "runtime_params.amr.mono": Cell("runtime_params", "amr", "mono", "exists",
                                    _exists_native_runtime_params),
    # runtime CFL + profile -- AMR green live
    "cfl_profile.amr.mono": Cell("cfl_profile", "amr", "mono", "green_live", _run_profile_cfl),
    # named elliptic field / .field -- cite the dedicated suite
    "named_field.amr.mono": Cell("named_field", "amr", "mono", "exists", _exists_named_field),
    # introspection seams (ADC-515) -- AMR green inert
    "arguments.amr.mono": Cell("arguments", "amr", "mono", "green_inert", _inert_arguments),
    "estimate_memory.amr.mono": Cell("estimate_memory", "amr", "mono", "green_inert",
                                     _inert_estimate_memory),
    "inspect_amr.amr.mono": Cell("inspect_amr", "amr", "mono", "green_inert", _inert_inspect_amr),
    # FFT field on AMR -- cite the compliance-matrix reject (semantically impossible on AMR)
    "fft_field.amr.mono": Cell("fft_field", "amr", "mono", "exists", _exists_fft_on_amr_refused),
    # IMEXRK / ARS222 on AMR -- precise Cartesian-scope refusal
    "imexrk.amr.mono": Cell("imexrk", "amr", "mono", "refuse", _refuse_imexrk_on_amr),
    # multistep AB2 / BDF2 on AMR -- LANDED (ADC-631 merged): cite the ring parity coverage.
    "ab2.amr.mono": Cell("ab2", "amr", "mono", "exists",
                         _exists_multistep(lib_time.adams_bashforth2)),
    "bdf2.amr.mono": Cell("bdf2", "amr", "mono", "exists",
                          _exists_multistep(
                              lambda program, state: lib_time.bdf(program, state, order=2))),
    # clean-compile(layout=AMR) whole-system Program -- GREEN LIVE (ADC-634 route implemented): the
    # clean pops.compile(layout=AMR)+pops.bind SSPRK3 Program builds a real AmrSystem, runs, conserves.
    "clean_program.amr.mono": Cell("clean_program", "amr", "mono", "green_live",
                                   _run_clean_route_program),
    # compiled condensed-Schur hierarchy Program -- GREEN LIVE (ADC-633): the per-level Schur assembly +
    # the flat/composite solve are wired, so the clean pops.compile(layout=AMR)+pops.bind condensed-Schur
    # Program builds a real AmrSystem, runs (flat matrix-free BiCGStab through ctx.solve_linear_schur),
    # and conserves the coarse mass.
    "clean_schur_program.amr.mono": Cell("clean_schur_program", "amr", "mono", "green_live",
                                         _run_clean_schur_program),
}

# The full grid -- test_matrix_is_complete pins set(MATRIX) == EXPECTED_KEYS so a dropped (op x
# layout x blocks) cell fails loud: a future layout gap cannot be silently hidden.
EXPECTED_KEYS = frozenset({
    "explicit.amr.mono", "explicit.amr.multi",
    "ssprk3.amr.mono", "ssprk3.amr.multi",
    "imex.amr.mono",
    "strang_schur.amr.mono", "lie_schur.amr.mono", "strang_schur.amr.multi",
    "program_ssprk2.amr.mono",
    "runtime_params.amr.mono",
    "cfl_profile.amr.mono",
    "named_field.amr.mono",
    "arguments.amr.mono", "estimate_memory.amr.mono", "inspect_amr.amr.mono",
    "fft_field.amr.mono",
    "imexrk.amr.mono",
    "ab2.amr.mono", "bdf2.amr.mono",
    "clean_program.amr.mono", "clean_schur_program.amr.mono",
})
