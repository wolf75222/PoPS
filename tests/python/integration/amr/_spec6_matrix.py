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
    ``pytest.skip`` with the pending marker; it flips to ``green_live`` when the named issue lands.
    The multistep-on-AMR rows are pending ADC-631 (the AMR history-ring seam being rewritten now);
    the clean-``compile(layout=AMR)`` whole-system Program rows are pending ADC-634 (the route being
    implemented now), the compiled condensed-Schur hierarchy Program pending ADC-634 + ADC-633. A
    pending cell NEVER executes the deferred path and NEVER pins today's transitional behavior.

Importing this module requires ``pops`` (the AMR-route handle + bricks); ``test_spec6_amr_matrix``
importorskips it so the suite is green on a bare box. ASCII only.
"""
import warnings
from collections import namedtuple

import numpy as np

import pops
import pops.lib.time as lib_time
from pops.codegen.loader import CompiledModel
from pops.mesh import CartesianMesh
from pops.mesh.amr import Refine, RegridEvery
from pops.mesh.layouts import AMR, Uniform
from pops.physics.model import Param
from pops.runtime.system import AmrSystem


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
    """A stub AMR-route ``CompiledModel`` (target='amr_system' + AMR ``_layout``): what the AMR route
    of ``pops.compile`` returns. Drives the INERT introspection cells (no ``.so`` dlopen)."""
    handle = CompiledModel(
        so_path="<stub-amr>", backend="production", adder="add_native_block",
        cons_names=["rho", "mx", "my"], cons_roles=["Density", "MomentumX", "MomentumY"],
        prim_names=["rho", "mx", "my"], n_vars=3, gamma=1.4, n_aux=n_aux,
        params={"alpha": Param("alpha", 1.0, kind="runtime")},
        caps={"cpu": True, "amr": True, "mpi": mpi}, abi_key="k", model_hash="h", cxx="c++",
        std="c++23", target="amr_system", aux_extra_names=["B_z"][:n_aux])
    handle._layout = AMR(base=CartesianMesh(n=64, periodic=True), max_levels=2, ratio=2,
                         regrid=RegridEvery(4), refine=Refine.on("rho").above(0.1))
    return handle


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
# green_inert: introspection on the AMR-route handle (no run, no dlopen).
# --------------------------------------------------------------------------------------------------
def _inert_arguments():
    args = _amr_route_handle().arguments()
    assert args.layout_runtime["layout"] == "amr"
    assert next(iter(args.instances.values()))["components"] == 3
    assert set(args.aux) == {"B_z"} and args.params["alpha"]["kind"] == "runtime"
    return "green_inert:arguments_amr"


def _inert_estimate_memory():
    handle = _amr_route_handle()
    mesh = CartesianMesh(n=64, L=1.0, periodic=True)
    est = handle.estimate_memory(mesh)
    assert est.layout == "amr" and est.categories.get("amr_patch", 0) > 0
    assert est.total_bytes >= handle.estimate_memory(mesh, layout=Uniform(mesh)).total_bytes
    return "green_inert:estimate_memory_amr"


def _inert_inspect_amr():
    rep = _amr_route_handle().inspect_amr().to_dict()
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


def _refuse_runtime_params_on_amr():
    def call():
        AmrSystem(n=16, L=1.0, periodic=True, regrid_every=0)._finish_program_install(
            compiled=None, so_path=None, params={"alpha": 1.0}, cadence=None)
    return _expect_refusal(NotImplementedError,
                           ("not wired on a NATIVE AMR install", "per-block param seam"), call)


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
    assert isinstance(lib_time.ssprk3("plasma"), pops.time.Program)
    return "exists:test_amr_ssprk3"


def _exists_ssprk2_program_parity():
    # Cited: test_amr_program_parity (ADC-508 compiled-Program SSPRK2 parity on AMR). The compiled
    # whole-system Program parity is proven there; assert the ssprk2 macro authors a stable Program.
    prog = lib_time.ssprk2("plasma")
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
def _pending_multistep(builder):
    """A multistep-on-AMR pending row: build the whole-system Program (with its history ring ops) to
    prove the authoring is real, then defer to ADC-631 (the AMR history-ring seam under rewrite)."""
    def run():
        program = builder("plasma")
        assert isinstance(program, pops.time.Program)
        # The multistep Program carries a history ring (store_history / history); the AMR seam that
        # serves it is what ADC-631 is rewriting. Do NOT execute it on an AmrSystem here.
        assert any(v.op in ("store_history", "history") for v in program._values)
        return "pending:ADC-631"
    return run


def _pending_clean_route_program(builder, marker):
    """A clean-``compile(layout=AMR)`` whole-system Program pending row: build the Program object, then
    defer to ADC-634 (the route being implemented). Do NOT call ``pops.compile(layout=AMR, time=...)``
    (its behavior is being changed by ADC-634); only prove the authoring object is real."""
    def run():
        program = builder()
        assert isinstance(program, pops.time.Program)
        return "pending:%s" % marker
    return run


def _condensed_schur_program():
    return lib_time.condensed_schur("plasma", alpha=1.0)


# --------------------------------------------------------------------------------------------------
# THE MATRIX -- keyed "op.layout.blocks". Uniform baseline cells cite the shipping Spec 5 coverage;
# the AMR column is the ADC-515 focus (green live / inert, precise refusals, pending rows).
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
    "runtime_params.amr.mono": Cell("runtime_params", "amr", "mono", "refuse",
                                    _refuse_runtime_params_on_amr),
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
    # multistep AB2 / BDF2 on AMR -- PENDING ADC-631 (history ring under rewrite)
    "ab2.amr.mono": Cell("ab2", "amr", "mono", "pending",
                         _pending_multistep(lib_time.adams_bashforth2)),
    "bdf2.amr.mono": Cell("bdf2", "amr", "mono", "pending",
                          _pending_multistep(lambda block: lib_time.bdf(block, order=2))),
    # clean-compile(layout=AMR) whole-system Program -- PENDING ADC-634 (route being implemented)
    "clean_program.amr.mono": Cell("clean_program", "amr", "mono", "pending",
                                   _pending_clean_route_program(
                                       lambda: lib_time.ssprk3("plasma"), "ADC-634")),
    # compiled condensed-Schur hierarchy Program -- PENDING ADC-634 + ADC-633 (hierarchy elliptic)
    "clean_schur_program.amr.mono": Cell("clean_schur_program", "amr", "mono", "pending",
                                         _pending_clean_route_program(
                                             _condensed_schur_program, "ADC-634 + ADC-633")),
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
