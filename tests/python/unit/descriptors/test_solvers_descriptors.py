"""ADC-500 (Spec 5 sec.5.7 / criterion 4 / sec.13.11.1): the pops.solvers central package.

pops.solvers homes the linear / nonlinear / Schur / elliptic solver + preconditioner catalog
as inert typed descriptors. These tests construct each entry, exercise the RICH GeometricMG
parameter surface (typed smoother / coarse / tolerance + capabilities) and its protocol
(inspect / options / capabilities / lower), check that a bare string is rejected where a typed
sub-descriptor is expected (Spec 5 sec.7), and assert lower() carries the right native id /
scheme. They also confirm pops.solvers is the ONE public home for the solver descriptors (the
transitional pops.lib.solvers shim is removed). The descriptors compute nothing; only their
metadata is asserted.
"""
import pytest
from decimal import Decimal
from fractions import Fraction

pops = pytest.importorskip("pops")
solvers = pytest.importorskip("pops.solvers")

from pops.solvers import elliptic, krylov, nonlinear, schur
from pops.solvers.options import Chebyshev, DirectSmallGrid, RedBlackGaussSeidel
from pops.solvers.preconditioners import preconditioners
from pops.solvers.tolerances import Absolute, AbsoluteFloor, Relative


# --- the package is wired and exposed ----------------------------------------------------

def test_solvers_is_top_level_and_exposed():
    assert pops.solvers is solvers
    for sub in ("elliptic", "krylov", "nonlinear", "schur",
                "options", "tolerances", "preconditioners", "requirements"):
        assert hasattr(solvers, sub), "pops.solvers missing sub-module %r" % sub


# --- Krylov solvers (moved from pops.lib.solvers) ----------------------------------------

def test_krylov_native_ids_and_schemes():
    assert krylov.CG(max_iter=200).native_id == "pops::cg_solve"
    assert krylov.CG(max_iter=200).scheme == "cg"
    assert krylov.BiCGStab(max_iter=200).native_id == "pops::bicgstab_solve"
    assert krylov.BiCGStab(max_iter=200).scheme == "bicgstab"
    assert krylov.GMRES(max_iter=200).native_id == "pops::gmres_solve"
    assert krylov.GMRES(max_iter=200).scheme == "gmres"
    assert krylov.Richardson(max_iter=200).native_id == "pops::richardson_solve"
    assert krylov.Richardson(max_iter=200).scheme == "richardson"
    for d in (krylov.CG(max_iter=200), krylov.GMRES(max_iter=200),
              krylov.BiCGStab(max_iter=200), krylov.Richardson(max_iter=200)):
        assert d.brick_type == "native"
        assert d.available().ok
        assert d.category == "solver"


def test_krylov_max_iter_is_mandatory_and_carried():
    # ADC-535: max_iter is a MANDATORY positive int on the descriptor; the native pops::*_solve
    # loops throw on a non-positive budget, so the descriptor refuses one BEFORE the runtime.
    for factory in (krylov.CG, krylov.BiCGStab, krylov.GMRES, krylov.Richardson):
        d = factory(max_iter=123)
        assert d.options["max_iter"] == 123, d.options
        assert d.inspect()["options"]["max_iter"] == 123
        assert d.lower().to_dict()["options"]["max_iter"] == 123


@pytest.mark.parametrize("factory", [krylov.CG, krylov.BiCGStab, krylov.GMRES, krylov.Richardson])
def test_krylov_missing_max_iter_is_refused(factory):
    # A missing budget is refused at construction (pre-runtime), naming the factory.
    with pytest.raises(ValueError, match="max_iter is required"):
        factory()


@pytest.mark.parametrize("factory", [krylov.CG, krylov.BiCGStab, krylov.GMRES, krylov.Richardson])
@pytest.mark.parametrize("bad", [0, -1, -100, True, 2.0, "200"])
def test_krylov_nonpositive_or_nonint_max_iter_is_refused(factory, bad):
    # Zero / negative / bool / non-int budgets are all refused (dynamic loops require a real budget).
    with pytest.raises(ValueError, match="max_iter"):
        factory(max_iter=bad)


def test_krylov_descriptors_compute_nothing():
    d = krylov.GMRES(max_iter=200)
    assert not hasattr(d, "eval")
    assert not hasattr(d, "compile")
    # value identity: the same descriptor twice (same budget) compares equal.
    assert krylov.GMRES(max_iter=200) == krylov.GMRES(max_iter=200)
    # a different budget is a distinct descriptor (the budget is part of the identity key).
    assert krylov.GMRES(max_iter=200) != krylov.GMRES(max_iter=400)


def test_krylov_lower_carries_native_id_and_scheme():
    rec = krylov.CG(max_iter=200).lower().to_dict()
    assert rec["native_id"] == "pops::cg_solve"
    assert rec["scheme"] == "cg"


def test_krylov_declare_amr_route_capabilities():
    # Spec 6 sec.4 / sec.9: the matrix-free Krylov solvers are layout-agnostic (they run over
    # pops::MultiFab dot / saxpy / apply), so each declares every route -- uniform / amr / mpi /
    # gpu -- and a route check can see they are AMR-capable instead of guessing from an empty set.
    for d in (krylov.CG(max_iter=200), krylov.GMRES(max_iter=200),
              krylov.BiCGStab(max_iter=200), krylov.Richardson(max_iter=200)):
        caps = d.capabilities
        assert caps["supports_uniform"] is True
        assert caps["supports_amr"] is True
        assert caps["supports_mpi"] is True
        assert caps["supports_gpu"] is True
        # the capabilities are surfaced by inspect() too.
        assert d.inspect()["capabilities"]["supports_amr"] is True


# --- nonlinear solvers (planned: no native type yet) -------------------------------------

def test_nonlinear_are_planned():
    for d in (nonlinear.Newton(), nonlinear.FixedPoint()):
        assert d.available().ok is False
        assert d.native_id == ""
        assert d.category == "solver"
    assert nonlinear.Newton().scheme == "newton"
    assert nonlinear.FixedPoint().scheme == "fixed_point"


def test_nonlinear_refuse_cleanly_with_no_native_backing():
    # ADC-535: Newton / FixedPoint have NO native solver TYPE (Newton is the implicit-stepper
    # kernel; a fixed point is authored over Krylov). They must REFUSE cleanly -- validate()
    # raises a clear "no native C++ symbol yet" message, never fabricating a symbol.
    for d in (nonlinear.Newton(), nonlinear.FixedPoint()):
        with pytest.raises(ValueError, match=r"no native C\+\+ symbol"):
            d.validate()
        # the refusal is surfaced structurally too (the ADC-549 capability-matrix row).
        row = d.capability_matrix().rows[0]
        assert row.status == "unavailable"
        assert row.error_message  # names the unsupported route


# --- Schur-condensation solver -----------------------------------------------------------

def test_schur_native_id_and_alias():
    assert schur.Schur().native_id == "pops::SchurCondensationOperator"
    assert schur.Schur().scheme == "schur"
    # CondensedSchur is an alias naming the same native operator (distinct from the
    # pops.time CondensedSchur splitting POLICY).
    assert schur.CondensedSchur().native_id == "pops::SchurCondensationOperator"
    assert schur.CondensedSchur() == schur.Schur()


def test_schur_declares_amr_route_capabilities():
    # Spec 6 sec.4 / sec.9: the Schur-condensation solver runs on AMR (the amr-schur source
    # stage) and System, under MPI and on the GPU, so it declares every route capability.
    for d in (schur.Schur(), schur.CondensedSchur()):
        caps = d.capabilities
        assert caps["supports_uniform"] is True
        assert caps["supports_amr"] is True
        assert caps["supports_mpi"] is True
        assert caps["supports_gpu"] is True


# --- the RICH GeometricMG elliptic solver ------------------------------------------------

def test_geometric_mg_defaults():
    g = elliptic.GeometricMG()
    assert g.name == "geometric_mg"
    assert g.scheme == "geometric_mg"
    assert g.native_id == "pops::GeometricMG"
    assert g.category == "elliptic_solver"
    opts = g.options()
    # ADC-613 reconciled the descriptor defaults to the native kMG* constants (the values the
    # V-cycle actually used): Gauss-Seidel smoother, 50 cycles, explicit sweep counts.
    assert opts == {"smoother": "red_black_gauss_seidel", "coarse": "direct_small_grid",
                    "tolerance": "relative", "max_cycles": 50, "min_coarse": 2,
                    "pre_sweeps": 2, "post_sweeps": 2, "bottom_sweeps": 50}


def test_geometric_mg_rich_surface():
    g = elliptic.GeometricMG(
        smoother=RedBlackGaussSeidel(),
        coarse=DirectSmallGrid(threshold=64),
        tolerance=Relative(rel=1e-8, floor=AbsoluteFloor(1e-14)),
        max_cycles=30)
    assert g.smoother.name == "red_black_gauss_seidel"
    assert g.coarse.options() == {"threshold": 64}
    assert g.tolerance.options() == {"rel": 1e-8, "abs_floor": 1e-14}
    assert g.options()["max_cycles"] == 30
    # An Absolute tolerance is accepted too.
    assert elliptic.GeometricMG(tolerance=Absolute(1e-9)).tolerance.name == "absolute"
    # A Chebyshev smoother of a chosen degree is accepted.
    assert elliptic.GeometricMG(smoother=Chebyshev(degree=4)).smoother.options() == {"degree": 4}


def test_geometric_mg_capabilities():
    caps = elliptic.GeometricMG().capabilities()
    assert caps.supports("uniform") is True
    assert caps.supports("amr") is True
    assert caps.supports("mpi") is True
    assert caps.supports("gpu") is True
    assert caps.supports("variable_epsilon") is True
    assert caps.supports("anisotropic") is False
    assert caps.supports("screened") is False


def test_geometric_mg_inspect_and_lower():
    # Chebyshev is structurally refused since ADC-613 (covered in test_geometric_mg_options);
    # the round-trip uses the natively wired Gauss-Seidel smoother.
    g = elliptic.GeometricMG(smoother=RedBlackGaussSeidel())
    view = g.inspect()
    assert view["name"] == "geometric_mg"
    assert view["native_id"] == "pops::GeometricMG"
    assert view["scheme"] == "geometric_mg"
    assert view["available"] is True
    assert view["capabilities"]["supports_amr"] is True
    rec = g.lower().to_dict()
    assert rec["native_id"] == "pops::GeometricMG"
    assert rec["scheme"] == "geometric_mg"
    assert rec["smoother"] == {"kind": "red_black_gauss_seidel"}
    assert rec["coarse"]["kind"] == "direct_small_grid"
    assert rec["tolerance"]["kind"] == "relative"
    assert rec["mg_options"]["max_cycles"] == 50
    assert rec["mg_options"]["rel_tol"] == 1e-8


def test_geometric_mg_rejects_string_for_typed_subdescriptor():
    # Spec 5 sec.7: a bare string / number for a typed sub-descriptor slot is rejected loud.
    with pytest.raises(TypeError, match="smoother"):
        elliptic.GeometricMG(smoother="chebyshev")
    with pytest.raises(TypeError, match="coarse"):
        elliptic.GeometricMG(coarse="direct")
    with pytest.raises(TypeError, match="tolerance"):
        elliptic.GeometricMG(tolerance=1e-6)
    with pytest.raises(TypeError, match="max_cycles"):
        elliptic.GeometricMG(max_cycles="20")
    # The tolerance floor must itself be a typed AbsoluteFloor.
    with pytest.raises(TypeError, match="floor"):
        Relative(rel=1e-6, floor=1e-12)


# --- the FFT elliptic solver (real pops::PoissonFFTSolver) --------------------------------

def test_fft_is_a_real_solver_with_route_constraints():
    f = elliptic.FFT()
    assert f.name == "fft"
    # A real, runtime-wired solver -- not unimplemented.
    assert f.native_id == "pops::PoissonFFTSolver"
    assert f.scheme == "fft"
    status = f.available()
    # partial = genuine route constraints (periodic / const-coeff / power-of-two), not "no symbol".
    assert status.status == "partial"
    assert any("periodic" in m for m in status.missing)
    assert "pops.solvers.elliptic.GeometricMG()" in status.alternatives
    assert f.inspect()["available"] == "partial"
    # spectral=True selects the fft_spectral token.
    spectral = elliptic.FFT(spectral=True)
    assert spectral.scheme == "fft_spectral"
    assert spectral.options() == {"spectral": True}


def test_fft_rejects_amr_layout_with_precise_message():
    # Spec 6 sec.8: pairing FFT with an AMR layout is a MATHEMATICAL incompatibility -- it must
    # be refused with the PRECISE message (not a vague "AMR unsupported"), naming GeometricMG.
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import AMR, Uniform
    amr = AMR(base=CartesianMesh(n=64))
    status = elliptic.FFT().available({"layout": amr})
    assert status.status == "no"
    assert status.reason == "FFT requires Uniform(periodic=True), got AMR. Use GeometricMG()."
    assert "pops.solvers.elliptic.GeometricMG()" in status.alternatives
    # the context may BE the layout descriptor, not only wrap it under a "layout" key.
    assert elliptic.FFT().available(amr).status == "no"
    # a Uniform layout context (or no context at all) keeps the plain route-constraint 'partial'.
    assert elliptic.FFT().available({"layout": Uniform(CartesianMesh(n=64))}).status == "partial"
    assert elliptic.FFT().available().status == "partial"


def test_fft_available_never_raises_on_odd_context():
    # available() must ALWAYS return an Availability, never raise (Spec 5 sec.6: an explainable
    # status) -- even when the context's capabilities() needs an argument or itself raises. Such a
    # context is simply "not an AMR layout", so the route keeps its plain partial status.
    class _ArgCaps:
        def capabilities(self, required):  # callable but needs an argument
            return {"layout": "amr"}

    class _RaiseCaps:
        def capabilities(self):
            raise RuntimeError("boom")

    for ctx in ({"layout": _ArgCaps()}, _ArgCaps(), {"layout": _RaiseCaps()}, _RaiseCaps()):
        assert elliptic.FFT().available(ctx).status == "partial"


def test_geometric_mg_accepts_amr_layout():
    # GeometricMG is the AMR-capable elliptic solver: it advertises amr and stays available with
    # an AMR layout context (no rejection), so it is the alternative FFT points at.
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import AMR
    g = elliptic.GeometricMG()
    assert g.capabilities().supports("amr") is True
    assert g.available(AMR(base=CartesianMesh(n=64))).status == "yes"


# --- preconditioners ---------------------------------------------------------------------

def test_preconditioners_catalog():
    pre = solvers.preconditioners
    assert pre.GeometricMG().native_id == "pops::GeometricMG"
    assert pre.GeometricMG().category == "preconditioner"
    for d in (pre.Identity(), pre.Jacobi(), pre.BlockJacobi()):
        assert d.available().ok is False
        assert d.native_id == ""


# --- requirements vocabulary -------------------------------------------------------------

def test_capability_vocabulary_rejects_unknown_tag():
    from pops.solvers.requirements import capability_map
    assert capability_map(uniform=True)["supports_uniform"] is True
    with pytest.raises(ValueError, match="unknown solver capability tag"):
        capability_map(quantum=True)


# --- one public home: the pops.lib.solvers shim is removed (no second public path) -------

def test_lib_solvers_shim_is_removed():
    # No-soft-compat: the solver descriptors live in exactly ONE public home, pops.solvers. The
    # transitional pops.lib.solvers re-export shim is gone -- importing it must fail, and pops.lib
    # exposes no solvers / preconditioners attribute.
    import importlib
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pops.lib.solvers")
    import pops.lib
    assert not hasattr(pops.lib, "solvers"), "pops.lib must not re-export the solver catalog"
    assert not hasattr(pops.lib, "preconditioners"), "pops.lib must not re-export preconditioners"

    # The one public home resolves the flat factory namespace and the preconditioners.
    ns = solvers.solvers
    assert ns.GMRES(max_iter=200).scheme == "gmres"
    assert ns.CG(max_iter=200).native_id == "pops::cg_solve"
    assert ns.Schur().native_id == "pops::SchurCondensationOperator"
    assert ns.Newton().available().ok is False
    assert solvers.preconditioners.GeometricMG().native_id == "pops::GeometricMG"

    # The custom-solver authoring / generation DSL is internal / experimental under
    # pops.codegen.solvers (criterion 19); it is NOT a public attribute of pops.solvers.
    for absent in ("solver", "build_solver_ir", "generate_solver_cpp", "SolverContext", "SolverIR"):
        assert not hasattr(solvers, absent), \
            "pops.solvers is a catalog, not the authoring DSL (saw %r)" % absent
    # The registry hooks are wired onto the shared solvers namespace by the DSL package.
    cs = pytest.importorskip("pops.codegen.solvers")
    assert getattr(cs, "__experimental__", None) is True
    assert callable(cs.solver)
    assert callable(cs.generate_solver_cpp)
    assert callable(ns.custom)
    assert callable(ns.registered)


def test_install_path_token_resolution_for_rich_descriptor():
    # The unified-install solver-token resolver reads .scheme; the new rich GeometricMG
    # resolves to the same 'geometric_mg' token as the brick-catalog pops.fields.catalog descriptor.
    from pops.runtime._system_unified_install import _SystemUnifiedInstall
    assert _SystemUnifiedInstall._solver_token(elliptic.GeometricMG()) == "geometric_mg"
    assert _SystemUnifiedInstall._solver_token(pops.fields.catalog.GeometricMG()) == "geometric_mg"


# --- ADC-644: the wired GeometricMG preconditioner option surface -----------------------------
def test_precond_geometric_mg_default_has_no_options():
    # A default GeometricMG() preconditioner carries an EMPTY options dict, so the lowering returns
    # None and the emitted V-cycle stays byte-identical to the historical single-cycle preconditioner.
    d = preconditioners.GeometricMG()
    assert d.category == "preconditioner"
    assert d.scheme == "geometric_mg"
    assert d.options == {}


def test_precond_geometric_mg_carries_validated_shape_knobs():
    d = preconditioners.GeometricMG(n_vcycles=3, pre_sweeps=1, post_sweeps=1, bottom_sweeps=80,
                                    min_coarse=4)
    assert d.options == {"n_vcycles": 3, "pre_sweeps": 1, "post_sweeps": 1, "bottom_sweeps": 80,
                         "min_coarse": 4}


@pytest.mark.parametrize("kw", [{"tolerance": 1e-6}, {"max_cycles": 10}])
def test_precond_geometric_mg_refuses_iterative_knobs(kw):
    # A Krylov preconditioner must be a FIXED linear map; tolerance/max_cycles describe an iterative
    # solve-to-convergence and are refused loud (never swallowed).
    with pytest.raises(ValueError, match="FIXED linear map"):
        preconditioners.GeometricMG(**kw)


def test_precond_geometric_mg_refuses_unknown_kwarg():
    with pytest.raises(TypeError, match="unknown option"):
        preconditioners.GeometricMG(bogus=1)


@pytest.mark.parametrize("kw", [{"n_vcycles": 0}, {"min_coarse": 0}, {"pre_sweeps": -1}])
def test_precond_geometric_mg_refuses_out_of_domain(kw):
    with pytest.raises((ValueError, TypeError)):
        preconditioners.GeometricMG(**kw)


# --- ADC-644: DirectSmallGrid threshold is None by default (wired, not dropped) -----------------
def test_direct_small_grid_default_is_disabled():
    # The default threshold is None ("governed by min_coarse"), lowering to the disabled sentinel 0
    # so an unconfigured GeometricMG() keeps today's coarsening hierarchy bit-for-bit.
    assert DirectSmallGrid().threshold is None
    assert elliptic.GeometricMG().mg_options()["coarse_threshold"] == 0


def test_direct_small_grid_explicit_threshold_reaches_mg_options():
    assert DirectSmallGrid(64).threshold == 64
    opts = elliptic.GeometricMG(coarse=DirectSmallGrid(64)).mg_options()
    assert opts["coarse_threshold"] == 64


@pytest.mark.parametrize("bad", [0, -3])
def test_direct_small_grid_refuses_non_positive(bad):
    with pytest.raises(ValueError):
        DirectSmallGrid(bad)


# --- ADC-645: CompositeFAC / Richardson omega / Krylov rel_tol --------------------------------
def test_composite_fac_defaults_and_domain():
    from pops.solvers.options import CompositeFAC
    d = CompositeFAC()
    # None -> the 0 wire sentinels (native kFAC* defaults), the CondensedSchur fac_* convention.
    assert d.options() == {"max_iters": None, "fine_sweeps": None, "tol": None,
                           "coarse_rel_tol": None, "coarse_cycles": None, "verbose": False}
    kw = d.set_poisson_kwargs()
    assert kw["composite"] is True and kw["fac_max_iters"] == 0
    cfg = CompositeFAC(max_iters=10, fine_sweeps=200, tol=1e-8, coarse_rel_tol=1e-11,
                       coarse_cycles=50, verbose=True)
    assert cfg.set_poisson_kwargs() == {"composite": True, "fac_max_iters": 10,
                                        "fac_fine_sweeps": 200, "fac_tol": 1e-8,
                                        "fac_coarse_rel_tol": 1e-11, "fac_coarse_cycles": 50,
                                        "fac_verbose": True}
    for bad in ({"max_iters": 0}, {"fine_sweeps": -1}, {"tol": 1.5}, {"coarse_rel_tol": 0.0},
                {"coarse_cycles": 0}):
        with pytest.raises(ValueError):
            CompositeFAC(**bad)
    for bad in ({"max_iters": 1.9}, {"max_iters": True}, {"fine_sweeps": False},
                {"verbose": 1}):
        with pytest.raises(TypeError):
            CompositeFAC(**bad)


def test_solver_tolerances_retain_exact_domains_until_native_lowering():
    rel = Relative(Fraction(1, 3), AbsoluteFloor(Decimal("1e-30")))
    absolute = Absolute(Decimal("1e-24"))

    assert rel.rel == Fraction(1, 3)
    assert rel.floor.abs_floor == Decimal("1e-30")
    assert rel.options()["rel"] == Fraction(1, 3)
    assert absolute.abs_tol == Decimal("1e-24")
    mg = elliptic.GeometricMG(tolerance=rel).mg_options()
    assert mg["rel_tol"] == Fraction(1, 3)
    assert mg["abs_tol"] == Decimal("1e-30")


@pytest.mark.parametrize("factory", [Relative, Absolute, AbsoluteFloor])
@pytest.mark.parametrize("bad", [True, 0, -1, float("nan"), float("inf")])
def test_solver_tolerances_reject_bool_nonpositive_and_nonfinite(factory, bad):
    with pytest.raises((TypeError, ValueError)):
        factory(bad)


def test_geometric_mg_amr_composite_slot():
    from pops.solvers.options import CompositeFAC
    # Default None: the options view is UNCHANGED (omit-when-default, byte-identity).
    g = elliptic.GeometricMG()
    assert g.amr_composite is None
    assert "amr_composite" not in g.options()
    # Typed slot: a CompositeFAC is carried; a bare bool/string refuses.
    g2 = elliptic.GeometricMG(amr_composite=CompositeFAC())
    assert g2.options()["amr_composite"] == "composite_fac"
    with pytest.raises(TypeError, match="CompositeFAC"):
        elliptic.GeometricMG(amr_composite=True)


def test_richardson_omega_and_krylov_rel_tol():
    # omega: carried only when set (omit-when-default keeps the descriptor identity unchanged).
    d = krylov.Richardson(max_iter=100)
    assert "omega" not in d.options and "rel_tol" not in d.options
    d2 = krylov.Richardson(max_iter=100, omega=0.8)
    assert d2.options["omega"] == 0.8
    with pytest.raises(ValueError, match="omega"):
        krylov.Richardson(max_iter=100, omega=0.0)
    # rel_tol on every factory; out-of-domain refuses.
    for factory in (krylov.CG, krylov.BiCGStab, krylov.GMRES, krylov.Richardson):
        assert factory(max_iter=10, rel_tol=1e-9).options["rel_tol"] == 1e-9
        with pytest.raises(ValueError, match="rel_tol"):
            factory(max_iter=10, rel_tol=2.0)


def test_krylov_descriptor_controls_preserve_exact_number_domains():
    from decimal import Decimal
    from fractions import Fraction

    descriptor = krylov.Richardson(
        max_iter=10, rel_tol=Decimal("1e-12"), omega=Fraction(2, 3))

    assert descriptor.options["rel_tol"] == Decimal("1e-12")
    assert isinstance(descriptor.options["rel_tol"], Decimal)
    assert descriptor.options["omega"] == Fraction(2, 3)
    assert isinstance(descriptor.options["omega"], Fraction)

    from pops.ir import ScalarLiteral
    annotated = ScalarLiteral.from_value(Fraction(1, 2), unit="s")
    with pytest.raises(ValueError, match="rel_tol"):
        krylov.CG(max_iter=10, rel_tol=annotated)
    with pytest.raises(ValueError, match="omega"):
        krylov.Richardson(max_iter=10, omega=annotated)


def test_condensed_schur_precond_knobs():
    # ADC-645: n_precond_vcycles in {1, 2}; polar_precond in {radial_line, jacobi}; defaults 0/"".
    cs = pops.CondensedSchur()
    assert cs.n_precond_vcycles == 0 and cs.polar_precond == ""
    cs2 = pops.CondensedSchur(n_precond_vcycles=2, polar_precond="jacobi")
    assert cs2.n_precond_vcycles == 2 and cs2.polar_precond == "jacobi"
    with pytest.raises(ValueError, match="n_precond_vcycles"):
        pops.CondensedSchur(n_precond_vcycles=3)
    with pytest.raises(ValueError, match="polar_precond"):
        pops.CondensedSchur(polar_precond="bogus")


def test_weno5_epsilon_descriptor():
    from pops.numerics.reconstruction import reconstruction
    # Default: no epsilon option (omit-when-default; the native kWenoEpsilon literal governs).
    assert "epsilon" not in reconstruction.WENO5().options
    assert reconstruction.WENO5(epsilon=1e-30).options["epsilon"] == 1e-30
    with pytest.raises(ValueError, match="epsilon"):
        reconstruction.WENO5(epsilon=-1.0)
    # The Spatial ride-along (mirror of waves_provider).
    sp = pops.Spatial(reconstruction=reconstruction.WENO5(epsilon=1e-30))
    assert sp.weno_epsilon == 1e-30
    assert pops.Spatial(reconstruction=reconstruction.WENO5()).weno_epsilon is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
