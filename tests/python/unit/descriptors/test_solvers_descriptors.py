"""ADC-500 (Spec 5 sec.5.7 / criterion 4 / sec.13.11.1): the pops.solvers central package.

pops.solvers homes the executable linear / nonlinear / elliptic solver + preconditioner catalog
as inert typed descriptors. These tests construct each entry, exercise the RICH GeometricMG
parameter surface (typed smoother / coarse / tolerance + capabilities) and its protocol
(inspect / options / capabilities / lower), check that a bare string is rejected where a typed
sub-descriptor is expected (Spec 5 sec.7), and assert lower() carries the right native id /
scheme. They also confirm pops.solvers is the ONE public home for the solver descriptors (the
transitional pops.lib.solvers shim is removed). The descriptors compute nothing; only their
metadata is asserted.
"""
import pytest
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from pops.solvers.krylov._native_contract import PREPARED_GMRES_MAX_RESTART

pops = pytest.importorskip("pops")

solvers = pytest.importorskip("pops.solvers")
elliptic = pytest.importorskip("pops.solvers.elliptic")
krylov = pytest.importorskip("pops.solvers.krylov")
nonlinear = pytest.importorskip("pops.solvers.nonlinear")
_options = pytest.importorskip("pops.solvers.options")
_preconditioners = pytest.importorskip("pops.solvers.preconditioners")
_tolerances = pytest.importorskip("pops.solvers.tolerances")

Chebyshev = _options.Chebyshev
DirectSmallGrid = _options.DirectSmallGrid
RedBlackGaussSeidel = _options.RedBlackGaussSeidel
preconditioners = _preconditioners.preconditioners
Absolute = _tolerances.Absolute
AbsoluteFloor = _tolerances.AbsoluteFloor
Relative = _tolerances.Relative


# --- the package is wired and exposed ----------------------------------------------------

def test_solvers_is_top_level_and_exposed():
    assert pops.solvers is solvers
    for sub in ("elliptic", "krylov", "nonlinear",
                "options", "tolerances", "preconditioners", "requirements"):
        assert hasattr(solvers, sub), "pops.solvers missing sub-module %r" % sub


# --- Krylov solvers (moved from pops.lib.solvers) ----------------------------------------

def test_krylov_native_ids_and_schemes():
    assert krylov.CG(max_iter=200).native_id == "pops::solve_prepared_affine"
    assert krylov.CG(max_iter=200).scheme == "cg"
    assert krylov.BiCGStab(max_iter=200).native_id == "pops::solve_prepared_affine"
    assert krylov.BiCGStab(max_iter=200).scheme == "bicgstab"
    assert krylov.GMRES(max_iter=200).native_id == "pops::solve_prepared_affine"
    assert krylov.GMRES(max_iter=200).scheme == "gmres"
    assert krylov.Richardson(max_iter=200).native_id == "pops::solve_prepared_affine"
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


def test_krylov_integer_controls_match_the_native_signed_int_capacity():
    cpp_int_max = (1 << 31) - 1

    for factory in (krylov.CG, krylov.BiCGStab, krylov.GMRES, krylov.Richardson):
        assert factory(max_iter=cpp_int_max).options["max_iter"] == cpp_int_max
        with pytest.raises(ValueError, match="max_iter"):
            factory(max_iter=cpp_int_max + 1)

    assert krylov.GMRES(
        max_iter=1, restart=PREPARED_GMRES_MAX_RESTART
    ).options["method_options"]["restart"] == PREPARED_GMRES_MAX_RESTART
    with pytest.raises(ValueError, match="restart"):
        krylov.GMRES(max_iter=1, restart=PREPARED_GMRES_MAX_RESTART + 1)


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
    assert rec["native_id"] == "pops::solve_prepared_affine"
    assert rec["scheme"] == "cg"


@pytest.mark.parametrize("factory", [krylov.CG, krylov.BiCGStab, krylov.GMRES,
                                      krylov.Richardson])
def test_krylov_absolute_tolerance_is_exact_and_nonnegative(factory):
    assert factory(max_iter=10).options["abs_tol"] == 0
    absolute = Fraction(1, 10**12)
    assert factory(max_iter=10, abs_tol=absolute).options["abs_tol"] == absolute
    absolute_only = factory(max_iter=10, rel_tol=0, abs_tol=absolute)
    assert absolute_only.options["rel_tol"] == 0
    assert absolute_only.options["abs_tol"] == absolute
    prepared = absolute_only.prepare_program_solve()
    assert prepared.tolerance == 0
    assert prepared.absolute_tolerance == absolute
    with pytest.raises(ValueError, match="abs_tol"):
        factory(max_iter=10, abs_tol=-1)
    with pytest.raises(ValueError, match="at least one stopping threshold"):
        factory(max_iter=10, rel_tol=0, abs_tol=0)


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


# --- executable nonlinear solvers ---------------------------------------------------------

def test_nonlinear_surface_contains_only_executable_descriptors():
    assert not hasattr(nonlinear, "FixedPoint")
    local = nonlinear.LocalNewton(
        tolerance=1e-10, max_iterations=12, finite_difference_step=1e-6)
    assert local.to_data() == {
        "scheme": "newton",
        "tolerance": 1e-10,
        "max_iterations": 12,
        "finite_difference_step": 1e-6,
    }
    global_newton = nonlinear.Newton(restart=51)
    assert global_newton.options()["restart"] == 51
    assert global_newton.available().ok is True


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
    assert caps.supports("screened") is True


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
    assert rec["mg_options"]["schema_version"] == 1
    assert rec["mg_options"]["kind"] == "geometric_mg_options"
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
    from pops.layouts import Uniform
    from tests.python.support.layout_plan import cartesian_grid, final_amr_layout
    amr = final_amr_layout(cartesian_grid(n=64))
    status = elliptic.FFT().available({"layout": amr})
    assert status.status == "no"
    assert status.reason == "FFT requires Uniform(periodic=True), got AMR. Use GeometricMG()."
    assert "pops.solvers.elliptic.GeometricMG()" in status.alternatives
    # the context may BE the layout descriptor, not only wrap it under a "layout" key.
    assert elliptic.FFT().available(amr).status == "no"
    # a Uniform layout context (or no context at all) keeps the plain route-constraint 'partial'.
    assert elliptic.FFT().available({"layout": Uniform(cartesian_grid(n=64))}).status == "partial"
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
    from tests.python.support.layout_plan import cartesian_grid, final_amr_layout
    g = elliptic.GeometricMG()
    assert g.capabilities().supports("amr") is True
    assert g.available(final_amr_layout(cartesian_grid(n=64))).status == "yes"


# --- preconditioners ---------------------------------------------------------------------

def test_preconditioners_catalog():
    pre = solvers.preconditioners
    assert pre.GeometricMG().native_id == "pops::GeometricMG"
    assert pre.GeometricMG().category == "preconditioner"
    identity = pre.Identity()
    assert identity.available().ok is True
    assert identity.native_id == "pops::ApplyFn"
    for removed in ("Jacobi", "BlockJacobi"):
        assert not hasattr(pre, removed)
    for extension in (
        "Prepared", "register", "Provider", "IntOption", "ScratchResource",
        "NativeComponent", "HeaderOnlyComponent"
    ):
        assert callable(getattr(pre, extension))


def test_prepared_preconditioner_descriptor_requires_authenticated_provider():
    from pops.descriptors import _native
    from pops.time._program.solve import _lower_preconditioner

    forged = _native(
        "identity", "pops::ApplyFn", "identity", category="preconditioner"
    )
    with pytest.raises(ValueError, match="not authenticated"):
        _lower_preconditioner(forged)


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
    assert ns.CG(max_iter=200).native_id == "pops::solve_prepared_affine"
    assert ns.Newton().available().ok is True
    assert ns.LocalNewton().scheme == "newton"
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


def test_install_path_has_no_bind_time_solver_token_adapter():
    # Solver descriptors are lowered into resolved field plans before bind; the runtime install
    # seam must not reinterpret descriptor classes or tokens.
    from pops.runtime._system_unified_install import _SystemUnifiedInstall
    assert not hasattr(_SystemUnifiedInstall, "_solver_token")
    assert not hasattr(_SystemUnifiedInstall, "_install_solver")


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


def test_precond_geometric_mg_option_schema_matches_native_constructor_contract():
    from pops.solvers._prepared_preconditioner_registry import (
        prepared_preconditioner_provider_by_id,
    )

    provider = prepared_preconditioner_provider_by_id(
        "pops.preconditioner.geometric-mg"
    )
    assert [
        (option.name, option.default, option.minimum, option.maximum)
        for option in provider.options
    ] == [
        ("pre_sweeps", 2, 0, (1 << 31) - 1),
        ("post_sweeps", 2, 0, (1 << 31) - 1),
        ("bottom_sweeps", 50, 1, (1 << 31) - 1),
        ("min_coarse", 2, 1, (1 << 31) - 1),
        ("n_vcycles", 1, 1, (1 << 31) - 1),
    ]
    assert preconditioners.GeometricMG(
        pre_sweeps=0,
        post_sweeps=0,
        bottom_sweeps=1,
        min_coarse=1,
        n_vcycles=1,
    ).options == {
        "pre_sweeps": 0,
        "post_sweeps": 0,
        "bottom_sweeps": 1,
        "min_coarse": 1,
        "n_vcycles": 1,
    }


def test_preconditioner_provider_consumes_option_protocol_without_core_type_dispatch():
    from pops.solvers._prepared_preconditioner_registry import (
        PreparedPreconditionerNativeEmission,
        PreparedPreconditionerProvider,
        PreparedPreconditionerUsePolicy,
    )
    from pops.native_components import PreparedNativeComponent

    @dataclass(frozen=True, slots=True)
    class EnumOption:
        name: str
        default: str
        choices: tuple[str, ...]

        def validate(self, value, *, where):
            if type(value) is not str or value not in self.choices:
                raise ValueError("%s %s is not a supported enum value" % (where, self.name))
            return value

        def resolve(self, values, *, where):
            return self.validate(values.get(self.name, self.default), where=where)

        def emit_cpp_literal(self, value):
            return "Mode::%s" % value.capitalize()

        def contract_data(self, value):
            return value

        def authority(self):
            return {
                "schema_version": 1,
                "type_id": "pops.test.prepared-preconditioner.option.enum@1",
                "name": self.name,
                "default": self.default,
                "choices": list(self.choices),
            }

    provider = PreparedPreconditionerProvider(
        provider_id="pops.test.prepared-enum",
        interface_version=1,
        options_schema="pops.test.prepared-enum.options@1",
        scheme="test_enum",
        descriptor_name="test_enum",
        display_name="test enum",
        native_id="TestEnum",
        validator_id="pops.test.prepared-enum.validate@1",
        planner_id="pops.test.prepared-enum.plan@1",
        emitter_id="test.enum@1",
        preconditioned=True,
        prepared_buffers=0,
        use_policy=PreparedPreconditionerUsePolicy(
            "pops.test.enum.use", 1,
            {"methods": ("gmres",)}, lambda _use, _where: None,
        ),
        options=(EnumOption("mode", "safe", ("safe", "fast")),),
        emitter=lambda *_args: PreparedPreconditionerNativeEmission("TestEnum{}"),
        native_component=PreparedNativeComponent.pops_builtin(
            "pops.test.prepared-enum"
        ),
    )
    assert provider.resolved_cpp_option_literals({}, where="test") == ("Mode::Safe",)
    assert provider.resolved_cpp_option_literals(
        {"mode": "fast"}, where="test"
    ) == ("Mode::Fast",)
    with pytest.raises(ValueError, match="mode"):
        provider.resolved_cpp_option_literals({"mode": "unknown"}, where="test")


@pytest.mark.parametrize("kw", [{"tolerance": 1e-6}, {"max_cycles": 10}])
def test_precond_geometric_mg_refuses_iterative_knobs(kw):
    # A Krylov preconditioner must be a FIXED linear map; tolerance/max_cycles describe an iterative
    # solve-to-convergence and are refused loud (never swallowed).
    with pytest.raises(ValueError, match="FIXED linear map"):
        preconditioners.GeometricMG(**kw)


def test_precond_geometric_mg_refuses_unknown_kwarg():
    with pytest.raises(TypeError, match="unknown option"):
        preconditioners.GeometricMG(bogus=1)


@pytest.mark.parametrize(
    "kw",
    [
        {"n_vcycles": 0},
        {"min_coarse": 0},
        {"bottom_sweeps": 0},
        {"pre_sweeps": -1},
        {"post_sweeps": -1},
    ],
)
def test_precond_geometric_mg_refuses_out_of_domain(kw):
    with pytest.raises((ValueError, TypeError)):
        preconditioners.GeometricMG(**kw)


def test_precond_geometric_mg_refuses_bool_and_reauthenticates_mutated_options():
    with pytest.raises(TypeError, match="pre_sweeps"):
        preconditioners.GeometricMG(pre_sweeps=True)

    from pops.time._program.solve import _lower_preconditioner

    mutated = preconditioners.GeometricMG()
    mutated.options["bottom_sweeps"] = 0
    with pytest.raises(ValueError, match="bottom_sweeps"):
        _lower_preconditioner(mutated)


@pytest.mark.parametrize(
    "name",
    ["n_vcycles", "pre_sweeps", "post_sweeps", "bottom_sweeps", "min_coarse"],
)
def test_precond_geometric_mg_integer_knobs_match_native_int_capacity(name):
    cpp_int_max = (1 << 31) - 1
    assert preconditioners.GeometricMG(**{name: cpp_int_max}).options[name] == cpp_int_max
    with pytest.raises(ValueError, match=name):
        preconditioners.GeometricMG(**{name: cpp_int_max + 1})


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
    # None remains authored omission until the final field resolver snapshots the native POD.
    assert d.options() == {"max_iters": None, "fine_sweeps": None, "rel_tol": None,
                           "abs_tol": None,
                           "coarse_rel_tol": None, "coarse_abs_tol": None,
                           "coarse_cycles": None, "verbose": False}
    cfg = CompositeFAC(max_iters=10, fine_sweeps=200, rel_tol=1e-8, abs_tol=1e-14,
                       coarse_rel_tol=1e-11, coarse_abs_tol=1e-15, coarse_cycles=50,
                       verbose=True)
    assert cfg.options() == {
        "max_iters": 10, "fine_sweeps": 200, "rel_tol": 1e-8, "abs_tol": 1e-14,
        "coarse_rel_tol": 1e-11, "coarse_abs_tol": 1e-15, "coarse_cycles": 50,
        "verbose": True,
    }
    assert not hasattr(cfg, "set_poisson_kwargs")
    assert CompositeFAC(abs_tol=0.0).abs_tol == 0.0
    assert CompositeFAC(coarse_abs_tol=0.0).coarse_abs_tol == 0.0
    for bad in ({"max_iters": 0}, {"fine_sweeps": -1}, {"rel_tol": 1.5}, {"abs_tol": -1.0},
                {"coarse_rel_tol": 0.0}, {"coarse_abs_tol": -1.0}, {"coarse_cycles": 0}):
        with pytest.raises(ValueError):
            CompositeFAC(**bad)
    for bad in ({"max_iters": 1.9}, {"max_iters": True}, {"fine_sweeps": False},
                {"verbose": 1}):
        with pytest.raises(TypeError):
            CompositeFAC(**bad)


@pytest.mark.parametrize("name", ["max_iters", "fine_sweeps", "coarse_cycles"])
def test_composite_fac_integer_knobs_match_native_int_capacity(name):
    from pops.solvers.options import CompositeFAC

    cpp_int_max = (1 << 31) - 1
    assert getattr(CompositeFAC(**{name: cpp_int_max}), name) == cpp_int_max
    with pytest.raises(ValueError, match=name):
        CompositeFAC(**{name: cpp_int_max + 1})


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


def test_geometric_mg_fac_slot():
    from pops.solvers.options import CompositeFAC
    # Default None: the options view is UNCHANGED (omit-when-default, byte-identity).
    g = elliptic.GeometricMG()
    assert g.fac is None
    assert "fac" not in g.options()
    # Typed slot: a CompositeFAC is carried; a bare bool/string refuses.
    g2 = elliptic.GeometricMG(fac=CompositeFAC())
    assert g2.options()["fac"] == "composite_fac"
    with pytest.raises(TypeError, match="CompositeFAC"):
        elliptic.GeometricMG(fac=True)


def test_richardson_omega_and_krylov_rel_tol():
    # Provider-owned options are explicit even at their preset default.
    d = krylov.Richardson(max_iter=100)
    assert d.options["method_options"]["relaxation"] == {"kind": "integer", "value": "1"}
    assert "rel_tol" not in d.options
    d2 = krylov.Richardson(max_iter=100, omega=0.8)
    assert d2.options["method_options"]["relaxation"]["kind"] == "binary64"
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
    assert descriptor.options["method_options"]["relaxation"] == {
        "kind": "rational", "numerator": "2", "denominator": "3"
    }

    from pops._ir import ScalarLiteral
    annotated = ScalarLiteral.from_value(Fraction(1, 2), unit="s")
    with pytest.raises(ValueError, match="rel_tol"):
        krylov.CG(max_iter=10, rel_tol=annotated)
    with pytest.raises(ValueError, match="omega"):
        krylov.Richardson(max_iter=10, omega=annotated)


def test_weno5_epsilon_descriptor():
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.numerics import FiniteVolume
    from pops.numerics.reconstruction import reconstruction
    from pops.numerics.riemann import ScalarUpwind
    from pops.numerics.variables import Conservative
    # Default: no epsilon option (omit-when-default; the native kWenoEpsilon literal governs).
    assert "epsilon" not in reconstruction.WENO5().options
    assert reconstruction.WENO5(epsilon=1e-30).options["epsilon"] == 1e-30
    with pytest.raises(ValueError, match="epsilon"):
        reconstruction.WENO5(epsilon=-1.0)
    # The exact descriptor rides in the final typed finite-volume method; no root Spatial facade.
    frame = Rectangle("weno-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    model = pops.Model("weno", frame=frame)
    state = model.state("U", components=("u",))
    (u,) = state
    velocity = model.vector("a", frame=frame, components={frame.x: 1, frame.y: 0})
    flux = model.flux(
        "F", frame=frame, state=state,
        components={frame.x: (u,), frame.y: (0 * u,)},
        waves={frame.x: (1,), frame.y: (0,)},
    )
    method = FiniteVolume(
        flux=flux,
        variables=Conservative(state),
        reconstruction=reconstruction.WENO5(epsilon=1e-30),
        riemann=ScalarUpwind(velocity=velocity),
    )
    assert method.reconstruction.options["epsilon"] == 1e-30
    assert "epsilon" not in reconstruction.WENO5().options


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
