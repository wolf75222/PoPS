"""ADC-500 (Spec 5 sec.5.7 / criterion 4 / sec.13.11.1): the pops.solvers central package.

pops.solvers homes the linear / Schur / elliptic solver + preconditioner catalog
as inert typed descriptors. These tests construct each entry, exercise the RICH GeometricMG
parameter surface (typed smoother / coarse / tolerance + capabilities) and its protocol
(inspect / options / capabilities / lower), check that a bare string is rejected where a typed
sub-descriptor is expected (Spec 5 sec.7), and assert lower() carries the right native id /
scheme. They also confirm pops.solvers is the ONE public home for the solver descriptors (the
transitional pops.lib.solvers shim is removed). The descriptors compute nothing; only their
metadata is asserted.
"""
import pytest
from pathlib import Path

pops = pytest.importorskip("pops")
solvers = pytest.importorskip("pops.solvers")

from pops.solvers import elliptic, krylov, nonlinear, schur
from pops.solvers.options import Chebyshev, DirectSmallGrid, RedBlackGaussSeidel
from pops.solvers.tolerances import Absolute, AbsoluteFloor, Relative


# --- the package is wired and exposed ----------------------------------------------------

def test_solvers_is_top_level_and_exposed():
    assert pops.solvers is solvers
    for sub in ("elliptic", "krylov", "schur", "nonlinear",
                "options", "tolerances", "preconditioners", "requirements"):
        assert hasattr(solvers, sub), "pops.solvers missing sub-module %r" % sub


# --- Krylov solvers (moved from pops.lib.solvers) ----------------------------------------

def test_krylov_native_ids_and_schemes():
    assert krylov.CG(max_iter=100).native_id == "pops::cg_solve"
    assert krylov.CG(max_iter=100).scheme == "cg"
    assert krylov.BiCGStab(max_iter=100).native_id == "pops::bicgstab_solve"
    assert krylov.BiCGStab(max_iter=100).scheme == "bicgstab"
    assert krylov.GMRES(max_iter=100).native_id == "pops::gmres_solve"
    assert krylov.GMRES(max_iter=100).scheme == "gmres"
    assert krylov.Richardson(max_iter=100).native_id == "pops::richardson_solve"
    assert krylov.Richardson(max_iter=100).scheme == "richardson"
    for d in (krylov.CG(max_iter=100), krylov.GMRES(max_iter=100),
              krylov.BiCGStab(max_iter=100), krylov.Richardson(max_iter=100)):
        assert d.brick_type == "native"
        assert d.available().ok
        assert d.category == "solver"


def test_krylov_descriptors_compute_nothing():
    d = krylov.GMRES(max_iter=100)
    assert not hasattr(d, "eval")
    assert not hasattr(d, "compile")
    # value identity: the same descriptor twice compares equal.
    assert krylov.GMRES(max_iter=100) == krylov.GMRES(max_iter=100)


def test_krylov_lower_carries_native_id_and_scheme():
    rec = krylov.CG(max_iter=100).lower()
    assert rec["native_id"] == "pops::cg_solve"
    assert rec["scheme"] == "cg"


def test_krylov_descriptor_options_validate_and_feed_solve_defaults():
    d = krylov.BiCGStab(tolerance=1e-10, max_iter=400)
    assert d.options["tolerance"] == 1e-10
    assert d.options["max_iter"] == 400
    assert krylov.GMRES(max_iter=400, restart=12).options["restart"] == 12
    with pytest.raises(ValueError, match="max_iter is required"):
        krylov.CG()
    with pytest.raises(ValueError, match="max_iter"):
        krylov.CG(max_iter=0)
    with pytest.raises(ValueError, match="tolerance"):
        krylov.Richardson(tolerance=0)
    with pytest.raises(ValueError, match="restart"):
        krylov.GMRES(restart=0)


def test_krylov_declare_amr_route_capabilities():
    # Spec 6 sec.4 / sec.9: the matrix-free Krylov solvers are layout-agnostic (they run over
    # pops::MultiFab dot / saxpy / apply), so each declares every route -- uniform / amr / mpi /
    # gpu -- and a route check can see they are AMR-capable instead of guessing from an empty set.
    for d in (krylov.CG(max_iter=100), krylov.GMRES(max_iter=100),
              krylov.BiCGStab(max_iter=100), krylov.Richardson(max_iter=100)):
        caps = d.capabilities
        assert caps["supports_uniform"] is True
        assert caps["supports_amr"] is True
        assert caps["supports_mpi"] is True
        assert caps["supports_gpu"] is True
        # the capabilities are surfaced by inspect() too.
        assert d.inspect()["capabilities"]["supports_amr"] is True


# --- nonlinear solver descriptors --------------------------------------------------------

def test_nonlinear_newton_descriptor_names_generated_cpp_route():
    d = nonlinear.Newton(tolerance=1e-9, max_iter=12, fd_eps=1e-6, damping=0.8)
    assert d.brick_type == "generated"
    assert d.category == "nonlinear_solver"
    assert d.scheme == "newton"
    assert d.available().ok
    assert d.options["max_iter"] == 12
    assert d.options["tolerance"] == 1e-9
    # Newton lives in the nonlinear subpackage, not as a top-level legacy constructor.
    assert hasattr(solvers.nonlinear, "Newton")
    assert not hasattr(solvers, "Newton")
    assert not hasattr(solvers, "FixedPoint")
    assert not hasattr(solvers.solvers, "Newton")
    assert not hasattr(solvers.solvers, "FixedPoint")


def test_public_solver_catalogs_have_no_notimplemented_placeholders():
    """TASK-040/050/051: exposed solver descriptors must name real compiled routes."""
    for module in (krylov, schur, nonlinear):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "NotImplementedError" not in source, (
            "%s must not expose transitional solver placeholders" % module.__name__)


# --- Schur-condensation solver -----------------------------------------------------------

def test_schur_native_id():
    assert schur.Schur().native_id == "pops::SchurCondensationOperator"
    assert schur.Schur().scheme == "schur"
    condensed = schur.CondensedSchur(theta=0.5, alpha=2.0, tolerance=1e-9, max_iter=150)
    assert condensed.native_id == "pops::CondensedSchurSourceStepper"
    assert condensed.scheme == "condensed_schur"
    assert condensed.category == "schur_solver"
    assert condensed.options["theta"] == 0.5
    assert condensed.options["max_iter"] == 150
    assert not hasattr(solvers, "CondensedSchur")


def test_schur_declares_amr_route_capabilities():
    # Spec 6 sec.4 / sec.9: the Schur-condensation solver runs on AMR (the amr-schur source
    # stage) and System, under MPI and on the GPU, so it declares every route capability.
    caps = schur.Schur().capabilities
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
    assert opts == {"smoother": "chebyshev", "coarse": "direct_small_grid",
                    "tolerance": "relative", "max_cycles": 20}


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
    assert caps["supports_uniform"] is True
    assert caps["supports_amr"] is True
    assert caps["supports_mpi"] is True
    assert caps["supports_gpu"] is True
    assert caps["supports_variable_epsilon"] is True
    assert caps["supports_anisotropic"] is False
    assert caps["supports_screened"] is False


def test_geometric_mg_inspect_and_lower():
    g = elliptic.GeometricMG(smoother=Chebyshev(degree=3))
    view = g.inspect()
    assert view["name"] == "geometric_mg"
    assert view["native_id"] == "pops::GeometricMG"
    assert view["scheme"] == "geometric_mg"
    assert view["available"] is True
    assert view["capabilities"]["supports_amr"] is True
    rec = g.lower()
    assert rec["native_id"] == "pops::GeometricMG"
    assert rec["scheme"] == "geometric_mg"
    assert rec["smoother"] == {"kind": "chebyshev", "degree": 3}
    assert rec["coarse"]["kind"] == "direct_small_grid"
    assert rec["tolerance"]["kind"] == "relative"
    assert rec["max_cycles"] == 20


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
    assert g.capabilities()["supports_amr"] is True
    assert g.available(AMR(base=CartesianMesh(n=64))).status == "yes"


# --- preconditioners ---------------------------------------------------------------------

def test_preconditioners_catalog():
    pre = solvers.preconditioners
    ident = pre.Identity()
    assert ident.available().ok
    assert ident.scheme == "identity"
    assert ident.native_id == ""
    assert pre.GeometricMG().native_id == "pops::GeometricMG"
    assert pre.GeometricMG().category == "preconditioner"
    assert not hasattr(pre, "Jacobi")
    assert not hasattr(pre, "BlockJacobi")


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
    assert ns.GMRES(max_iter=100).scheme == "gmres"
    assert ns.CG(max_iter=100).native_id == "pops::cg_solve"
    assert ns.Schur().native_id == "pops::SchurCondensationOperator"
    assert not hasattr(ns, "Newton")
    assert solvers.preconditioners.GeometricMG().native_id == "pops::GeometricMG"

    # The custom-solver authoring / generation DSL is internal / experimental under
    # pops.codegen.solvers (criterion 19); it is NOT a public attribute of pops.solvers.
    for absent in ("solver", "build_solver_ir", "generate_solver_cpp", "SolverContext", "SolverIR"):
        assert not hasattr(solvers, absent), \
            "pops.solvers is a catalog, not the authoring DSL (saw %r)" % absent
    # The experimental registry stays in pops.codegen.solvers; importing it must not mutate the
    # public pops.solvers catalog.
    cs = pytest.importorskip("pops.codegen.solvers")
    assert getattr(cs, "__experimental__", None) is True
    assert callable(cs.solver)
    assert callable(cs.generate_solver_cpp)
    assert callable(cs.custom_solver)
    assert callable(cs.registered_solvers)
    assert not hasattr(ns, "custom")
    assert not hasattr(ns, "registered")


def test_install_path_token_resolution_for_rich_descriptor():
    # The unified-install solver-token resolver reads .scheme; the new rich GeometricMG
    # resolves to the same 'geometric_mg' token as the brick-catalog pops.fields.catalog descriptor.
    from pops.runtime._system_unified_install import _SystemUnifiedInstall
    assert _SystemUnifiedInstall._solver_token(elliptic.GeometricMG()) == "geometric_mg"
    assert _SystemUnifiedInstall._solver_token(pops.fields.catalog.GeometricMG()) == "geometric_mg"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
