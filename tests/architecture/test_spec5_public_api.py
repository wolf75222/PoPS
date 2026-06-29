"""Final Spec corrective guards for the public PoPS API.

The current public route is intentionally narrow:

* Python authors typed models/programs.
* :func:`pops.compile_problem` builds the compiled artifact.
* :meth:`pops.System.install` / :meth:`pops.AmrSystem.install` wire that artifact to runtime data.
* C++/Kokkos/MPI executes all numerical work.

This file deliberately rejects transitional front doors (`Case`, `compile`, `bind`,
`CompiledTime`, public runtime setters) because keeping two ways to do the same thing is
exactly the legacy surface the corrective spec removes.
"""
import importlib

import pytest

try:
    import pops._pops  # noqa: F401
except Exception as _exc:  # pragma: no cover - only without a built extension
    pytest.skip("compiled _pops extension not importable: %s" % _exc, allow_module_level=True)

import pops  # noqa: E402


def test_top_level_public_surface_is_single_route():
    allowed = {
        "__version__",
        "System",
        "AmrSystem",
        "time",
        "model",
        "math",
        "physics",
        "moments",
        "lib",
        "abi_key",
        "set_threads",
        "has_kokkos",
        "parallel_info",
        "doctor",
        "compile_problem",
        "CompiledProblem",
        "compile_library",
        "read_library_manifest",
        "LibraryManifest",
        "inspect_capabilities",
        "CapabilityMatrix",
        "CapabilityEntry",
    }
    assert set(pops.__all__) == allowed

    for forbidden in (
        "Case",
        "Problem",
        "compile",
        "bind",
        "integrate",
        "PythonFlux",
        "CompiledTime",
        "Explicit",
        "IMEX",
        "IMEXRK",
        "SourceImplicit",
        "SourceImplicitBE",
        "Split",
        "Strang",
        "ElectrostaticLorentzSchur",
        "CompositeModel",
        "FluidState",
        "FiniteVolume",
        "Spatial",
        "Role",
        "Profile",
        "PerformanceSummary",
        "EllipticSolver",
        "solver",
    ):
        assert forbidden not in pops.__all__
        assert not hasattr(pops, forbidden), "pops.%s must not be public" % forbidden


def test_runtime_install_is_the_only_public_runtime_wiring_path():
    for sim in (pops.System(), pops.AmrSystem(n=8, L=1.0)):
        assert hasattr(sim, "install"), "%s.install is the explicit runtime API" % type(sim).__name__
        assert hasattr(sim, "_install_compiled"), "%s keeps one private compiled-install seam" % (
            type(sim).__name__,)
        for forbidden in (
            "add_block",
            "add_equation",
            "install_program",
            "initialize_compiled_program",
            "set_program_cadence",
            "set_param",
            "set_aux_field",
            "set_field_solver",
            "set_poisson",
            "set_disc_domain",
            "set_geometry_mode",
            "eval_rhs",
            "get_state",
            "set_state",
        ):
            assert not hasattr(sim, forbidden), "%s.%s must not be public" % (
                type(sim).__name__, forbidden)


def test_no_public_python_runtime_integrator_or_time_shim():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pops.runtime.integrate")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pops.codegen.orchestration")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pops.case")
    assert not hasattr(pops.time, "CompiledTime")
    with pytest.raises(ImportError):
        from pops.time.program import CompiledTime  # noqa: F401


def test_program_public_surface_is_operator_first_only():
    P = pops.time.Program("arch")
    assert hasattr(P, "call"), "Program must expose typed operator calls"
    assert hasattr(P, "define"), "Program must expose T.define(...) temporal SSA sugar"

    for forbidden in ("solve_fields", "rhs", "source", "linear_source"):
        assert not hasattr(P, forbidden), "Program.%s must not be public" % forbidden

    u = P.state("U", block="plasma").n
    with pytest.raises(TypeError, match="string operator selectors"):
        P.apply("lorentz", state=u)

    # The old private helpers must not stay as stable architecture names.
    for forbidden in ("_solve_fields", "_rhs_legacy"):
        assert not hasattr(P, forbidden), "Program.%s must be removed, not documented" % forbidden


def test_model_module_has_no_old_dsl_lowering_surface():
    module = pops.model.Module("arch")
    assert not hasattr(module, "to_dsl")
    compile_module = importlib.import_module("pops.codegen.compile_drivers")
    assert not hasattr(compile_module, "_module_to_model")


def test_legacy_m_facade_cannot_enter_modern_compile_or_install_routes():
    legacy = importlib.import_module("pops.physics.facade").Model("legacy")
    program = pops.time.Program("arch")
    u = program.state("U", block="plasma").n
    program.commit("plasma", program.linear_combine("identity", u))

    with pytest.raises(TypeError, match="private _m"):
        pops.compile_problem(model=legacy, time=program)

    with pytest.raises(TypeError, match="private _m"):
        pops.System()._resolve_instance_model(legacy)


def test_physics_model_is_a_writing_facade_not_a_compiler():
    from pops import model as model_pkg

    m = pops.physics.Model("arch")
    for forbidden in (
        "compile",
        "compile_so",
        "compile_aot",
        "compile_native",
        "compile_or_jit",
        "dsl",
        "_dsl",
    ):
        assert not hasattr(m, forbidden), (
            "pops.physics.Model must not expose %r; it authors a Module only" % forbidden)
    assert hasattr(m, "lower") and hasattr(m, "to_module")
    assert isinstance(m.lower(), model_pkg.Module)
    assert isinstance(m.to_module(), model_pkg.Module)
    assert type(m).to_module is type(m).lower


def test_physics_package_does_not_reexport_codegen_engines():
    forbidden = (
        "PdeModel",
        "HyperbolicModel",
        "HybridModel",
        "NativeBrick",
        "CompiledBrick",
        "CompiledHyperbolicBrick",
        "CompiledSourceBrick",
        "CompiledEllipticBrick",
        "HyperbolicBrick",
        "SourceBrick",
        "EllipticBrick",
        "CompiledCoupledSource",
    )
    for name in forbidden:
        assert name not in pops.physics.__all__
        assert not hasattr(pops.physics, name), "pops.physics.%s must not be public" % name


def test_physics_params_are_typed_no_public_kind_param():
    assert "Param" not in pops.physics.__all__
    assert not hasattr(pops.physics, "Param")
    assert hasattr(pops.physics, "ConstParam")
    assert hasattr(pops.physics, "RuntimeParam")


def test_moments_toolkit_has_one_public_home_at_top_level():
    moments = importlib.import_module("pops.moments")
    assert moments.CartesianVelocityMoments is not None
    assert moments.MomentModel is not None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pops.lib.moments")
    assert hasattr(pops, "moments")
    assert not hasattr(pops.lib, "moments")


def test_no_public_custom_solver_decorator_or_lib_solver_shim():
    import pops.lib  # noqa: PLC0415
    import pops.solvers  # noqa: PLC0415

    assert not hasattr(pops, "solver")
    assert not hasattr(pops.lib, "solver")
    assert not hasattr(pops.solvers, "solver")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pops.lib.solvers")
    assert not hasattr(pops.lib, "solvers")
    assert not hasattr(pops.lib, "preconditioners")


def test_solvers_have_one_public_home():
    import pops.solvers  # noqa: PLC0415
    from pops.solvers import BiCGStab, CG, GMRES, GeometricMG, Schur  # noqa: F401,PLC0415

    for name in ("BiCGStab", "CG", "GMRES", "GeometricMG", "Schur"):
        assert hasattr(pops.solvers, name)
    for name in ("Newton", "FixedPoint"):
        assert not hasattr(pops.solvers, name), "%s must not be a public placeholder" % name


def test_solver_generation_dsl_is_internal_experimental():
    import pops.codegen.solvers as codegen_solvers  # noqa: PLC0415

    assert getattr(codegen_solvers, "__experimental__", False) is True
    assert hasattr(codegen_solvers, "solver")
