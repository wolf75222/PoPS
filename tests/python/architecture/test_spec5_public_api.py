"""Spec 5 final API alignment (epic ADC-479): the public ``pops`` surface is the canonical one.

This guards the FOUNDATION cleanup of the public bindings surface:

* the top-level compilable assembly is ``pops.Problem`` (Phase 5 re-adopts it as the declarative
  assembly root; the transitional ``pops.Case`` name is gone, with NO back-compat alias) ;
* ``pops.PythonFlux`` is removed from the public surface (it computes a numpy residual in Python,
  which the PoPS "no public Python numeric" rule excludes); it is reachable only as
  ``pops.experimental.PythonFlux`` for residual prototyping in tests ;
* there is no public custom-solver authoring DSL (no ``pops.solver`` / ``pops.lib.solver``
  decorator); the ``@solver`` generation DSL lives only under the internal / experimental
  ``pops.codegen.solvers`` ;
* the solver descriptors have exactly ONE public home, ``pops.solvers`` -- the transitional
  ``pops.lib.solvers`` re-export shim is removed (no second public path) ;
* the public lowering surface takes a ``layout`` (Uniform / AMR), never a ``target=`` kwarg.

The module imports ``pops``, so it needs the compiled ``_pops`` extension. If ``_pops`` cannot be
loaded the whole module is skipped (not failed), so the source-only architecture checks still run
in a bare interpreter.
"""
import pytest

# Skip the whole module when the native extension is unavailable: pops/_bootstrap raises a custom
# ImportError whose .name does not match "pops._pops", so importorskip would re-raise instead of
# skipping. Catch any import failure and skip at module level.
try:
    import pops._pops  # noqa: F401
except Exception as _exc:  # pragma: no cover - only without a built extension
    pytest.skip("compiled _pops extension not importable: %s" % _exc, allow_module_level=True)

import pops  # noqa: E402


def test_problem_is_the_assembly_root():
    # Phase 5 re-adopts pops.Problem as the declarative assembly root; pops.Case is gone with NO
    # back-compat alias (ADC-553/ADC-526/ADC-523: no soft-compat alias).
    assert hasattr(pops, "Problem"), "pops.Problem must be the top-level compilable assembly"
    assert not hasattr(pops, "Case"), "pops.Case must be gone (renamed to pops.Problem, no alias)"
    assert "Problem" in pops.__all__, "Problem must be exported in pops.__all__"
    assert "Case" not in pops.__all__, "Case must not linger in pops.__all__"
    # pops.Case must raise AttributeError (not be a silently-aliased attribute), pointing at Problem.
    with pytest.raises(AttributeError) as excinfo:
        pops.Case  # noqa: B018
    assert "pops.Problem" in str(excinfo.value), "the AttributeError must point at pops.Problem"


def test_problem_keeps_the_assembly_chaining_surface():
    # The rename preserves the authoring surface (chaining setters + inspect/route/lower path).
    problem = pops.Problem(name="arch")
    for member in ("block", "field", "param", "aux", "output", "time", "layout",
                   "validate", "inspect", "explain_routes", "available", "requirements",
                   "capabilities", "lower"):
        assert hasattr(problem, member), "pops.Problem lost the %r surface member" % member
    # ``amr`` is a property that raises for a non-AMR layout, so probe the class, not the instance.
    assert hasattr(type(problem), "amr"), "pops.Problem lost the .amr handle"
    assert problem.category == "problem"
    assert "arch" in repr(problem) and repr(problem).startswith("Problem(")


def test_field_problem_classes_are_untouched():
    # FieldProblem / PoissonProblem / LinearProblem are a DIFFERENT concept and must NOT be renamed.
    from pops.fields import FieldProblem  # noqa: PLC0415
    from pops.linalg import LinearProblem  # noqa: PLC0415

    assert FieldProblem is not None
    assert LinearProblem is not None


def test_python_flux_is_off_the_public_surface():
    # PythonFlux computes a numpy residual in Python: it is excluded from the public pops surface.
    assert not hasattr(pops, "PythonFlux"), "pops.PythonFlux must be removed from the public surface"
    assert "PythonFlux" not in pops.__all__, "PythonFlux must not linger in pops.__all__"
    with pytest.raises(AttributeError):
        pops.PythonFlux  # noqa: B018


def test_python_flux_is_reachable_only_under_experimental():
    # The TESTS-ONLY backend is reachable under pops.experimental (for residual prototyping).
    from pops.experimental import PythonFlux  # noqa: PLC0415

    assert PythonFlux is not None
    assert pops.experimental.PythonFlux is PythonFlux
    # The package advertises itself as non-stable.
    assert getattr(pops.experimental, "__experimental__", False) is True


def test_install_is_not_the_public_path():
    # Spec 5 sec.11 (epic ADC-479, item #3): `install` is no longer a public method on
    # System / AmrSystem; the internal seam is `_install_compiled`, and pops.bind is the
    # documented entry. The public surface advertises pops.compile / pops.bind, not install.
    # ADC-545: the engines left the top-level surface -- reach them via the advanced seam.
    from pops.runtime.system import System, AmrSystem  # ADC-545 advanced runtime seam

    assert not hasattr(pops, "System") and not hasattr(pops, "AmrSystem"), (
        "pops.System / pops.AmrSystem must be gone from the top-level surface (ADC-545)")
    sim = System()
    assert not hasattr(sim, "install"), "System.install must be gone (renamed to _install_compiled)"
    assert hasattr(sim, "_install_compiled"), "the internal install seam is _install_compiled"
    amr = AmrSystem(n=8, L=1.0)
    assert not hasattr(amr, "install"), "AmrSystem.install must be gone (renamed to _install_compiled)"
    assert hasattr(amr, "_install_compiled"), "AmrSystem keeps the internal _install_compiled seam"
    # ADC-660 exposes the typed final install phase, never an engine mutation method.
    assert {"validate", "resolve", "compile", "bind", "install"} <= set(pops.__all__)
    assert "install_program" not in pops.__all__


def test_physics_model_is_a_writing_facade_not_a_compiler():
    # Spec 5 sec.11 (item #7): pops.physics.Model authors physics and LOWERS to a
    # pops.model.Module; it has NO public compile_* method. pops.compile does the compile.
    from pops import model as model_pkg

    m = pops.physics.Model("arch")
    for forbidden in ("compile", "compile_so", "compile_aot", "compile_native", "compile_or_jit"):
        assert not hasattr(m, forbidden), (
            "pops.physics.Model must not expose %r (it is a writing facade, not a compiler)"
            % forbidden)
    # lower() / to_module() return the pops.model.Module pops.compile / compile_problem accept.
    assert hasattr(m, "lower") and hasattr(m, "to_module"), "physics.Model needs lower()/to_module()"
    module = m.lower()
    assert isinstance(module, model_pkg.Module), "physics.Model.lower() returns a pops.model.Module"
    assert isinstance(m.to_module(), model_pkg.Module), "to_module() returns a pops.model.Module too"
    # to_module IS the lower method (Spec 5 sec.11 alias), not a re-implementation.
    assert type(m).to_module is type(m).lower, "to_module() must be the lower() alias"


def test_no_public_custom_solver_decorator():
    # The custom-solver authoring DSL (@solver) is not a user API on pops / pops.lib / pops.solvers.
    import pops.lib  # noqa: PLC0415
    import pops.solvers  # noqa: PLC0415

    assert not hasattr(pops, "solver"), "there must be no top-level pops.solver decorator"
    assert not hasattr(pops.lib, "solver"), "there must be no pops.lib.solver decorator"
    assert not hasattr(pops.solvers, "solver"), "pops.solvers is a catalog, not the authoring DSL"


def test_solvers_have_one_public_home_no_lib_shim():
    # No-soft-compat: the solver descriptors live in exactly ONE public home, pops.solvers. The
    # transitional pops.lib.solvers re-export shim is REMOVED -- importing it fails, and the
    # descriptors are NOT reachable through pops.lib (no second public path).
    import importlib  # noqa: PLC0415

    import pops.lib  # noqa: PLC0415
    import pops.solvers  # noqa: PLC0415

    # The one public home resolves every solver descriptor.
    from pops.solvers import CG, GMRES, GeometricMG, Newton, Schur  # noqa: F401,PLC0415

    # The shim module is gone: importing it raises, and pops.lib exposes no solvers / preconditioners.
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pops.lib.solvers")
    assert not hasattr(pops.lib, "solvers"), "pops.lib must not re-export the solver catalog"
    assert not hasattr(pops.lib, "preconditioners"), "pops.lib must not re-export preconditioners"
    # The descriptor attributes are NOT a second public path under pops.lib.solvers.
    for name in ("CG", "GMRES", "GeometricMG", "Newton", "Schur", "BiCGStab", "FixedPoint"):
        assert hasattr(pops.solvers, name), "pops.solvers is the one public home (missing %r)" % name


def test_preconditioners_home_ratified_in_solvers_not_linalg():
    # ADC-502: the preconditioner catalog lives in exactly ONE public home,
    # pops.solvers.preconditioners (a preconditioner configures a solver). pops.linalg exposes NO
    # preconditioners submodule and no shim exists -- no second public path, no move.
    import importlib  # noqa: PLC0415

    import pops.linalg  # noqa: PLC0415
    from pops.solvers.preconditioners import preconditioners  # noqa: PLC0415

    # The one public home resolves the catalog.
    assert hasattr(preconditioners, "GeometricMG"), "pops.solvers.preconditioners is the one home"
    # pops.linalg must NOT expose (or re-export) a preconditioners catalog.
    assert not hasattr(pops.linalg, "preconditioners"), (
        "pops.linalg must not expose preconditioners (ADC-502: they live under pops.solvers)")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pops.linalg.preconditioners")


def test_no_public_target_kwarg_on_compile_or_bind():
    # Spec 5 sec.11 (#5): the LAYOUT (Uniform / AMR) chooses the runtime; a user never passes
    # target=. The public lowering entry points pops.compile / pops.bind take no target kwarg.
    import inspect  # noqa: PLC0415

    for fn in (pops.compile, pops.bind):
        sig = inspect.signature(fn)
        assert "target" not in sig.parameters, (
            "%s must not accept a public target= kwarg (the layout picks the runtime)" % fn.__name__)
    # The public assembly pops.Problem has no compile / install / target surface either.
    problem = pops.Problem(name="arch")
    for forbidden in ("compile", "install", "target"):
        assert not hasattr(problem, forbidden), "pops.Problem must not expose %r" % forbidden
    # pops.physics.Model (the writing facade) lowers; it has no target= path.
    pm = pops.physics.Model("arch")
    assert not hasattr(pm, "target"), "pops.physics.Model must not expose a target surface"


def test_compile_problem_off_public_surface():
    # ADC-660 removes parallel low-level compiler exports from both public surfaces.
    assert "compile" in pops.__all__ and "bind" in pops.__all__
    for removed in ("compile_problem", "CompiledProblem"):
        assert removed not in pops.__all__, "pops.%s must not linger in pops.__all__" % removed
        assert not hasattr(pops, removed), "pops.%s must be gone from the top-level surface" % removed
        with pytest.raises(AttributeError) as excinfo:
            getattr(pops, removed)
        msg = str(excinfo.value)
        assert "pops.compile" in msg, "the AttributeError must point at pops.compile (got %r)" % msg
        assert "typed phase API" in msg
    import importlib  # noqa: PLC0415

    codegen = importlib.import_module("pops.codegen")
    assert not hasattr(codegen, "compile_problem") and not hasattr(codegen, "CompiledProblem")
    assert pops.CompiledSimulationArtifact is codegen.CompiledSimulationArtifact


def test_layout_is_resolved_before_compile():
    import inspect  # noqa: PLC0415

    assert tuple(inspect.signature(pops.resolve).parameters)[:2] == ("problem", "layout")
    assert tuple(inspect.signature(pops.compile).parameters) == ("plan",)


def test_solver_generation_dsl_is_internal_experimental():
    # The @solver generation DSL lives ONLY under the internal / experimental pops.codegen.solvers.
    import pops.codegen.solvers as codegen_solvers  # noqa: PLC0415

    assert getattr(codegen_solvers, "__experimental__", False) is True
    assert hasattr(codegen_solvers, "solver"), (
        "the @solver authoring DSL must still live under the internal pops.codegen.solvers")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
