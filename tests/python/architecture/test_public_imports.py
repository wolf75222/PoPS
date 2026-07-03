"""Spec 4 (36.4): the public import surface must work.

These are the imports a user (and ``adc_cases``) is promised. They exercise the package
boundaries: the physics facade, the time-program facade, the time-scheme library, the
ready-made moment models, the moment-model construction kit, and the top-level runtime
entry points.

Unlike the other architecture tests, this one IMPORTS ``pops`` and therefore needs the
compiled ``_pops`` extension (drop the prebuilt ``.so`` into ``python/pops`` and run with
the matching interpreter). If ``_pops`` cannot be loaded the test is skipped, not failed,
so the source-only checks still run in a bare interpreter.
"""
import importlib

import pytest

# Skip the whole module if the native extension cannot be loaded in this interpreter.
# importorskip is too strict here: pops/_bootstrap raises a custom ImportError whose .name
# does not match "pops._pops", so importorskip would re-raise instead of skipping. Catch any
# import failure and skip at module level so the source-only checks still run bare.
try:
    import pops._pops  # noqa: F401
except Exception as _exc:  # pragma: no cover - exercised only without a built extension
    pytest.skip("compiled _pops extension not importable: %s" % _exc, allow_module_level=True)


def test_physics_model():
    from pops.physics import Model

    assert Model is not None


def test_time_program():
    from pops.time import Program

    assert Program is not None


def test_lib_time_scheme():
    from pops.lib.time import predictor_corrector_local_linear

    assert callable(predictor_corrector_local_linear)


def test_lib_models_moments_hyqmom15():
    from pops.lib.models.moments import HyQMOM15

    assert HyQMOM15 is not None


def test_lib_presets():
    # ADC-524: pops.lib.presets is the home for ready-to-run compose-and-go bundles. A preset really
    # composes a provided model and a provided time scheme (not a stub).
    from pops.lib.presets import Preset, vlasov_poisson_magnetic_euler
    from pops.time import Program

    assert Preset is not None
    preset = vlasov_poisson_magnetic_euler()
    assert preset.category == "preset"
    assert preset.model() is not None
    assert isinstance(preset.time_scheme("f"), Program)


def test_moments_kit():
    # Spec 5 (sec.4): the moment-model construction kit lives in the top-level pops.moments.
    from pops.moments import CartesianVelocityMoments, MomentModel

    assert CartesianVelocityMoments is not None
    assert MomentModel is not None


def test_numerics_and_diagnostics_packages():
    # Spec 5 (sec.4): discretisation + diagnostics catalogs are top-level packages now.
    from pops.numerics.riemann import HLL
    from pops.numerics.reconstruction import MUSCL
    from pops.numerics.reconstruction.limiters import Minmod
    from pops.diagnostics import norm

    assert HLL().native_id == "pops::HLLFlux"
    assert MUSCL().scheme == "minmod"
    assert Minmod().native_id == "pops::Minmod"
    assert norm().category == "diagnostic"


def test_top_level_runtime_entry_points():
    pops = importlib.import_module("pops")
    # ADC-545: the runtime engines left the top-level surface; pops.System / pops.AmrSystem raise.
    assert not hasattr(pops, "System") and not hasattr(pops, "AmrSystem")
    for name in ("System", "AmrSystem"):
        with pytest.raises(AttributeError):
            getattr(pops, name)
    # The engines stay reachable as the advanced runtime seam.
    from pops.runtime.system import System, AmrSystem  # ADC-545 advanced runtime seam

    assert System is not None and AmrSystem is not None
    # ADC-523: pops.compile / pops.bind are the public front doors; the low-level compile_problem
    # driver left the top-level surface (still reachable as pops.codegen.compile_problem).
    assert pops.compile is not None and pops.bind is not None
    with pytest.raises(AttributeError):
        pops.compile_problem  # noqa: B018


def test_adc545_retired_names_raise_and_advanced_seams_import():
    # ADC-545: eight names left the pops root. Each raises a targeted AttributeError naming its
    # advanced home, and every advanced seam still imports (behaviour unchanged behind it).
    pops = importlib.import_module("pops")
    homes = {
        "System": "pops.runtime.system", "AmrSystem": "pops.runtime.system",
        "SystemConfig": "pops._bootstrap", "AmrSystemConfig": "pops._bootstrap",
        "CompiledTime": "pops.time", "compile_library": "pops.codegen",
        "read_library_manifest": "pops.codegen", "LibraryManifest": "pops.codegen",
    }
    for name, home in homes.items():
        assert name not in pops.__all__, "%s must not linger in pops.__all__" % name
        assert not hasattr(pops, name), "pops.%s must be gone from the top-level surface" % name
        with pytest.raises(AttributeError) as excinfo:
            getattr(pops, name)
        msg = str(excinfo.value)
        assert "ADC-545" in msg, "the AttributeError for %s must cite ADC-545 (got %r)" % (name, msg)
        assert home in msg, "the AttributeError for %s must name %s (got %r)" % (name, home, msg)
    # The advanced seams import (the engines / configs / time policy / library manifest API).
    from pops.runtime.system import System, AmrSystem, SystemConfig, AmrSystemConfig  # noqa: F401
    from pops.time import CompiledTime  # noqa: F401
    from pops.codegen import compile_library, read_library_manifest, LibraryManifest  # noqa: F401

    assert all(obj is not None for obj in (
        System, AmrSystem, SystemConfig, AmrSystemConfig, CompiledTime,
        compile_library, read_library_manifest, LibraryManifest))


def test_adc545_compile_backend_default_is_typed_production():
    # ADC-545: pops.compile no longer defaults backend= to the "production" string; it resolves the
    # typed Production() in-body (signature default None), and a bare string is refused with a
    # TypeError naming Production() (byte-identical lowering to the same token).
    import inspect  # noqa: PLC0415

    pops = importlib.import_module("pops")
    default = inspect.signature(pops.compile).parameters["backend"].default
    assert default is None, "backend= must default to None (lazy Production() resolve)"
    from pops.codegen.backends import Production, lower_backend  # noqa: PLC0415

    assert Production().lower() == "production" == lower_backend(Production())
    from pops.problem import Problem  # noqa: PLC0415

    with pytest.raises(TypeError) as excinfo:
        pops.compile(Problem(name="arch_backend"), backend="production")
    assert "Production()" in str(excinfo.value), "the TypeError must name Production()"
