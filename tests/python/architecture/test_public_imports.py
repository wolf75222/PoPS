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
    assert pops.System is not None
    # ADC-523: pops.compile / pops.bind are the public front doors; the low-level compile_problem
    # driver left the top-level surface (still reachable as pops.codegen.compile_problem).
    assert pops.compile is not None and pops.bind is not None
    with pytest.raises(AttributeError):
        pops.compile_problem  # noqa: B018
