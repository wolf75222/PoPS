"""Structured native numerical/solver/physical defaults."""
from __future__ import annotations

from typing import Any


# ADC-618: the classification of EVERY user-visible native numeric constant. This is the single
# Python-side source of truth (mirrored by numerical_defaults_report_to_dict in C++). The
# source-scanning fence test asserts no inline constexpr numeric constant of numerical_defaults.hpp /
# types.hpp / runtime_params.hpp is missing here. Classes: public_knob (configurable end to end),
# internal_default (fixed but inspectable), diagnostic_only (a counter / instrumented fact),
# hard_limit (a fixed cap enforced fail-fast).
_CONSTANT_CLASSIFICATION: dict = {
    "kNewtonFailNone": "internal_default",
    "kNewtonFailWarn": "internal_default",
    "kNewtonFailThrow": "internal_default",
    "kNewtonDefaultMaxIters": "public_knob",
    "kNewtonDefaultRelTol": "public_knob",
    "kNewtonDefaultAbsTol": "public_knob",
    "kNewtonDefaultFdEps": "public_knob",
    "kNewtonDefaultDamping": "public_knob",
    "kNewtonDefaultFailPolicy": "public_knob",
    "kNewtonFiniteAbsLimit": "internal_default",
    "kKrylovDefaultRelTol": "public_knob",
    "kTensorKrylovDefaultMaxIters": "internal_default",
    "kSchurKrylovCartesianMaxIters": "public_knob",
    "kSchurKrylovPolarMaxIters": "public_knob",
    "kKrylovBreakdownTiny": "internal_default",
    "kMGDefaultRelTol": "public_knob",
    "kMGDefaultMaxCycles": "public_knob",
    "kMGDefaultAbsTol": "public_knob",
    "kMGDefaultMinCoarse": "public_knob",
    "kMGDefaultPreSmooth": "public_knob",
    "kMGDefaultPostSmooth": "public_knob",
    "kMGDefaultBottomSweeps": "public_knob",
    "kFACDefaultMaxIters": "public_knob",
    "kFACDefaultFineSweeps": "public_knob",
    "kFACDefaultTol": "public_knob",
    "kFACInitialCoarseRelTol": "public_knob",
    "kFACInitialCoarseMaxCycles": "public_knob",
    "kFFTDefaultSpectral": "public_knob",
    "kFFTZeroMeanGauge": "internal_default",
    "kFFTDirectDftFallback": "diagnostic_only",
    "kEbCutFractionFloor": "public_knob",
    "kWenoEpsilon": "internal_default",
    "kEbFaceOpenEps": "public_knob",
    "kEbKappaMin": "public_knob",
    "kAmrDefaultMaxLevels": "internal_default",
    "kAmrRefinementDisabledThreshold": "internal_default",
    "kAmrPhiRefinementDisabledThreshold": "internal_default",
    "kAdaptiveNoEvolvingBlockSentinel": "diagnostic_only",
    "kAmrClusterMinEfficiency": "public_knob",
    "kAmrClusterMinBoxSize": "public_knob",
    "kAmrClusterMaxBoxSize": "public_knob",
    "kAmrDriftSpeedFloor": "internal_default",
    "kPhysicalDefaultB0": "public_knob",
    "kPhysicalDefaultGamma": "public_knob",
    "kPhysicalDefaultFluidStateCs2": "public_knob",
    "kPhysicalDefaultNativeIsothermalCs2": "internal_default",
    "kPhysicalDefaultVacuumFloor": "public_knob",
    "kPhysicalDefaultQOverM": "public_knob",
    "kPhysicalDefaultChargeQ": "public_knob",
    "kPhysicalDefaultAlpha": "public_knob",
    "kPhysicalDefaultBackgroundN0": "public_knob",
    "kPhysicalDefaultGravitySign": "public_knob",
    "kPhysicalDefaultFourPiG": "public_knob",
    "kPhysicalDefaultGravityRho0": "public_knob",
    "kCflSpeedFloor": "internal_default",
    "kMaxRuntimeParams": "hard_limit",
}


def _static_report() -> dict:
    return {
        "schema_version": 1,
        "source": "pops.runtime.defaults.static_fallback",
        "newton": {
            "max_iters": 2,
            "rel_tol": 0.0,
            "abs_tol": 0.0,
            "fd_eps": 1e-7,
            "damping": 1.0,
            "fail_policy": "none",
            "finite_abs_limit": 1e300,
        },
        "krylov": {
            "rel_tol": 1e-10,
            "tensor_max_iters": 200,
            "schur_cartesian_max_iters": 400,
            "schur_polar_max_iters": 600,
            "breakdown_tiny": 1e-300,
        },
        "mg": {
            "rel_tol": 1e-8,
            "max_cycles": 50,
            "abs_tol": 0.0,
            "min_coarse": 2,
            "pre_smooth": 2,
            "post_smooth": 2,
            "bottom_sweeps": 50,
        },
        "fac": {
            "max_iters": 30,
            "fine_sweeps": 400,
            "tol": 1e-9,
            "initial_coarse_rel_tol": 1e-12,
            "initial_coarse_max_cycles": 100,
        },
        "fft": {
            "spectral_default": False,
            "zero_mean_gauge": True,
            "direct_dft_fallback": True,
        },
        "eb": {"cut_fraction_floor": 1e-3, "face_open_eps": 1e-6, "kappa_min": 1e-2},
        "weno": {"epsilon": 1e-40},
        "performance": {
            "cfl_speed_floor": 1e-30,
            "adaptive_no_evolving_block_sentinel": 1e30,
        },
        "amr": {
            "max_levels": 2,
            "refinement_ratio": 2,
            "refinement_disabled_threshold": 1e30,
            "phi_refinement_disabled_threshold": 0.0,
        },
        "runtime": {"max_runtime_params": 32},
        "diagnostics": {"fft_direct_dft_fallback_count": 0},
        # ADC-618: the classification fence (mirror of numerical_defaults_report_to_dict). The
        # source-scanning architecture test asserts every user-visible inline constexpr numeric
        # constant of numerical_defaults.hpp / types.hpp / runtime_params.hpp appears here.
        "classification": _CONSTANT_CLASSIFICATION,
        "physical": {
            "preset": "legacy_native_brick_defaults",
            "B0": 1.0,
            "gamma": 1.4,
            "fluid_state_cs2": 0.5,
            "native_brick_isothermal_cs2": 1.0,
            "vacuum_floor": 0.0,
            "qom": 1.0,
            "charge_q": 1.0,
            "alpha": 1.0,
            "n0": 0.0,
            "gravity_sign": 1.0,
            "four_pi_G": 1.0,
            "gravity_rho0": 1.0,
            "cs2_note": (
                "FluidState defaults to 0.5 while the raw native IsothermalFlux brick "
                "defaults to 1.0."
            ),
        },
    }


def numerical_defaults_report() -> dict:
    """Return structured native numerical, solver and physical defaults."""
    try:
        from pops import _pops  # noqa: PLC0415

        fn: Any = getattr(_pops, "numerical_defaults_report", None)
        if callable(fn):
            report: Any = fn()
            return dict(report)
    except Exception:
        pass
    return _static_report()


_DEFAULTS: Any = numerical_defaults_report()
_NEWTON: Any = _DEFAULTS["newton"]
_PHYSICAL: Any = _DEFAULTS["physical"]

NEWTON_DEFAULT_MAX_ITERS = int(_NEWTON["max_iters"])
NEWTON_DEFAULT_REL_TOL = float(_NEWTON["rel_tol"])
NEWTON_DEFAULT_ABS_TOL = float(_NEWTON["abs_tol"])
NEWTON_DEFAULT_FD_EPS = float(_NEWTON["fd_eps"])
NEWTON_DEFAULT_DAMPING = float(_NEWTON["damping"])
NEWTON_DEFAULT_FAIL_POLICY = str(_NEWTON["fail_policy"])
PHYSICAL_DEFAULT_B0 = float(_PHYSICAL["B0"])
PHYSICAL_DEFAULT_GAMMA = float(_PHYSICAL["gamma"])
PHYSICAL_DEFAULT_FLUID_STATE_CS2 = float(_PHYSICAL["fluid_state_cs2"])
PHYSICAL_DEFAULT_NATIVE_ISOTHERMAL_CS2 = float(_PHYSICAL["native_brick_isothermal_cs2"])
PHYSICAL_DEFAULT_VACUUM_FLOOR = float(_PHYSICAL["vacuum_floor"])
PHYSICAL_DEFAULT_QOM = float(_PHYSICAL["qom"])
PHYSICAL_DEFAULT_CHARGE_Q = float(_PHYSICAL["charge_q"])
PHYSICAL_DEFAULT_ALPHA = float(_PHYSICAL["alpha"])
PHYSICAL_DEFAULT_BACKGROUND_N0 = float(_PHYSICAL["n0"])
PHYSICAL_DEFAULT_GRAVITY_SIGN = float(_PHYSICAL["gravity_sign"])
PHYSICAL_DEFAULT_FOUR_PI_G = float(_PHYSICAL["four_pi_G"])
PHYSICAL_DEFAULT_GRAVITY_RHO0 = float(_PHYSICAL["gravity_rho0"])


__all__ = [
    "numerical_defaults_report",
    "NEWTON_DEFAULT_MAX_ITERS",
    "NEWTON_DEFAULT_REL_TOL",
    "NEWTON_DEFAULT_ABS_TOL",
    "NEWTON_DEFAULT_FD_EPS",
    "NEWTON_DEFAULT_DAMPING",
    "NEWTON_DEFAULT_FAIL_POLICY",
    "PHYSICAL_DEFAULT_B0",
    "PHYSICAL_DEFAULT_GAMMA",
    "PHYSICAL_DEFAULT_FLUID_STATE_CS2",
    "PHYSICAL_DEFAULT_NATIVE_ISOTHERMAL_CS2",
    "PHYSICAL_DEFAULT_VACUUM_FLOOR",
    "PHYSICAL_DEFAULT_QOM",
    "PHYSICAL_DEFAULT_CHARGE_Q",
    "PHYSICAL_DEFAULT_ALPHA",
    "PHYSICAL_DEFAULT_BACKGROUND_N0",
    "PHYSICAL_DEFAULT_GRAVITY_SIGN",
    "PHYSICAL_DEFAULT_FOUR_PI_G",
    "PHYSICAL_DEFAULT_GRAVITY_RHO0",
]
