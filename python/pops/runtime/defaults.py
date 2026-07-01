"""Structured native numerical/solver/physical defaults."""


def _static_report():
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
        "eb": {"cut_fraction_floor": 1e-3},
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


def numerical_defaults_report():
    """Return structured native numerical, solver and physical defaults."""
    try:
        from pops import _pops  # noqa: PLC0415

        fn = getattr(_pops, "numerical_defaults_report", None)
        if callable(fn):
            return dict(fn())
    except Exception:
        pass
    return _static_report()


_DEFAULTS = numerical_defaults_report()
_NEWTON = _DEFAULTS["newton"]
_PHYSICAL = _DEFAULTS["physical"]

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
