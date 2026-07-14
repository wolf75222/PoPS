"""ADC-603 numerical defaults and effective options reports."""

import math

import pytest
from pops.runtime._system import AmrSystem, System  # ADC-545 advanced runtime seam

pytest.importorskip("pops")
import pops.runtime._engine_descriptors as engine
from pops.runtime.defaults import numerical_defaults_report


def _isothermal_model(cs2=0.7, charge=-2.0):
    return engine.Model(
        engine.FluidState.isothermal(cs2=cs2),
        engine.IsothermalFlux(),
        engine.NoSource(),
        engine.ChargeDensity(charge=charge),
    )


def test_numerical_defaults_report_is_structured():
    d = numerical_defaults_report()
    assert d["schema_version"] == 1
    assert d["newton"]["max_iters"] == 2
    assert d["newton"]["fd_eps"] == pytest.approx(1e-7)
    assert d["krylov"]["schur_cartesian_max_iters"] == 400
    assert d["krylov"]["schur_polar_max_iters"] == 600
    assert d["mg"]["rel_tol"] == pytest.approx(1e-8)
    assert d["mg"]["max_cycles"] == 50
    assert d["fac"]["rel_tol"] == pytest.approx(1e-9)
    assert d["fac"]["abs_tol"] == pytest.approx(0.0)
    assert d["fac"]["coarse_rel_tol"] == pytest.approx(1e-12)
    assert d["fac"]["coarse_abs_tol"] == pytest.approx(0.0)
    assert d["fac"]["coarse_cycles"] == 100
    assert d["fft"]["zero_mean_gauge"] is True
    assert d["eb"]["cut_fraction_floor"] == pytest.approx(1e-3)
    assert d["eb"]["face_open_eps"] == pytest.approx(1e-6)  # ADC-615
    assert d["eb"]["kappa_min"] == pytest.approx(1e-2)
    assert d["weno"]["epsilon"] == pytest.approx(1e-40)
    assert d["physical"]["gamma"] == pytest.approx(1.4)
    assert d["physical"]["B0"] == pytest.approx(1.0)
    assert d["physical"]["charge_q"] == pytest.approx(1.0)
    assert d["physical"]["fluid_state_cs2"] == pytest.approx(0.5)
    assert d["physical"]["native_brick_isothermal_cs2"] == pytest.approx(1.0)


def test_numerical_defaults_report_classifies_every_constant():
    """ADC-618: the C++ report agrees with the Python-side classification single source of truth."""
    from pops.runtime.defaults import _CONSTANT_CLASSIFICATION

    d = numerical_defaults_report()
    classification = d["classification"]
    allowed = {"public_knob", "internal_default", "diagnostic_only", "hard_limit"}
    assert all(v in allowed for v in classification.values())
    # The native (C++) classification and the Python fallback map must be identical (single truth).
    assert dict(classification) == dict(_CONSTANT_CLASSIFICATION)
    assert d["runtime"]["max_runtime_params"] == 32
    assert d["diagnostics"]["fft_direct_dft_fallback_count"] == 0
    assert classification["kMaxRuntimeParams"] == "hard_limit"
    assert classification["kNewtonDefaultFdEps"] == "public_knob"
    # ADC-644/645: the newly wired knobs are public.
    assert classification["kMGDefaultCoarseThreshold"] == "public_knob"
    assert classification["kWenoEpsilon"] == "public_knob"
    assert classification["kCflSpeedFloor"] == "public_knob"


def test_system_inspect_reports_effective_block_and_solver_options():
    sim = System(n=8, L=1.0, periodic=True)
    sim.add_equation(
        "ion",
        _isothermal_model(),
        time=engine.IMEX(
            newton_max_iters=4,
            newton_rel_tol=1e-6,
            newton_fd_eps=2e-7,
            newton_damping=0.8,
            newton_fail_policy="throw",
            newton_diagnostics=True,
        ),
        spatial=engine.Spatial(positivity_floor=1e-12),
    )
    sim.set_poisson(abs_tol=1e-11)

    options = sim.inspect().to_dict()["options"]
    assert options["defaults"]["newton"]["max_iters"] == 2
    assert options["poisson"]["solver"] == "geometric_mg"
    assert options["poisson"]["epsilon"] == pytest.approx(1.0)
    assert options["poisson"]["abs_tol"] == pytest.approx(1e-11)

    block = options["blocks"][0]
    assert block["name"] == "ion"
    assert block["transport"] == "isothermal"
    assert block["time"] == "imex"
    assert block["newton"]["max_iters"] == 4
    assert block["newton"]["rel_tol"] == pytest.approx(1e-6)
    assert block["newton"]["fd_eps"] == pytest.approx(2e-7)
    assert block["newton"]["fail_policy"] == "throw"
    assert block["newton"]["diagnostics"] is True
    assert block["physical"]["cs2"] == pytest.approx(0.7)
    assert block["physical"]["q"] == pytest.approx(-2.0)
    assert block["positivity_floor"] == pytest.approx(1e-12)


def test_invalid_newton_and_refinement_values_are_rejected():
    with pytest.raises(ValueError, match="newton_max_iters"):
        engine.IMEX(newton_max_iters=0)

    amr = AmrSystem(n=8, L=1.0, periodic=True)
    with pytest.raises(RuntimeError, match="threshold must be finite"):
        amr.set_refinement(math.inf)
    with pytest.raises(RuntimeError, match="grad_threshold must be finite"):
        amr.set_phi_refinement(math.nan)
