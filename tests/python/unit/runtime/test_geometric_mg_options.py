"""ADC-613 GeometricMG options reach the native V-cycle end to end.

Two tiers:
  * descriptor-only (no ``_pops``): the default reconciliation to the native kMG* constants, the
    resolved mg_options() mapping, and the STRUCTURAL refusal of the un-wired Chebyshev smoother /
    out-of-domain tolerances;
  * runtime (``_pops`` built): a uniform System reports and executes CartesianCG, and explicitly
    refuses GeometricMG instead of mapping that name onto the CG implementation.
"""

import pytest

from pops.solvers.elliptic import GeometricMG
from pops.solvers.options import Chebyshev, RedBlackGaussSeidel
from pops.solvers.tolerances import Absolute, AbsoluteFloor, Relative


# --- descriptor tier (pure Python, no _pops) ---------------------------------

def test_default_descriptor_reconciles_to_native_mg_constants():
    """GeometricMG() defaults ARE the native kMG* constants (bit-identity source of truth)."""
    from pops.runtime.defaults import numerical_defaults_report

    mg = numerical_defaults_report()["mg"]
    opts = GeometricMG().mg_options()
    assert opts["rel_tol"] == pytest.approx(mg["rel_tol"])
    assert opts["abs_tol"] == pytest.approx(0.0)
    assert opts["max_cycles"] == mg["max_cycles"]
    assert opts["min_coarse"] == mg["min_coarse"]
    assert opts["pre_smooth"] == mg["pre_smooth"]
    assert opts["post_smooth"] == mg["post_smooth"]
    assert opts["bottom_sweeps"] == mg["bottom_sweeps"]
    # The default smoother is the natively-wired Gauss-Seidel, not the un-wired Chebyshev.
    assert isinstance(GeometricMG().smoother, RedBlackGaussSeidel)


def test_relative_tolerance_maps_rel_and_floor():
    opts = GeometricMG(tolerance=Relative(1e-4, AbsoluteFloor(1e-11)), max_cycles=5).mg_options()
    assert opts["rel_tol"] == pytest.approx(1e-4)
    assert opts["abs_tol"] == pytest.approx(1e-11)
    assert opts["max_cycles"] == 5


def test_absolute_tolerance_keeps_native_rel_and_sets_floor():
    opts = GeometricMG(tolerance=Absolute(1e-9)).mg_options()
    assert opts["rel_tol"] > 0.0  # native mixed criterion requires rel_tol > 0
    assert opts["abs_tol"] == pytest.approx(1e-9)


def test_sweep_knobs_pass_through():
    opts = GeometricMG(min_coarse=4, pre_sweeps=3, post_sweeps=1, bottom_sweeps=80).mg_options()
    assert opts["min_coarse"] == 4
    assert opts["pre_smooth"] == 3
    assert opts["post_smooth"] == 1
    assert opts["bottom_sweeps"] == 80


def test_coarse_threshold_default_disabled_and_override(caplog=None):
    """ADC-644: DirectSmallGrid threshold reaches mg_options as coarse_threshold (0 = disabled)."""
    from pops.solvers.options import DirectSmallGrid

    # Default coarse solver -> None -> disabled sentinel 0 (bit-identical hierarchy).
    assert GeometricMG().mg_options()["coarse_threshold"] == 0
    # An explicit threshold reaches the resolved options.
    opts = GeometricMG(coarse=DirectSmallGrid(64)).mg_options()
    assert opts["coarse_threshold"] == 64


def test_chebyshev_smoother_refuses_structurally():
    report = GeometricMG(smoother=Chebyshev()).validate()
    assert not report.ok
    codes = {i.code for i in report.issues}
    assert "elliptic_solver.smoother_not_wired" in codes
    # lower() must also refuse (never a silent drop of the un-wired smoother).
    with pytest.raises(ValueError, match="Gauss-Seidel"):
        GeometricMG(smoother=Chebyshev()).lower()


def test_out_of_domain_cycles_and_tolerance_refuse():
    with pytest.raises(ValueError):
        GeometricMG(max_cycles=0)
    with pytest.raises(ValueError):
        GeometricMG(min_coarse=0)
    with pytest.raises(ValueError, match="Relative"):
        Relative(0.0)


# --- runtime-family separation tier (needs _pops) ----------------------------

pops = pytest.importorskip("pops")
import pops.runtime._engine_descriptors as engine  # noqa: E402
from pops.runtime._system import System  # noqa: E402  (ADC-545 advanced runtime seam)


def _sim(**poisson):
    sim = System(n=16, L=1.0, periodicity=(True, True))
    sim.add_equation(
        "ion",
        engine.Model(engine.FluidState.isothermal(cs2=0.7), engine.IsothermalFlux(), engine.NoSource(),
                   engine.ChargeDensity(charge=-1.0)),
        spatial=engine.Spatial(),
    )
    if poisson:
        sim.set_poisson(**poisson)
    return sim


def test_uniform_system_reports_cartesian_cg_defaults():
    """A uniform System must never advertise its CG kernel as geometric multigrid."""
    sim = _sim()
    report = sim.inspect().to_dict()["options"]
    poisson = report["poisson"]
    cg = report["defaults"]["cartesian_cg"]
    assert poisson["solver"] == "cartesian_cg"
    assert poisson["solver_option_schema"] == "pops.system.cartesian-cg-options@1"
    assert poisson["rel_tol"] == pytest.approx(cg["rel_tol"])
    assert poisson["abs_tol"] == pytest.approx(cg["abs_tol"])
    assert poisson["max_iterations"] == cg["max_iterations"]


def test_cartesian_cg_override_visible_in_effective_report():
    sim = _sim(rel_tol=1e-4, max_iterations=7)
    poisson = sim.inspect().to_dict()["options"]["poisson"]
    assert poisson["rel_tol"] == pytest.approx(1e-4)
    assert poisson["max_iterations"] == 7


def test_uniform_system_refuses_geometric_mg_instead_of_aliasing_it():
    with pytest.raises((RuntimeError, ValueError), match="GeometricMG.*AMR"):
        _sim(solver="geometric_mg")


def test_native_set_poisson_refuses_out_of_domain():
    with pytest.raises((RuntimeError, ValueError)):
        _sim(rel_tol=0.0)
    with pytest.raises((RuntimeError, ValueError)):
        _sim(max_iterations=0)


def main():
    test_default_descriptor_reconciles_to_native_mg_constants()
    test_relative_tolerance_maps_rel_and_floor()
    test_absolute_tolerance_keeps_native_rel_and_sets_floor()
    test_sweep_knobs_pass_through()
    test_coarse_threshold_default_disabled_and_override()
    test_chebyshev_smoother_refuses_structurally()
    test_out_of_domain_cycles_and_tolerance_refuse()
    test_uniform_system_reports_cartesian_cg_defaults()
    test_cartesian_cg_override_visible_in_effective_report()
    test_uniform_system_refuses_geometric_mg_instead_of_aliasing_it()
    test_native_set_poisson_refuses_out_of_domain()
    print("OK  ADC-613 GeometricMG options")


if __name__ == "__main__":
    main()
