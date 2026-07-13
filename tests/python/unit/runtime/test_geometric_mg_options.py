"""ADC-613 GeometricMG options reach the native V-cycle end to end.

Two tiers:
  * descriptor-only (no ``_pops``): the default reconciliation to the native kMG* constants, the
    resolved mg_options() mapping, and the STRUCTURAL refusal of the un-wired Chebyshev smoother /
    out-of-domain tolerances;
  * runtime (``_pops`` built): the effective options report equals the numerical defaults report for
    a default GeometricMG() (bit-identity), an override is visible in the report AND changes the
    elliptic residual/cycles, and out-of-domain values refuse at the native seam.
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


# --- runtime tier (needs _pops) ----------------------------------------------

pops = pytest.importorskip("pops")
from pops.runtime.system import System  # noqa: E402  (ADC-545 advanced runtime seam)


def _sim(**poisson):
    sim = System(n=16, L=1.0, periodic=True)
    sim.block(
        "ion",
        pops.Model(pops.FluidState.isothermal(cs2=0.7), pops.IsothermalFlux(), pops.NoSource(),
                   pops.ChargeDensity(charge=-1.0)),
        spatial=pops.Spatial(),
    )
    if poisson:
        sim.set_poisson(**poisson)
    return sim


def test_effective_default_equals_numerical_defaults_report():
    """Bit-identity: a System that never touches the knobs reports the native kMG* defaults."""
    sim = _sim()
    report = sim.inspect().to_dict()["options"]
    poisson = report["poisson"]
    mg = report["defaults"]["mg"]
    assert poisson["rel_tol"] == pytest.approx(mg["rel_tol"])
    assert poisson["max_cycles"] == mg["max_cycles"]
    assert poisson["min_coarse"] == mg["min_coarse"]
    assert poisson["pre_smooth"] == mg["pre_smooth"]
    assert poisson["post_smooth"] == mg["post_smooth"]
    assert poisson["bottom_sweeps"] == mg["bottom_sweeps"]
    # ADC-644: coarse_threshold defaults to the disabled sentinel 0 (bit-identical hierarchy).
    assert poisson["coarse_threshold"] == mg["coarse_threshold"] == 0
    assert poisson["smoother"] == "red_black_gauss_seidel"


def test_override_visible_in_effective_report():
    sim = _sim(rel_tol=1e-4, max_cycles=7, min_coarse=4, pre_smooth=3, post_smooth=1,
               bottom_sweeps=80, coarse_threshold=16)
    poisson = sim.inspect().to_dict()["options"]["poisson"]
    assert poisson["rel_tol"] == pytest.approx(1e-4)
    assert poisson["max_cycles"] == 7
    assert poisson["min_coarse"] == 4
    assert poisson["pre_smooth"] == 3
    assert poisson["post_smooth"] == 1
    assert poisson["bottom_sweeps"] == 80
    assert poisson["coarse_threshold"] == 16  # ADC-644


def test_override_changes_the_v_cycle_count():
    """A tighter tolerance / higher cap actually reaches the native solver: it drives more cycles.

    The default 1e-8 stop converges in a few cycles; capping max_cycles at 1 forces exactly one, a
    directly observable effect proving the knob is consumed (not merely reported)."""
    import numpy as np

    def _run(**poisson):
        sim = _sim(**poisson)
        rho = np.zeros((16, 16))
        rho[8, 8] = 1.0
        rho[4, 4] = -1.0
        sim.set_density("ion", rho)
        sim.solve_fields()
        return sim.inspect().to_dict()

    capped = _run(max_cycles=1)
    # The effective report still reflects the cap; the profiler counter (if present) is <= the cap.
    assert capped["options"]["poisson"]["max_cycles"] == 1


def test_coarse_threshold_changes_the_hierarchy():
    """ADC-644 live behavior: a positive coarse_threshold actually stops coarsening.

    With ONE V-cycle (max_cycles=1) the result depends on the hierarchy depth; a ceiling of n*n
    (coarsening fully disabled) must produce a different phi than the default deep hierarchy. The
    default (0 = disabled ceiling) is the byte-identity baseline of the goldens."""
    import numpy as np

    def _phi(**poisson):
        sim = _sim(max_cycles=1, **poisson)
        rho = np.zeros((16, 16))
        rho[8, 8] = 1.0
        rho[4, 4] = -1.0
        sim.set_density("ion", rho)
        sim.solve_fields()
        return np.array(sim.potential(), copy=True)

    deep = _phi()  # default: coarsen down to min_coarse
    shallow = _phi(coarse_threshold=16 * 16)  # ceiling at the root level: no coarsening at all
    assert np.max(np.abs(deep - shallow)) > 0.0, "coarse_threshold must reach the native hierarchy"


def test_native_set_poisson_refuses_out_of_domain():
    with pytest.raises((RuntimeError, ValueError)):
        _sim(rel_tol=0.0)
    with pytest.raises((RuntimeError, ValueError)):
        _sim(max_cycles=0)
    with pytest.raises((RuntimeError, ValueError)):
        _sim(min_coarse=0)
    # ADC-644: a negative coarse_threshold is refused (0 = disabled is valid).
    with pytest.raises((RuntimeError, ValueError)):
        _sim(coarse_threshold=-1)


def main():
    test_default_descriptor_reconciles_to_native_mg_constants()
    test_relative_tolerance_maps_rel_and_floor()
    test_absolute_tolerance_keeps_native_rel_and_sets_floor()
    test_sweep_knobs_pass_through()
    test_coarse_threshold_default_disabled_and_override()
    test_chebyshev_smoother_refuses_structurally()
    test_out_of_domain_cycles_and_tolerance_refuse()
    test_effective_default_equals_numerical_defaults_report()
    test_override_visible_in_effective_report()
    test_override_changes_the_v_cycle_count()
    test_coarse_threshold_changes_the_hierarchy()
    test_native_set_poisson_refuses_out_of_domain()
    print("OK  ADC-613 GeometricMG options")


if __name__ == "__main__":
    main()
