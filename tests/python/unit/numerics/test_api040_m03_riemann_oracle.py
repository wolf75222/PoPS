"""Pure mathematical checks of M03's exact oracle; no PoPS runtime involved."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ORACLE = (Path(__file__).resolve().parents[4] / "examples" / "migration"
          / "scientific" / "api040_m03_riemann_oracle.py")
SPEC = importlib.util.spec_from_file_location("api040_m03_riemann_oracle", ORACLE)
oracle = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = oracle
SPEC.loader.exec_module(oracle)


LEFT = oracle.Primitive(1., 0., 1.)
RIGHT = oracle.Primitive(.125, 0., .1)


def test_ideal_sod_star_matches_independent_published_riemann_values():
    gas = oracle.Gas(1.4)
    pressure, velocity = oracle.star_state(LEFT, RIGHT, gas)
    assert pressure == pytest.approx(.303130178050647, rel=2.e-12)
    assert velocity == pytest.approx(.92745262004895, rel=2.e-12)
    assert oracle._wave_function(pressure, LEFT, gas) + oracle._wave_function(
        pressure, RIGHT, gas) == pytest.approx(0., abs=1.e-14)
    speeds = oracle.wave_locations(LEFT, RIGHT, gas)
    assert len(speeds) == 4  # left fan head/tail, contact, right shock
    assert speeds[0] < speeds[1] < speeds[2] < speeds[3]


@pytest.mark.parametrize("gas", [oracle.Gas(1.4, 0.), oracle.Gas(1.4, .3)])
def test_sod_cell_means_and_exact_integral_balances(gas):
    n = 200
    initial = oracle.cell_averages(n, 0., LEFT, RIGHT, gas, quadrature_order=16)
    final = oracle.cell_averages(n, .15, LEFT, RIGHT, gas, quadrature_order=24)
    high_order = oracle.cell_averages(n, .15, LEFT, RIGHT, gas, quadrature_order=48)
    np.testing.assert_allclose(initial[:, :n // 2],
                               np.broadcast_to(oracle.conservative(LEFT, gas)[:, None], (4, n // 2)),
                               rtol=0., atol=1.e-14)
    np.testing.assert_allclose(initial[:, n // 2:],
                               np.broadcast_to(oracle.conservative(RIGHT, gas)[:, None], (4, n // 2)),
                               rtol=0., atol=1.e-14)
    np.testing.assert_allclose(final, high_order, rtol=0., atol=2.e-12)
    # Integral conservation follows the far-field fluxes, independently of the
    # finite-volume solver: left minus right momentum flux is p_L-p_R.
    assert np.mean(final[0]) == pytest.approx(np.mean(initial[0]), abs=2.e-12)
    assert np.mean(final[1]) == pytest.approx(.15 * (LEFT.p - RIGHT.p), abs=2.e-12)
    assert np.mean(final[2]) == pytest.approx(0., abs=2.e-12)
    assert np.mean(final[3]) == pytest.approx(np.mean(initial[3]), abs=2.e-12)
    rho = final[0]
    internal = final[3] - (final[1]**2 + final[2]**2) / (2 * rho)
    assert np.all(rho > 0)
    assert np.all(internal > 0)


def test_stiffened_pressure_shift_and_wave_star():
    gas = oracle.Gas(1.4, .3)
    pressure, velocity = oracle.star_state(LEFT, RIGHT, gas)
    assert 0.1 < pressure < 1.
    assert velocity > 0.
    assert oracle._wave_function(pressure, LEFT, gas) + oracle._wave_function(
        pressure, RIGHT, gas) == pytest.approx(0., abs=1.e-14)
    for state in (LEFT, RIGHT):
        conserved = oracle.conservative(state, gas)
        recovered_p = (gas.gamma - 1) * (conserved[3] - conserved[1]**2 / (2 * conserved[0]))
        recovered_p -= gas.gamma * gas.p_inf
        assert recovered_p == pytest.approx(state.p, abs=1.e-14)
