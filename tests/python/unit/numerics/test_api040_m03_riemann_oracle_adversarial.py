"""Independent conservation, symmetry and averaging checks of the M03 oracle.

These checks do not execute PoPS or certify the native finite-volume solver.
"""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ORACLE_PATH = (Path(__file__).resolve().parents[4] / "examples" / "migration"
               / "scientific" / "api040_m03_riemann_oracle.py")
SPEC = importlib.util.spec_from_file_location("m03_adversarial_oracle", ORACLE_PATH)
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oracle
SPEC.loader.exec_module(oracle)

LEFT = oracle.Primitive(1., 0., 1.)
RIGHT = oracle.Primitive(.125, 0., .1)
GASES = (oracle.Gas(1.4), oracle.Gas(1.4, .3), oracle.Gas(1.6, 2.))


def physical_state_and_flux(state, gas):
    # Explicitly distinguish rho*e, specific e, and total density E. Neither
    # the oracle's conservative() nor its shock-density helper is used here.
    rho, velocity, pressure = state.rho, state.u, state.p
    specific_internal = (pressure + gas.gamma * gas.p_inf) / (
        (gas.gamma - 1) * rho)
    energy = rho * specific_internal + .5 * rho * velocity**2
    conserved = np.array((rho, rho * velocity, 0., energy))
    flux = np.array((rho * velocity, rho * velocity**2 + pressure,
                     0., (energy + pressure) * velocity))
    return conserved, flux


@pytest.mark.parametrize("gas", GASES)
@pytest.mark.parametrize("mirror", [False, True])
def test_shock_rankine_hugoniot_including_stiffened_total_energy(gas, mirror):
    left, right = (RIGHT, LEFT) if mirror else (LEFT, RIGHT)
    speeds = oracle.wave_locations(left, right, gas)
    shock_speed = speeds[0] if mirror else speeds[-1]
    before = oracle.sample(shock_speed - 1.e-7, 1., left, right, gas)
    after = oracle.sample(shock_speed + 1.e-7, 1., left, right, gas)
    assert before.rho != pytest.approx(after.rho)  # really straddles a shock
    u_before, f_before = physical_state_and_flux(before, gas)
    u_after, f_after = physical_state_and_flux(after, gas)
    np.testing.assert_allclose(shock_speed * (u_after - u_before),
                               f_after - f_before, rtol=2.e-12, atol=2.e-12)


@pytest.mark.parametrize("gas", GASES)
@pytest.mark.parametrize("mirror", [False, True])
def test_both_fans_preserve_entropy_and_the_correct_riemann_invariant(gas, mirror):
    left, right = (RIGHT, LEFT) if mirror else (LEFT, RIGHT)
    speeds = oracle.wave_locations(left, right, gas)
    low, high = speeds[-2:] if mirror else speeds[:2]
    side = 1 if mirror else -1
    upstream = right if mirror else left
    sound = lambda q: np.sqrt(gas.gamma * (q.p + gas.p_inf) / q.rho)
    for fraction in (.13, .51, .89):
        xi = low + fraction * (high - low)
        state = oracle.sample(xi, 1., left, right, gas)
        assert state.rho != pytest.approx(upstream.rho)
        assert (state.p + gas.p_inf) / state.rho**gas.gamma == pytest.approx(
            (upstream.p + gas.p_inf) / upstream.rho**gas.gamma, rel=2.e-13)
        assert state.u - side * 2 * sound(state) / (gas.gamma - 1) == pytest.approx(
            upstream.u - side * 2 * sound(upstream) / (gas.gamma - 1), rel=2.e-13)
        assert state.u + side * sound(state) == pytest.approx(xi, abs=2.e-13)


@pytest.mark.parametrize("gas", GASES[1:])
def test_pressure_shift_maps_to_ideal_solution_but_total_energy_has_an_offset(gas):
    ideal = oracle.Gas(gas.gamma)
    shifted = lambda q: oracle.Primitive(q.rho, q.u, q.p + gas.p_inf)
    ideal_left, ideal_right = shifted(LEFT), shifted(RIGHT)
    speeds = oracle.wave_locations(LEFT, RIGHT, gas)
    # Midpoints cover each constant region and the nontrivial rarefaction fan.
    bounds = [speeds[0] - 1, *speeds, speeds[-1] + 1]
    for xi in [(a + b) / 2 for a, b in zip(bounds, bounds[1:])]:
        stiff = oracle.sample(xi, 1., LEFT, RIGHT, gas)
        ordinary = oracle.sample(xi, 1., ideal_left, ideal_right, ideal)
        np.testing.assert_allclose((stiff.rho, stiff.u, stiff.p + gas.p_inf),
                                   (ordinary.rho, ordinary.u, ordinary.p),
                                   rtol=2.e-13, atol=2.e-13)
        expected = oracle.conservative(ordinary, ideal)
        expected[3] += gas.p_inf
        np.testing.assert_allclose(oracle.conservative(stiff, gas), expected,
                                   rtol=2.e-13, atol=2.e-13)


@pytest.mark.parametrize("gas", GASES)
def test_galilean_shift_includes_kinetic_energy_and_boundary_energy_flux(gas):
    boost = .37
    shifted = lambda q: oracle.Primitive(q.rho, q.u + boost, q.p)
    left, right = shifted(LEFT), shifted(RIGHT)
    speeds = oracle.wave_locations(LEFT, RIGHT, gas)
    bounds = [speeds[0] - 1, *speeds, speeds[-1] + 1]
    for xi in [(a + b) / 2 for a, b in zip(bounds, bounds[1:])]:
        reference = oracle.sample(xi, 1., LEFT, RIGHT, gas)
        moving = oracle.sample(xi + boost, 1., left, right, gas)
        np.testing.assert_allclose((moving.rho, moving.u, moving.p),
                                   (reference.rho, reference.u + boost, reference.p),
                                   rtol=2.e-13, atol=2.e-13)
        base_u, _ = physical_state_and_flux(reference, gas)
        expected_energy = base_u[3] + boost * base_u[1] + .5 * boost**2 * base_u[0]
        assert oracle.conservative(moving, gas)[3] == pytest.approx(expected_energy)
    # Moving far fields have nonzero mass AND energy flux. Constant total mass
    # and energy would be the wrong oracle for this problem.
    time = .04
    assert time * max(abs(s) for s in oracle.wave_locations(left, right, gas)) < .5
    initial = oracle.cell_averages(17, 0., left, right, gas)
    final = oracle.cell_averages(17, time, left, right, gas, quadrature_order=32)
    refined = oracle.cell_averages(17, time, left, right, gas, quadrature_order=64)
    np.testing.assert_allclose(final, refined, rtol=0., atol=2.e-12)
    _, flux_left = physical_state_and_flux(left, gas)
    _, flux_right = physical_state_and_flux(right, gas)
    np.testing.assert_allclose(np.mean(final - initial, axis=1),
                               time * (flux_left - flux_right), rtol=0., atol=2.e-12)


@pytest.mark.parametrize("n", [1, 5, 17])
def test_initial_cell_cut_by_diaphragm_is_a_conservative_average(n):
    gas = oracle.Gas(1.6, 2.)
    left, right = oracle.Primitive(.7, -.6, 2.), oracle.Primitive(1.3, .9, .4)
    result = oracle.cell_averages(n, 0., left, right, gas)
    ul, _ = physical_state_and_flux(left, gas)
    ur, _ = physical_state_and_flux(right, gas)
    np.testing.assert_allclose(result[:, n // 2], .5 * (ul + ur), rtol=0., atol=2.e-13)
    assert not np.allclose(result[:, n // 2], oracle.conservative(right, gas))


def test_m03_far_field_balance_requires_wave_containment_in_the_domain():
    for gas in GASES[:2]:
        assert .15 * max(abs(s) for s in oracle.wave_locations(LEFT, RIGHT, gas)) < .5
    # The stronger offset is a valid EOS but not the same finite-domain
    # experiment: its outgoing shock has crossed x=.5 before the chosen end.
    assert .15 * max(oracle.wave_locations(LEFT, RIGHT, GASES[2])) > .5
