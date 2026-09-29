"""Normative Hall sign, FV representation and physical/numerical dissipation."""
import numpy as np
import pytest

from tests.python.support.api040_hall_independent_oracle import (
    ETA_HALL, FINAL_TIME, INITIAL_VECTOR, WAVENUMBER, apply_amplitude,
    cell_averages, exact_amplitude, laplacian_symbol, observed_amplitude,
    periodic_rhs, ssprk2_amplitude, theta_amplitude,
)


@pytest.mark.parametrize("cells", (32, 64, 128))
def test_exact_fv_average_and_stencil_symbol(cells):
    h = 2*np.pi/cells
    centers = (np.arange(cells)+.5)*h
    nodes, weights = np.polynomial.legendre.leggauss(8)
    quadrature = np.cos(WAVENUMBER*(centers[:, None]+h*nodes[None, :]/2)) @ weights/2
    expected = INITIAL_VECTOR[:, None]*quadrature
    initial = cell_averages(cells)
    np.testing.assert_allclose(initial, expected, atol=9e-15, rtol=0.)
    point_samples = INITIAL_VECTOR[:, None]*np.cos(WAVENUMBER*centers)
    assert np.max(np.abs(point_samples-initial)) > 3e-4
    expected_rhs = apply_amplitude(initial, -1j*ETA_HALL*laplacian_symbol(cells))
    np.testing.assert_allclose(periodic_rhs(initial), expected_rhs, atol=8e-12, rtol=0.)


def test_phase_sign_rejects_legacy_reverse_rotation_despite_identical_norm():
    initial = cell_averages(64)
    correct = apply_amplitude(initial, exact_amplitude())
    legacy_wrong_sign = apply_amplitude(initial, exact_amplitude().conjugate())
    np.testing.assert_allclose(np.linalg.norm(correct), np.linalg.norm(initial), atol=2e-15)
    np.testing.assert_allclose(np.linalg.norm(legacy_wrong_sign), np.linalg.norm(initial), atol=2e-15)
    assert np.angle(observed_amplitude(initial, correct)) == pytest.approx(-.24, abs=1e-15)
    assert np.angle(observed_amplitude(initial, legacy_wrong_sign)) == pytest.approx(.24, abs=1e-15)
    assert np.max(np.abs(correct-legacy_wrong_sign)) > .47


def test_hall_zero_and_ohmic_dissipation_are_separate():
    initial = cell_averages(32)
    np.testing.assert_array_equal(apply_amplitude(initial, exact_amplitude(eta_hall=0.)), initial)
    resistive = exact_amplitude(eta_ohmic=.07)
    assert abs(resistive) == pytest.approx(np.exp(-.07*4*.2), abs=2e-16)
    assert np.angle(resistive) == pytest.approx(-.24, abs=1e-15)
    assert abs(exact_amplitude()) == pytest.approx(1., abs=2e-16)
    with pytest.raises(ValueError, match="nonnegative"):
        exact_amplitude(eta_ohmic=-.01)


def test_centered_spatial_error_changes_phase_not_norm_and_converges_second_order():
    phases = np.array([-ETA_HALL*laplacian_symbol(n)*FINAL_TIME for n in (32, 64, 128)])
    errors = np.abs(phases+.24)
    assert np.all(np.log2(errors[:-1]/errors[1:]) > 1.99)
    for cells in (32, 64, 128):
        assert abs(exact_amplitude(symbol=laplacian_symbol(cells))) == pytest.approx(1., abs=2e-16)


def test_theta_time_discretization_has_its_own_amplitude_and_phase():
    cells, steps = 64, 10
    omega_dt = ETA_HALL*laplacian_symbol(cells)*FINAL_TIME/steps
    midpoint = theta_amplitude(cells, steps, theta=.5)
    assert abs(midpoint) == pytest.approx(1., abs=3e-15)
    assert np.angle(midpoint) == pytest.approx(-2*steps*np.arctan(omega_dt/2), abs=1e-15)
    forward = theta_amplitude(cells, steps, theta=0.)
    backward = theta_amplitude(cells, steps, theta=1.)
    assert abs(forward)**2 == pytest.approx((1+omega_dt**2)**steps, abs=3e-15)
    assert abs(backward)**2 == pytest.approx((1+omega_dt**2)**(-steps), abs=3e-15)
    assert abs(forward) > 1. and abs(backward) < 1.


def test_explicit_rk2_amplifies_but_rk4_has_a_bounded_imaginary_axis_interval():
    # Stability in the imaginary direction cannot be certified by an SPD or
    # hyperbolic speed test. Check the actual time-method polynomials instead.
    y = .2
    z = -1j*y
    rk2 = 1+z+z*z/2
    assert abs(rk2)**2 == pytest.approx(1+y**4/4, abs=5e-16)
    for y in (.2, 2.8):
        z = -1j*y
        assert abs(1+z+z*z/2+z**3/6+z**4/24) < 1.
    z = -3j
    assert abs(1+z+z*z/2+z**3/6+z**4/24) > 1.


def test_ssprk2_step_oracle_matches_direct_real_stencil_composition():
    state = cell_averages(64)
    initial = state.copy()
    steps = 20
    dt = FINAL_TIME/steps
    for _ in range(steps):
        predictor = state+dt*periodic_rhs(state)
        state = .5*state+.5*(predictor+dt*periodic_rhs(predictor))
    expected = apply_amplitude(initial, ssprk2_amplitude(64, steps))
    np.testing.assert_allclose(state, expected, atol=3e-14, rtol=0.)


def test_parabolic_step_scaling_is_not_uniform_ssprk2_hall_stability():
    log_gains = []
    for cells in (32, 64, 128):
        h = 2*np.pi/cells
        dt = .1*h*h
        y = ETA_HALL*4/h**2*dt  # highest grid frequency, not the smooth k=2 mode
        log_gains.append(.5*(FINAL_TIME/dt)*np.log1p(y**4/4))
    np.testing.assert_allclose(np.asarray(log_gains[1:])/log_gains[:-1], 4., atol=2e-15)
    assert log_gains[0] > 0.
