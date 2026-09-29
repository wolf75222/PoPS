"""Independent full-grid stencil checks for the boundary-only budget oracle."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_path = Path(__file__).resolve().parents[4] / "examples/migration/scientific/api040_m03_boundary_oracle.py"
_spec = importlib.util.spec_from_file_location("m03_boundary_oracle", _path)
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)


def full_residual(state, dx, dy, gamma, p_inf):
    # Independent assembly over every cell, including interior cancellation.
    # This deliberately does not call the boundary-stage implementation.
    _, ny, nx = state.shape
    rate = np.zeros_like(state)
    for j in range(ny):
        for i in range(nx):
            center = state[:, j, i]
            left = state[:, j, max(i - 1, 0)]
            right = state[:, j, min(i + 1, nx - 1)]
            bottom = state[:, (j - 1) % ny, i]
            top = state[:, (j + 1) % ny, i]
            rate[:, j, i] = (
                (oracle.rusanov(left, center, 0, gamma, p_inf)
                 - oracle.rusanov(center, right, 0, gamma, p_inf)) / dx
                + (oracle.rusanov(bottom, center, 1, gamma, p_inf)
                   - oracle.rusanov(center, top, 1, gamma, p_inf)) / dy)
    return rate


@pytest.mark.parametrize("p_inf", (0., .3))
def test_boundary_quadrature_matches_full_2d_ssprk2_with_transverse_variation(p_inf):
    ny, nx, gamma = 7, 8, 1.4
    dx, dy, dt = 1 / nx, 1 / ny, .003
    x, y = np.meshgrid((np.arange(nx) + .5) / nx, (np.arange(ny) + .5) / ny)
    density = 1. + .15 * np.cos(2 * np.pi * y) + .07 * x
    vx = .4 + .13 * x * np.sin(2 * np.pi * y)
    vy = .3 * np.cos(2 * np.pi * y) + .1 * x
    pressure = 1. + .2 * x * np.cos(2 * np.pi * y)
    energy = (pressure + gamma * p_inf) / (gamma - 1) + .5 * density * (vx * vx + vy * vy)
    state = np.stack((density, density * vx, density * vy, energy))
    first = state + dt * full_residual(state, dx, dy, gamma, p_inf)
    final = .5 * state + .5 * (first + dt * full_residual(first, dx, dy, gamma, p_inf))
    expected = dx * dy * np.sum(final - state, axis=(1, 2))
    observed = oracle.ssprk2_boundary_increment(oracle.boundary_bands(state), dt, dx, dy, gamma, p_inf)
    np.testing.assert_allclose(observed, expected, rtol=0., atol=2.e-15)
    # An unchanged far-field flux is measurably wrong even on this one step.
    flux, _ = oracle.physical_flux(state[:, :, [0, -1]], 0, gamma, p_inf)
    frozen = dt * dy * np.sum(flux[:, :, 0] - flux[:, :, 1], axis=1)
    assert np.max(np.abs(observed - frozen)) > 1.e-7


def test_constant_state_has_no_net_boundary_exchange():
    state = np.broadcast_to(np.array([1., .2, -.1, 3.])[:, None, None], (4, 6, 8))
    np.testing.assert_array_equal(
        oracle.ssprk2_boundary_increment(oracle.boundary_bands(state), .001, 1 / 8, 1 / 6, 1.4, .3),
        np.zeros(4))


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), -1., 0.))
def test_bad_stage_density_is_rejected_without_clipping(bad):
    state = np.broadcast_to(np.array([1., .2, -.1, 3.])[:, None, None], (4, 6, 8)).copy()
    state[0, 0, 0] = bad
    with pytest.raises(ValueError, match="density"):
        oracle.ssprk2_boundary_increment(oracle.boundary_bands(state), .001, 1 / 8, 1 / 6, 1.4, .3)
