"""Pre-native independent M07 equilibrium and two-sided face counter-tests."""

import numpy as np
import pytest

from tests.python.support import api040_m07_lake_oracle_independent as lake


@pytest.mark.parametrize("cells", lake.RESOLUTIONS)
def test_bottom_and_water_are_true_cell_means(cells):
    state, bottom = lake.initial_with_equilibrium_ghosts(cells)
    dx = 2.0 / cells
    nodes, weights = np.polynomial.legendre.leggauss(40)
    x_left = -1.0 + dx * np.arange(-1, cells + 1)
    points = x_left[:, None] + 0.5 * dx * (nodes + 1.0)
    independent_quadrature = 0.1 * np.sum(
        weights * np.exp(-50.0 * points * points), axis=1)
    np.testing.assert_allclose(bottom, independent_quadrature, atol=2e-15, rtol=0)
    np.testing.assert_allclose(state[0] + bottom, 1.0, atol=2e-16, rtol=0)
    assert np.min(state[0]) > 0.8
    midpoint = 0.2 * np.exp(-50.0 * (x_left + dx / 2.0)**2)
    assert np.max(np.abs(bottom - midpoint)) > 1e-6


@pytest.mark.parametrize("cells", lake.RESOLUTIONS)
def test_two_sided_hydrostatic_face_preserves_lake_but_shared_flux_moves_it(cells):
    state, bottom = lake.initial_with_equilibrium_ghosts(cells)
    proper_rate = lake.rhs(state, bottom)
    assert np.max(np.abs(proper_rate)) < 2e-14
    shared_rate = lake.rhs(state, bottom, corrected=False)
    one_step_wrong_momentum = 0.1 * (2.0 / cells) * np.max(np.abs(shared_rate[1]))
    assert one_step_wrong_momentum > 1e-3
    face_pairs = [lake.hydrostatic_face(state[:, i], state[:, i + 1],
                                        bottom[i], bottom[i + 1])
                  for i in range(cells + 1)]
    assert max(np.max(np.abs(left - right)) for _, left, right in face_pairs) > 1e-2
    final, max_change = lake.trajectory(cells)
    assert max_change <= lake.EQUILIBRIUM_TOLERANCE
    np.testing.assert_allclose(final[0] + bottom[1:-1], 1.0,
                               atol=lake.EQUILIBRIUM_TOLERANCE, rtol=0)
    assert np.max(np.abs(final[1])) <= lake.EQUILIBRIUM_TOLERANCE


def test_nonzero_velocity_face_retains_common_mass_flux_and_distinct_pressure():
    left = np.array([0.9, 0.09])
    right = np.array([0.8, -0.04])
    common, from_left, from_right = lake.hydrostatic_face(left, right, 0.1, 0.2)
    assert from_left[0] == common[0] == from_right[0]
    expected = 0.5 * (left[0]**2 - right[0]**2)
    assert abs((from_left[1] - from_right[1]) - expected) < 1e-15
