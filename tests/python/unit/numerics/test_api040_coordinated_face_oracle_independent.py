"""Adversarial pure-math contract for a generic coordinated-face consumer."""

import numpy as np
import pytest

from tests.python.support import api040_coordinated_face_oracle_independent as coupled


def test_asymmetric_signed_sides_and_permutation_are_not_a_shared_flux():
    left = np.array([0.2, -0.1, 0.3])
    right = np.array([0.25, -0.04, 0.34])
    common, left_source, right_source, speed = coupled.face(
        left, right, beta=0.7, gamma=-0.4)
    integral = np.array([0.0, 0.0112, -0.0054])
    np.testing.assert_allclose(left_source, -0.3*integral, rtol=0, atol=2e-17)
    np.testing.assert_allclose(right_source, -0.7*integral, rtol=0, atol=2e-17)
    np.testing.assert_allclose(left_source+right_source, -integral, rtol=0, atol=2e-17)
    assert speed == pytest.approx(1.238)
    assert common.shape == left_source.shape == right_source.shape == (3,)
    assert left_source[0] == right_source[0] == 0.0  # c is strictly conservative.
    assert not np.array_equal(left_source, right_source)
    order = coupled.PERMUTATIONS[1]
    permutation = [coupled.COMPONENTS.index(name) for name in order]
    permuted = coupled.face(left[permutation], right[permutation],
                            beta=0.7, gamma=-0.4, order=order)
    for reference, actual in zip((common, left_source, right_source), permuted[:3]):
        np.testing.assert_allclose(actual, reference[permutation], rtol=0, atol=2e-17)


def test_diagonal_and_closed_loop_show_genuine_nonconservative_product():
    state = np.array([0.2, -0.1, 0.3])
    common, left, right, _ = coupled.face(state, state, beta=0.7, gamma=-0.4)
    np.testing.assert_array_equal(common, coupled.physical_flux(state))
    np.testing.assert_array_equal(left, np.zeros(3))
    np.testing.assert_array_equal(right, np.zeros(3))
    # Around a c-r rectangle, the p-component integral is nonzero. B cannot
    # be replaced by the gradient of a conservative flux or dropped as a source.
    corners = [np.array([0.2, 0.0, 0.1]), np.array([0.4, 0.0, 0.1]),
               np.array([0.4, 0.0, 0.3]), np.array([0.2, 0.0, 0.3])]
    loop = sum((coupled.path_integral(corners[i], corners[(i+1) % 4], 0.7, -0.4)
                for i in range(4)), np.zeros(3))
    np.testing.assert_allclose(loop, [0.0, -0.028, 0.0], rtol=0, atol=2e-17)


@pytest.mark.parametrize("cells", (24, 48))
def test_cell_means_and_periodic_fe_observe_exact_ownership_and_live_coefficients(cells):
    initial = coupled.initial_cell_means(cells)
    nodes, weights = np.polynomial.legendre.leggauss(32)
    x = ((np.arange(cells)[:, None] + 0.5 + 0.5*nodes) / cells)
    integrand = np.array((0.2+0.05*np.cos(2*np.pi*x),
                          -0.1+0.04*np.sin(2*np.pi*x),
                          0.3+0.03*np.cos(4*np.pi*x)))
    quadrature = 0.5*np.sum(integrand*weights, axis=-1)
    np.testing.assert_allclose(initial, quadrature, rtol=0, atol=2e-16)
    positive = coupled.forward_euler(cells, beta=0.7, gamma=-0.4)
    negative = coupled.forward_euler(cells, beta=-0.7, gamma=0.4)
    np.testing.assert_allclose(positive[0].mean(), initial[0].mean(), atol=2e-16, rtol=0)
    np.testing.assert_allclose(negative[0].mean(), initial[0].mean(), atol=2e-16, rtol=0)
    one_positive = coupled.forward_euler(cells, beta=0.7, gamma=-0.4, steps=1)
    one_negative = coupled.forward_euler(cells, beta=-0.7, gamma=0.4, steps=1)
    # Equal absolute parameters give the same first-step speed and c update;
    # later r feedback can change the bound without changing c's total mass.
    np.testing.assert_array_equal(one_positive[0], one_negative[0])
    assert np.max(np.abs(positive[0]-negative[0])) < 1e-6
    assert np.max(np.abs(positive[1]-negative[1])) > 5e-4
    assert np.max(np.abs(positive[2]-negative[2])) > 2e-4
    order = coupled.PERMUTATIONS[1]
    permuted = coupled.forward_euler(cells, beta=0.7, gamma=-0.4, order=order)
    back = [order.index(name) for name in coupled.COMPONENTS]
    np.testing.assert_allclose(permuted[back], positive, rtol=0, atol=3e-17)


def test_active_nonfinite_face_and_parameters_are_refused():
    finite = np.array([0.2, -0.1, 0.3])
    with pytest.raises(ValueError, match="finite"):
        coupled.face(finite, [0.2, float("nan"), 0.3], beta=0.7, gamma=-0.4)
    with pytest.raises(ValueError, match="finite"):
        coupled.face(finite, finite, beta=float("inf"), gamma=-0.4)


def test_composed_jacobian_spectral_radius_alone_misses_asymmetric_split_growth():
    # F(u)=-u, B=+1: the physical composed Jacobian DF+B is exactly zero.
    # An asymmetric source split with theta=.7 nevertheless makes a checkerboard
    # grow if its authored dissipation is set to that zero spectral radius.
    cells = 24
    state = np.tile([1.0, -1.0], cells//2)
    jump = np.roll(state, -1)-state
    common = -0.5*(state+np.roll(state, -1))  # alpha=0
    left_source = -0.7*jump
    right_source = -0.3*jump
    rate = cells*(-common+np.roll(common, 1)+left_source+np.roll(right_source, 1))
    updated = state + (0.1/cells)*rate
    assert np.max(np.abs(updated)) == pytest.approx(1.08)
