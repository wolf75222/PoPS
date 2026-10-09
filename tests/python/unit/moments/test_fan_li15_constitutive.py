"""Rational and spectral witnesses for full-temperature Fan--Li15 algebra."""

from fractions import Fraction as Q
from math import sqrt

import numpy as np
import pytest

from pops._ir.expr import Expr, Var
from pops.moments import (
    FAN_LI15_INDICES,
    FAN_LI15_REGULARIZED_COMPONENTS,
    fan_li15_expressions,
    fan_li15_from_hermite,
)
from tests.python.support.fan_li15_oracle import (
    INDICES,
    TOP,
    derivative_gaussian_flux,
    gaussian_mixture,
    generating_hermite,
    primary_product,
)


def mixture():
    return gaussian_mixture(
        (
            (Q(2, 3), (Q(1, 3), Q(-1, 4)), (Q(5, 4), Q(1, 8), Q(3, 4))),
            (Q(1, 3), (Q(-1, 2), Q(1, 3)), (Q(3, 4), Q(-1, 9), Q(7, 8))),
        )
    )


def test_rational_transform_and_fifth_flux_match_independent_generating_function():
    raw = mixture()
    expr = fan_li15_expressions(raw)
    h, velocity, temperature = generating_hermite(raw)
    assert FAN_LI15_INDICES == INDICES
    assert FAN_LI15_REGULARIZED_COMPONENTS == TOP == (4, 8, 11, 13, 14)
    assert expr.velocity == velocity
    assert expr.temperature == temperature
    assert all(expr.hermite[alpha] == h.get(alpha, 0) for alpha in INDICES)
    assert any(h[alpha] for alpha in INDICES if sum(alpha) == 3)
    assert any(h[alpha] for alpha in INDICES if sum(alpha) == 4)
    assert (
        fan_li15_from_hermite(
            raw[0], velocity, temperature, {a: h[a] for a in INDICES if sum(a) >= 3}
        )
        == raw
    )
    direction = Q(3, 5), Q(-4, 5)
    assert expr.directional_flux(direction) == derivative_gaussian_flux(raw, direction)


def test_primary_multidimensional_regularization_sign_factorials_and_matrix():
    raw, direction = mixture(), (Q(2, 3), Q(-5, 7))
    delta = tuple(Q(k - 6, 17) for k in range(15))
    expr = fan_li15_expressions(raw)
    expected = primary_product(raw, delta, direction)
    assert expr.directional_nonconservative_action(direction, delta) == expected
    matrix = expr.directional_nonconservative_matrix(direction)
    assert tuple(sum(row[k] * delta[k] for k in range(15)) for row in matrix) == expected
    for row in range(15):
        if row not in TOP:
            assert all(type(value) is int and value == 0 for value in matrix[row])
    assert expr.directional_nonconservative_action(direction, raw) == (0,) * 15
    scaled = fan_li15_expressions(tuple(7 * value for value in raw))
    assert scaled.directional_nonconservative_matrix(direction) == matrix


def test_symbolic_flux_product_and_bounds_evaluate_as_numeric_algebra():
    raw = tuple(map(float, mixture()))
    symbolic = fan_li15_expressions(tuple(Var(f"m{k}", "cons") for k in range(15)))
    numeric = fan_li15_expressions(raw)
    env = {f"m{k}": value for k, value in enumerate(raw)}
    direction, delta = (0.6, -0.8), tuple((k - 6) / 17 for k in range(15))
    expressions = (
        symbolic.directional_flux(direction)
        + symbolic.directional_nonconservative_action(direction, delta)
        + (
            symbolic.directional_spectral_radius(direction),
            symbolic.fixed_direction_path_speed_majorant(direction),
        )
    )
    expected = (
        numeric.directional_flux(direction)
        + numeric.directional_nonconservative_action(direction, delta)
        + (
            numeric.directional_spectral_radius(direction),
            numeric.fixed_direction_path_speed_majorant(direction),
        )
    )
    actual = [value.eval(env) if isinstance(value, Expr) else value for value in expressions]
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("direction", [(1.0, 0.0), (0.0, 1.0), (0.6, -0.8), (2.0, -3.0)])
def test_complete_jacobian_has_primary_hermite_spectrum(direction):
    raw = np.asarray(tuple(map(float, mixture())))
    expr = fan_li15_expressions(raw)
    columns = []
    for column in range(15):
        perturbed = raw.astype(complex)
        perturbed[column] += 1e-25j
        columns.append(np.imag(fan_li15_expressions(perturbed).directional_flux(direction)) / 1e-25)
    complete = np.column_stack(columns) + np.asarray(
        expr.directional_nonconservative_matrix(direction)
    )
    actual = np.linalg.eigvals(complete)
    u, v = expr.velocity
    a, b, c = expr.temperature
    gx, gy = direction
    sigma = sqrt(gx * gx * a + 2 * gx * gy * b + gy * gy * c)
    roots = [
        root
        for degree in range(1, 6)
        for root in np.polynomial.hermite_e.hermeroots([0] * degree + [1])
    ]
    expected = np.sort(gx * u + gy * v + sigma * np.asarray(roots))
    assert np.max(abs(actual.imag)) < 2e-10
    np.testing.assert_allclose(np.sort(actual.real), expected, rtol=2e-10, atol=2e-10)
    assert expr.directional_spectral_radius(direction) == pytest.approx(
        max(abs(expected)), rel=3e-14
    )


def test_fixed_covector_majorant_bounds_interior_variance_mixture():
    left = gaussian_mixture(((Q(1), (Q(-3), Q(0)), (Q(1, 4), Q(0), Q(1))),))
    right = gaussian_mixture(((Q(7), (Q(3), Q(0)), (Q(1, 4), Q(0), Q(1))),))
    direction = 1.0, 0.0
    bound = max(
        fan_li15_expressions(endpoint).fixed_direction_path_speed_majorant(direction)
        for endpoint in (left, right)
    )
    endpoint_radius = max(
        fan_li15_expressions(endpoint).directional_spectral_radius(direction)
        for endpoint in (left, right)
    )
    observed = []
    for s in np.linspace(0, 1, 129):
        state = tuple((1 - s) * a + s * b for a, b in zip(left, right, strict=True))
        observed.append(fan_li15_expressions(state).directional_spectral_radius(direction))
    assert max(observed) > endpoint_radius
    assert max(observed) <= bound


def test_spd_hyperbolicity_domain_does_not_claim_full_moment_realizability():
    raw = fan_li15_from_hermite(Q(1), (Q(0), Q(0)), (Q(1), Q(0), Q(1)), {(4, 0): Q(-1, 6)})
    assert raw[4] == -1  # Impossible fourth moment, although rho>0 and Theta=I.
    assert fan_li15_expressions(raw).directional_spectral_radius((1, 0)) == sqrt(5 + sqrt(10))


def test_constitutive_input_shape_and_hermite_degree_contracts():
    with pytest.raises(ValueError, match="fifteen"):
        fan_li15_expressions([1] * 14)
    with pytest.raises(ValueError, match="degree-three/four"):
        fan_li15_from_hermite(1, (0, 0), (1, 0, 1), {(5, 0): 1})
    with pytest.raises(ValueError, match="theta_xx"):
        fan_li15_from_hermite(1, (0, 0), (1, 1), {})
