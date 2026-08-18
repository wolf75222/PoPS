"""Analytic cell-average quadrature (plan §7.3) for finite-volume oracles."""
from __future__ import annotations

import numpy as np
import pytest

from verification.pops_verify.cell_averages import analytic_cell_averages
from verification.pops_verify.reference_errors import reference_errors


def test_constant_field_average_equals_that_constant_in_1d_2d_3d():
    def constant(*coords):
        return 4.0

    assert analytic_cell_averages(constant, 0.0, 1.0) == pytest.approx(4.0)
    assert analytic_cell_averages(constant, [[0.0, 0.0]], [[2.0, 3.0]]) == pytest.approx(4.0)
    assert analytic_cell_averages(
        constant, [[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]]
    ) == pytest.approx(4.0)


def test_linear_field_average_equals_midpoint():
    lo = np.array([0.25, -1.0])
    hi = np.array([1.25, 3.0])
    averages = analytic_cell_averages(lambda x: 2.0 * x + 1.0, lo, hi)
    midpoints = 0.5 * (lo + hi)
    np.testing.assert_allclose(averages, 2.0 * midpoints + 1.0)


def test_quadratic_1d_average_is_one_third_and_differs_from_cell_center():
    average = analytic_cell_averages(lambda x: x**2, 0.0, 1.0)
    center_sample = 0.5**2
    assert average == pytest.approx(1.0 / 3.0)
    assert average != pytest.approx(center_sample)
    assert center_sample == pytest.approx(0.25)


def test_two_1d_cells_of_x_squared():
    lo = np.array([0.0, 1.0])
    hi = np.array([1.0, 2.0])
    averages = analytic_cell_averages(lambda x: x**2, lo, hi)
    np.testing.assert_allclose(averages, np.array([1.0 / 3.0, 7.0 / 3.0]))


def test_2d_xy_average_on_rectangle():
    average = analytic_cell_averages(lambda x, y: x * y, [[0.0, 0.0]], [[1.0, 2.0]])
    assert average == pytest.approx(0.5)


def test_3d_x_squared_on_unit_cube_is_not_the_center_value():
    average = analytic_cell_averages(
        lambda x, y, z: x**2, [[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]]
    )
    center_sample = 0.5**2
    assert average == pytest.approx(1.0 / 3.0)
    assert average != pytest.approx(center_sample)


def test_degree_7_monomial_is_exact_on_unit_interval():
    average = analytic_cell_averages(lambda x: x**7, 0.0, 1.0)
    assert average == pytest.approx(0.125)


def test_time_dependent_linear_field():
    average = analytic_cell_averages(lambda x, t: x + t, 0.0, 1.0, t=3.0)
    assert average == pytest.approx(3.5)


@pytest.mark.parametrize(
    ("u_exact", "cell_lo", "cell_hi", "kwargs"),
    [
        (lambda x: x, np.array([]), np.array([]), {}),
        (lambda x: x, np.array([0.0, 1.0]), np.array([1.0]), {}),
        (lambda *coords: 1.0, np.zeros((1, 4)), np.ones((1, 4)), {}),
        (lambda x: x, ["a"], ["b"], {}),
        (lambda x: x, np.array([np.nan]), np.array([1.0]), {}),
        (lambda x: x, 1.0, 0.0, {}),
        (lambda x: x, 0.0, 0.0, {}),
        ("not-callable", 0.0, 1.0, {}),
        (lambda x: np.nan, 0.0, 1.0, {}),
        (lambda x, t: x + t, 0.0, 1.0, {"t": np.array([1.0, 2.0])}),
        (lambda x, t: x + t, 0.0, 1.0, {"t": np.inf}),
    ],
    ids=[
        "empty_bounds",
        "shape_mismatch",
        "dimension_4",
        "non_numeric_bounds",
        "non_finite_bounds",
        "inverted_extent",
        "zero_extent",
        "non_callable",
        "non_finite_samples",
        "non_scalar_t",
        "non_finite_t",
    ],
)
def test_fail_closed_on_invalid_bounds_callable_or_time(u_exact, cell_lo, cell_hi, kwargs):
    with pytest.raises(ValueError):
        analytic_cell_averages(u_exact, cell_lo, cell_hi, **kwargs)


def test_true_averages_match_oracle_while_midpoints_do_not():
    lo = np.array([0.0, 1.0])
    hi = np.array([1.0, 2.0])
    volumes = hi - lo
    field = np.array([1.0 / 3.0, 7.0 / 3.0])
    oracle = analytic_cell_averages(lambda x: x**2, lo, hi)
    midpoints = (0.5 * (lo + hi)) ** 2
    exact = reference_errors(field, oracle, volumes)
    vs_centers = reference_errors(field, midpoints, volumes)
    assert exact.l1 == pytest.approx(0.0)
    assert exact.l2 == pytest.approx(0.0)
    assert exact.linf == pytest.approx(0.0)
    assert vs_centers.l1 > 0.0
