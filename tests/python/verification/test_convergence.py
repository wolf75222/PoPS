"""Observed order from an already-computed error series (plan §7.4)."""
from __future__ import annotations

import numpy as np
import pytest

from verification.pops_verify.convergence import observed_order
from verification.pops_verify.reference_errors import reference_errors


def test_ratio_two_pair_with_error_over_four_is_order_two():
    orders = observed_order([0.16, 0.04], [1.0, 0.5])
    np.testing.assert_allclose(orders, [2.0])


def test_ratio_two_pair_with_error_over_eight_is_order_three():
    orders = observed_order([0.08, 0.01], [1.0, 0.5])
    np.testing.assert_allclose(orders, [3.0])


def test_ratio_two_pair_with_error_over_two_is_order_one():
    orders = observed_order([0.4, 0.2], [1.0, 0.5])
    np.testing.assert_allclose(orders, [1.0])


def test_three_ratio_two_spacings_with_quadratic_error_are_both_order_two():
    spacings = np.array([1.0, 0.5, 0.25])
    errors = spacings**2
    orders = observed_order(errors, spacings)
    np.testing.assert_allclose(orders, [2.0, 2.0])


def test_canonical_five_point_series_as_reciprocal_n_is_order_two():
    n = np.array([16.0, 32.0, 64.0, 128.0, 256.0])
    spacings = 1.0 / n
    errors = spacings**2
    orders = observed_order(errors, spacings)
    assert orders.shape == (4,)
    np.testing.assert_allclose(orders, np.full(4, 2.0))


def test_non_ratio_two_spacings_still_recover_order_two():
    orders = observed_order([9.0, 1.0], [3.0, 1.0])
    np.testing.assert_allclose(orders, [2.0])


def test_fine_to_coarse_pair_matches_coarse_to_fine_order():
    coarse_to_fine = observed_order([0.16, 0.04], [1.0, 0.5])
    fine_to_coarse = observed_order([0.04, 0.16], [0.5, 1.0])
    np.testing.assert_allclose(fine_to_coarse, coarse_to_fine)


def test_divergent_series_yields_negative_order():
    orders = observed_order([0.04, 0.16], [1.0, 0.5])
    assert orders.shape == (1,)
    assert orders[0] < 0.0
    np.testing.assert_allclose(orders, [-2.0])


@pytest.mark.parametrize(
    ("errors", "resolutions"),
    [
        (np.array([]), np.array([])),
        ([0.16], [1.0]),
        ([0.16, 0.04], [1.0]),
        (0.16, 1.0),
        ([[0.16, 0.04]], [[1.0, 0.5]]),
        (["a", "b"], [1.0, 0.5]),
        ([np.nan, 0.04], [1.0, 0.5]),
        ([0.16, 0.04], [1.0, np.inf]),
        ([0.16, 0.04], [1.0, 0.0]),
        ([0.16, 0.04], [1.0, -0.5]),
        ([0.0, 0.04], [1.0, 0.5]),
        ([-0.16, 0.04], [1.0, 0.5]),
        ([0.16, 0.04], [1.0, 1.0]),
    ],
    ids=[
        "empty",
        "length_1",
        "length_mismatch",
        "not_1d_scalars",
        "not_1d_matrix",
        "non_numeric",
        "non_finite_error",
        "non_finite_spacing",
        "zero_spacing",
        "negative_spacing",
        "zero_error",
        "negative_error",
        "equal_consecutive_spacings",
    ],
)
def test_fail_closed_on_invalid_error_or_spacing_series(errors, resolutions):
    with pytest.raises(ValueError):
        observed_order(errors, resolutions)


def test_consumes_reference_error_scalars_without_recomputing_norms():
    spacings = np.array([1.0, 0.5, 0.25])
    l1_series = []
    for spacing in spacings:
        field = np.zeros(2)
        oracle = np.full(2, spacing**2)
        volumes = np.ones(2)
        l1_series.append(reference_errors(field, oracle, volumes).l1)
    orders = observed_order(l1_series, spacings)
    np.testing.assert_allclose(orders, [2.0, 2.0])
