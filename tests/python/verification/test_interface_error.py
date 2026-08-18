"""Coarse-fine interface-band errors from already-sampled fields (plan §7.9)."""
from __future__ import annotations

import numpy as np
import pytest

from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.interface_error import (
    band_max_abs_error,
    interface_band_mask,
    interface_bulk_ratio,
    max_error_location,
)


def test_default_band_uses_strict_four_cell_threshold():
    distance = np.array([0.0, 3.9, 4.0, 5.0])
    mask = interface_band_mask(distance, h_fine=1.0)
    np.testing.assert_array_equal(mask, [True, True, False, False])


def test_explicit_band_cells_narrows_mask():
    distance = np.array([0.0, 1.5, 2.0, 3.0])
    mask = interface_band_mask(distance, h_fine=1.0, band_cells=2)
    np.testing.assert_array_equal(mask, [True, True, False, False])


def test_interface_band_ignores_bulk_spike():
    field = np.array([0.0, 0.25, 0.0, 5.0])
    oracle = np.zeros(4)
    interface = np.array([False, True, True, False])
    np.testing.assert_allclose(band_max_abs_error(field, oracle, interface), 0.25)


def test_bulk_complement_sees_the_spike():
    field = np.array([0.0, 0.25, 0.0, 5.0])
    oracle = np.zeros(4)
    bulk = np.array([True, False, False, True])
    np.testing.assert_allclose(band_max_abs_error(field, oracle, bulk), 5.0)


def test_interface_bulk_ratio_matches_plan_quotient():
    np.testing.assert_allclose(interface_bulk_ratio(0.25, 5.0), 0.05)


def test_max_error_location_is_first_band_peak():
    field = np.array([0.0, 0.25, 0.0, 5.0])
    oracle = np.zeros(4)
    interface = np.array([False, True, True, False])
    assert max_error_location(field, oracle, interface) == (1,)


def test_max_error_location_is_2d_index_tuple():
    field = np.zeros((3, 4))
    field[1, 2] = 0.4
    field[2, 3] = 9.0
    oracle = np.zeros((3, 4))
    interface = np.zeros((3, 4), dtype=bool)
    interface[1, :] = True
    assert max_error_location(field, oracle, interface) == (1, 2)
    np.testing.assert_allclose(band_max_abs_error(field, oracle, interface), 0.4)


def test_unmasked_covered_parent_does_not_enter_either_band():
    field = np.array([0.1, 0.2, 50.0])
    oracle = np.zeros(3)
    interface = np.array([True, False, False])
    bulk = np.array([False, True, False])
    np.testing.assert_allclose(band_max_abs_error(field, oracle, interface), 0.1)
    np.testing.assert_allclose(band_max_abs_error(field, oracle, bulk), 0.2)


def test_open_band_error_and_ratio_are_nonzero_observations():
    error = band_max_abs_error([0.0, 0.3], [0.0, 0.0], [True, True])
    ratio = interface_bulk_ratio(0.3, 1.5)
    assert error != 0.0
    assert ratio != 0.0
    np.testing.assert_allclose(error, 0.3)
    np.testing.assert_allclose(ratio, 0.2)


def test_band_max_abs_error_matches_reference_errors_linf():
    field = np.array([0.0, 0.25, -0.5, 5.0])
    oracle = np.zeros(4)
    interface = np.array([False, True, True, False])
    subset_error = reference_errors(field[interface], oracle[interface], np.ones(2))
    np.testing.assert_allclose(
        band_max_abs_error(field, oracle, interface), subset_error.linf
    )


@pytest.mark.parametrize(
    ("distance", "h_fine", "band_cells"),
    [
        (np.array([]), 1.0, 4),
        (["a", "b"], 1.0, 4),
        ([0.0, np.nan], 1.0, 4),
        ([0.0, 1.0], [1.0, 2.0], 4),
        ([0.0, 1.0], np.nan, 4),
        ([0.0, 1.0], 0.0, 4),
        ([0.0, 1.0], -1.0, 4),
        ([0.0, 1.0], 1.0, 0),
        ([0.0, 1.0], 1.0, -2),
        ([0.0, 1.0], 1.0, np.inf),
    ],
    ids=[
        "empty",
        "non_numeric",
        "non_finite_distance",
        "non_scalar_h_fine",
        "non_finite_h_fine",
        "zero_h_fine",
        "negative_h_fine",
        "zero_band_cells",
        "negative_band_cells",
        "non_finite_band_cells",
    ],
)
def test_fail_closed_on_invalid_band_mask_inputs(distance, h_fine, band_cells):
    with pytest.raises(ValueError):
        interface_band_mask(distance, h_fine=h_fine, band_cells=band_cells)


@pytest.mark.parametrize(
    ("u", "u_exact", "mask"),
    [
        (np.array([]), np.array([]), np.array([], dtype=bool)),
        ([1.0, 0.0], [0.0], np.array([True, True])),
        ([1.0, 0.0], [0.0, 0.0], [1, 0]),
        ([1.0, 0.0], [0.0, 0.0], np.array([False, False])),
        (["a", "b"], [0.0, 0.0], np.array([True, True])),
        ([1.0, np.nan], [0.0, 0.0], np.array([True, True])),
    ],
    ids=[
        "empty",
        "length_mismatch",
        "non_boolean_mask",
        "empty_band",
        "non_numeric",
        "non_finite",
    ],
)
def test_fail_closed_on_invalid_band_error_and_location(u, u_exact, mask):
    with pytest.raises(ValueError):
        band_max_abs_error(u, u_exact, mask)
    with pytest.raises(ValueError):
        max_error_location(u, u_exact, mask)


@pytest.mark.parametrize(
    ("e_cf", "e_bulk"),
    [
        ("a", 1.0),
        ([1.0], 1.0),
        (1.0, np.nan),
        (np.inf, 1.0),
        (1.0, 0.0),
    ],
    ids=["non_numeric", "non_scalar", "non_finite_bulk", "non_finite_cf", "zero_bulk"],
)
def test_fail_closed_on_invalid_ratio(e_cf, e_bulk):
    with pytest.raises(ValueError):
        interface_bulk_ratio(e_cf, e_bulk)
