"""Symmetry diagnostics from an already-sampled field (plan §7.8)."""
from __future__ import annotations

import numpy as np
import pytest

from verification.pops_verify.symmetry import (
    radial_anisotropy,
    xy_symmetry_error,
)


def test_symmetric_index_sum_has_zero_swap_error():
    i = np.arange(5, dtype=np.float64)
    field = i[:, None] + i[None, :]
    np.testing.assert_allclose(xy_symmetry_error(field), 0.0)


def test_known_swap_error_is_sqrt_two():
    field = np.array([[0.0, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(xy_symmetry_error(field), np.sqrt(2.0))


def test_symmetric_gaussian_blob_has_zero_swap_error():
    axis = np.arange(9, dtype=np.float64)
    center = 4.0
    field = np.exp(-((axis[:, None] - center) ** 2 + (axis[None, :] - center) ** 2))
    np.testing.assert_allclose(xy_symmetry_error(field), 0.0, atol=1e-15)


def test_open_swap_error_is_nonzero_observation():
    i = np.arange(4, dtype=np.float64)
    field = np.broadcast_to(i[:, None], (4, 4)).copy()
    error = xy_symmetry_error(field)
    assert error != 0.0
    assert np.isfinite(error)


def test_constant_radii_have_zero_anisotropy():
    np.testing.assert_allclose(radial_anisotropy([2.0, 2.0, 2.0, 2.0]), 0.0)


def test_known_radial_anisotropy_is_one():
    np.testing.assert_allclose(radial_anisotropy([1.0, 2.0, 3.0]), 1.0)


def test_sedov_like_fourth_harmonic_anisotropy():
    r0 = 1.0
    epsilon = 0.1
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    radii = r0 + epsilon * np.cos(4.0 * theta)
    np.testing.assert_allclose(radial_anisotropy(radii), 2.0 * epsilon / r0)


def test_open_radial_anisotropy_is_nonzero_observation():
    error = radial_anisotropy([1.0, 1.5, 2.0])
    np.testing.assert_allclose(error, 2.0 / 3.0)
    assert error != 0.0


@pytest.mark.parametrize(
    "field",
    [
        np.array([]),
        1.0,
        [1.0, 2.0],
        np.zeros((2, 2, 2)),
        np.zeros((2, 3)),
        [["a", "b"], ["c", "d"]],
        [[1.0, np.nan], [0.0, 1.0]],
        np.zeros((3, 3)),
    ],
    ids=[
        "empty",
        "not_2d_scalar",
        "not_2d_vector",
        "not_2d_tensor",
        "non_square",
        "non_numeric",
        "non_finite",
        "vanishing_norm",
    ],
)
def test_fail_closed_on_invalid_swap_field(field):
    with pytest.raises(ValueError):
        xy_symmetry_error(field)


@pytest.mark.parametrize(
    "radii",
    [
        np.array([]),
        1.0,
        [[1.0, 2.0]],
        ["a", "b"],
        [1.0, np.nan],
        [-1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ],
    ids=[
        "empty",
        "not_1d_scalar",
        "not_1d_matrix",
        "non_numeric",
        "non_finite",
        "zero_mean",
        "all_zeros",
    ],
)
def test_fail_closed_on_invalid_radii(radii):
    with pytest.raises(ValueError):
        radial_anisotropy(radii)


def test_manufactured_swap_symmetric_field_is_zero_without_reference_errors():
    i = np.arange(6, dtype=np.float64)
    field = i[:, None] * i[None, :]
    assert field.shape == field.T.shape
    np.testing.assert_allclose(field, field.T)
    np.testing.assert_allclose(xy_symmetry_error(field), 0.0)
