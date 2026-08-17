"""Volume-weighted oracle-error norms against an external field (plan §7.1)."""
from __future__ import annotations

import numpy as np
import pytest

from verification.pops_verify.reference_errors import reference_errors


def test_exact_match_yields_zero_norms():
    field = np.array([1.5, -2.0, 0.25])
    volumes = np.array([1.0, 2.0, 0.5])
    result = reference_errors(field, field, volumes)
    assert result.l1 == 0.0
    assert result.l2 == 0.0
    assert result.linf == 0.0


def test_uniform_absolute_error_on_equal_volumes_equals_that_error():
    error = 3.0
    oracle = np.array([1.0, -4.0, 2.5])
    field = oracle + error
    volumes = np.array([2.0, 2.0, 2.0])
    result = reference_errors(field, oracle, volumes)
    assert result.l1 == pytest.approx(error)
    assert result.l2 == pytest.approx(error)
    assert result.linf == pytest.approx(error)


def test_large_volume_cell_dominates_l1_and_l2_while_linf_is_max_pointwise_error():
    field = np.array([1.0, 5.0])
    oracle = np.array([0.0, 0.0])
    volumes = np.array([100.0, 1.0])
    result = reference_errors(field, oracle, volumes)
    total_volume = 101.0
    assert result.l1 == pytest.approx((100.0 * 1.0 + 1.0 * 5.0) / total_volume)
    assert result.l2 == pytest.approx(np.sqrt((100.0 * 1.0 + 1.0 * 25.0) / total_volume))
    assert result.linf == pytest.approx(5.0)
    assert result.l1 < 1.1
    assert result.l2 < 1.2


@pytest.mark.parametrize(
    ("field", "oracle", "volumes"),
    [
        (np.ones(2), np.ones(3), np.ones(2)),
        (np.array([]), np.array([]), np.array([])),
        (np.array([np.nan]), np.array([0.0]), np.array([1.0])),
        (np.array([1.0]), np.array([np.inf]), np.array([1.0])),
        (np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])),
        (np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.array([1.0, -2.0])),
        (np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([1e308, 1e308])),
    ],
    ids=[
        "shape_mismatch",
        "empty",
        "non_finite_nan",
        "non_finite_inf",
        "zero_total_volume",
        "negative_total_volume",
        "overflowed_reductions",
    ],
)
def test_fail_closed_on_invalid_input(field, oracle, volumes):
    with pytest.raises(ValueError):
        reference_errors(field, oracle, volumes)


def test_unsigned_integer_difference_does_not_wrap():
    field = np.array([0], dtype=np.uint64)
    oracle = np.array([np.iinfo(np.uint64).max], dtype=np.uint64)
    volumes = np.array([1.0])
    result = reference_errors(field, oracle, volumes)
    expected = float(np.iinfo(np.uint64).max)
    assert result.l1 == pytest.approx(expected)
    assert result.l2 == pytest.approx(expected)
    assert result.linf == pytest.approx(expected)


def test_signed_integer_volume_sum_does_not_overflow():
    vmax = np.iinfo(np.int64).max
    field = np.array([1.0, 1.0])
    oracle = np.array([0.0, 0.0])
    volumes = np.array([vmax, vmax], dtype=np.int64)
    result = reference_errors(field, oracle, volumes)
    assert result.l1 == pytest.approx(1.0)
    assert result.l2 == pytest.approx(1.0)
    assert result.linf == pytest.approx(1.0)


def test_unsigned_integer_volume_sum_does_not_overflow():
    vmax = np.iinfo(np.uint64).max
    field = np.array([1.0, 1.0])
    oracle = np.array([0.0, 0.0])
    volumes = np.array([vmax, vmax], dtype=np.uint64)
    result = reference_errors(field, oracle, volumes)
    assert result.l1 == pytest.approx(1.0)
    assert result.l2 == pytest.approx(1.0)
    assert result.linf == pytest.approx(1.0)
