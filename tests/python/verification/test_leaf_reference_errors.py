"""Leaf-only AMR oracle norms (plan §7.2) over Task 2 reference_errors."""
from __future__ import annotations

import numpy as np
import pytest

from verification.pops_verify.leaf_reference_errors import leaf_reference_errors
from verification.pops_verify.reference_errors import ReferenceErrors, reference_errors


def test_covered_parent_error_is_excluded_from_all_norms():
    field = np.array([1.0, 100.0, 2.0])
    oracle = np.array([0.0, 0.0, 0.0])
    volumes = np.array([1.0, 50.0, 1.0])
    leaf_mask = np.array([True, False, True])
    result = leaf_reference_errors(field, oracle, volumes, leaf_mask)
    assert result.l1 == pytest.approx(1.5)
    assert result.l2 == pytest.approx(np.sqrt(2.5))
    assert result.linf == pytest.approx(2.0)


def test_matches_reference_errors_on_manually_sliced_leaf_subset():
    field = np.array([1.0, 4.0, -1.0, 8.0])
    oracle = np.array([0.0, 1.0, 0.0, 2.0])
    volumes = np.array([2.0, 3.0, 5.0, 7.0])
    leaf_mask = np.array([True, False, True, False])
    result = leaf_reference_errors(field, oracle, volumes, leaf_mask)
    expected = reference_errors(field[leaf_mask], oracle[leaf_mask], volumes[leaf_mask])
    assert result.l1 == pytest.approx(expected.l1)
    assert result.l2 == pytest.approx(expected.l2)
    assert result.linf == pytest.approx(expected.linf)


def test_large_volume_leaf_dominates_l1_l2_while_linf_is_max_leaf_error():
    field = np.array([1.0, 9.0, 5.0])
    oracle = np.array([0.0, 0.0, 0.0])
    volumes = np.array([100.0, 1.0, 1.0])
    leaf_mask = np.array([True, False, True])
    result = leaf_reference_errors(field, oracle, volumes, leaf_mask)
    total_leaf_volume = 101.0
    assert result.l1 == pytest.approx((100.0 * 1.0 + 1.0 * 5.0) / total_leaf_volume)
    assert result.l2 == pytest.approx(np.sqrt((100.0 * 1.0 + 1.0 * 25.0) / total_leaf_volume))
    assert result.linf == pytest.approx(5.0)
    assert result.linf < 9.0


def test_all_true_mask_equals_full_reference_errors():
    field = np.array([1.5, -2.0, 0.25])
    oracle = np.array([0.5, -1.0, 1.25])
    volumes = np.array([1.0, 2.0, 0.5])
    leaf_mask = np.array([True, True, True])
    result = leaf_reference_errors(field, oracle, volumes, leaf_mask)
    expected = reference_errors(field, oracle, volumes)
    assert result.l1 == pytest.approx(expected.l1)
    assert result.l2 == pytest.approx(expected.l2)
    assert result.linf == pytest.approx(expected.linf)


@pytest.mark.parametrize(
    ("field", "oracle", "volumes", "leaf_mask"),
    [
        (np.ones(3), np.zeros(3), np.ones(3), np.array([False, False, False])),
        (np.ones(3), np.zeros(3), np.ones(3), np.array([0, 1, 0])),
        (np.ones(3), np.zeros(3), np.ones(3), np.array([1, 0, 1], dtype=np.uint8)),
        (np.ones(2), np.ones(2), np.ones(2), np.array([True, False, True])),
        (np.array([1.0, np.nan, 1.0]), np.zeros(3), np.ones(3), np.array([True, True, False])),
    ],
    ids=[
        "empty_leaf_set",
        "integer_mask",
        "uint8_mask",
        "shape_mismatch",
        "non_finite_leaf",
    ],
)
def test_fail_closed_on_invalid_mask_or_leaf_values(field, oracle, volumes, leaf_mask):
    with pytest.raises(ValueError):
        leaf_reference_errors(field, oracle, volumes, leaf_mask)


def test_covered_non_finite_is_excluded_before_reduction():
    field = np.array([1.0, np.nan, 1.0])
    oracle = np.array([0.0, 0.0, 0.0])
    volumes = np.array([1.0, 1.0, 1.0])
    leaf_mask = np.array([True, False, True])
    result = leaf_reference_errors(field, oracle, volumes, leaf_mask)
    assert result.l1 == pytest.approx(1.0)
    assert result.l2 == pytest.approx(1.0)
    assert result.linf == pytest.approx(1.0)


def test_delegates_once_to_reference_errors_on_leaf_subset(monkeypatch):
    calls = []

    def fake_reference_errors(u, u_exact, volumes):
        calls.append(
            (
                np.asarray(u).copy(),
                np.asarray(u_exact).copy(),
                np.asarray(volumes).copy(),
            )
        )
        return ReferenceErrors(l1=0.1, l2=0.2, linf=0.3)

    monkeypatch.setattr(
        "verification.pops_verify.leaf_reference_errors.reference_errors",
        fake_reference_errors,
    )
    field = np.array([1.0, 4.0, 2.0])
    oracle = np.array([0.0, 1.0, 0.0])
    volumes = np.array([2.0, 3.0, 5.0])
    leaf_mask = np.array([True, False, True])
    result = leaf_reference_errors(field, oracle, volumes, leaf_mask)
    assert result == ReferenceErrors(l1=0.1, l2=0.2, linf=0.3)
    assert len(calls) == 1
    selected_u, selected_exact, selected_volumes = calls[0]
    np.testing.assert_array_equal(selected_u, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(selected_exact, np.array([0.0, 0.0]))
    np.testing.assert_array_equal(selected_volumes, np.array([2.0, 5.0]))
