"""Discrete conservation residual from an already-reduced balance (plan §7.5 / §7.0)."""
from __future__ import annotations

import numpy as np
import pytest

from pops.output.diagnostics import BalanceTerms
from verification.pops_verify.conservation import (
    conservation_residual,
    conservation_tolerance,
)


def test_closed_periodic_no_sources_is_zero_residual():
    residual = conservation_residual(0.0, 0.0, 0.0)
    np.testing.assert_allclose(residual, 0.0)


def test_outward_flux_balances_storage_loss():
    residual = conservation_residual(-1.0, 1.0, 0.0)
    np.testing.assert_allclose(residual, 0.0)


def test_source_balances_storage_gain():
    residual = conservation_residual(1.0, 0.0, 1.0)
    np.testing.assert_allclose(residual, 0.0)


def test_flux_statement_matches_plan_seven_five():
    q_t = 12.0
    q_0 = 10.0
    integrated_sources = 3.0
    integrated_outward_flux = 1.0
    residual = conservation_residual(
        q_t - q_0, integrated_outward_flux, integrated_sources
    )
    np.testing.assert_allclose(residual, 0.0)


def test_raw_storage_change_is_not_closed_when_flux_is_omitted():
    residual = conservation_residual(-1.0, 0.0, 0.0)
    np.testing.assert_allclose(residual, -1.0)


def test_reflux_and_projection_close_discrete_ledger():
    residual = conservation_residual(1.0, 0.0, 0.0, reflux=0.6, projection=0.4)
    np.testing.assert_allclose(residual, 0.0)


def test_open_statement_yields_nonzero_residual():
    residual = conservation_residual(1.0, 0.25, 0.1, reflux=0.0, projection=0.0)
    np.testing.assert_allclose(residual, 1.15)
    assert float(residual) != 0.0


def test_closed_series_is_zero_and_scalars_broadcast():
    storage_change = np.array([-1.0, -2.0, -3.0])
    residual = conservation_residual(storage_change, 1.0, 0.0)
    np.testing.assert_allclose(residual, np.array([0.0, -1.0, -2.0]))
    closed = conservation_residual(storage_change, -storage_change, 0.0)
    np.testing.assert_allclose(closed, [0.0, 0.0, 0.0])


def test_matches_public_balance_terms_residual():
    terms = BalanceTerms(
        storage_change=7.0,
        outward_boundary_flux=2.0,
        sources=3.0,
        reflux=1.0,
        projection=0.5,
    )
    residual = conservation_residual(
        terms.storage_change,
        terms.outward_boundary_flux,
        terms.sources,
        terms.reflux,
        terms.projection,
    )
    np.testing.assert_allclose(residual, terms.residual)


@pytest.mark.parametrize(
    ("storage_change", "outward_boundary_flux", "sources", "reflux", "projection"),
    [
        (np.array([]), 0.0, 0.0, 0.0, 0.0),
        ([[1.0, 2.0]], 0.0, 0.0, 0.0, 0.0),
        ([1.0, 2.0], [0.0, 0.0, 0.0], 0.0, 0.0, 0.0),
        (["a"], 0.0, 0.0, 0.0, 0.0),
        (np.nan, 0.0, 0.0, 0.0, 0.0),
        (1.0, np.inf, 0.0, 0.0, 0.0),
    ],
    ids=[
        "empty",
        "not_1d_matrix",
        "unbroadcastable",
        "non_numeric",
        "non_finite_storage",
        "non_finite_flux",
    ],
)
def test_fail_closed_on_invalid_balance_terms(
    storage_change, outward_boundary_flux, sources, reflux, projection
):
    with pytest.raises(ValueError):
        conservation_residual(
            storage_change,
            outward_boundary_flux,
            sources,
            reflux,
            projection,
        )


def test_tolerance_abs_branch_wins():
    tol = conservation_tolerance(1.0, abs_tol=1e-6, rel_tol=1e-12, n_updates=1)
    np.testing.assert_allclose(tol, 1e-6)


def test_tolerance_rel_branch_wins():
    tol = conservation_tolerance(1e3, abs_tol=1e-12, rel_tol=1e-8, n_updates=1)
    np.testing.assert_allclose(tol, 1e-5)


def test_tolerance_roundoff_floor_wins():
    eps = np.finfo(np.float64).eps
    tol = conservation_tolerance(0.0, abs_tol=0.0, rel_tol=1e-12, n_updates=10, c=2.0)
    np.testing.assert_allclose(tol, 2.0 * eps * 10.0)


def test_near_zero_scale_is_not_universal_1e_minus_12():
    eps = np.finfo(np.float64).eps
    tol = conservation_tolerance(0.0, abs_tol=1e-16, rel_tol=1e-12, n_updates=1)
    assert tol != 1e-12
    np.testing.assert_allclose(tol, max(1e-16, eps))


@pytest.mark.parametrize(
    ("q_scale", "kwargs"),
    [
        ("a", {"abs_tol": 1e-12, "rel_tol": 1e-12, "n_updates": 1}),
        ([1.0], {"abs_tol": 1e-12, "rel_tol": 1e-12, "n_updates": 1}),
        (1.0, {"abs_tol": np.nan, "rel_tol": 1e-12, "n_updates": 1}),
        (1.0, {"abs_tol": -1e-12, "rel_tol": 1e-12, "n_updates": 1}),
        (1.0, {"abs_tol": 1e-12, "rel_tol": -1e-12, "n_updates": 1}),
        (1.0, {"abs_tol": 1e-12, "rel_tol": 1e-12, "n_updates": -1}),
        (1.0, {"abs_tol": 1e-12, "rel_tol": 1e-12, "n_updates": 1, "c": -1.0}),
        (np.inf, {"abs_tol": 1e-12, "rel_tol": 1e-12, "n_updates": 1}),
    ],
    ids=[
        "non_numeric",
        "non_scalar",
        "non_finite_abs",
        "negative_abs",
        "negative_rel",
        "negative_updates",
        "negative_c",
        "non_finite_scale",
    ],
)
def test_fail_closed_on_invalid_tolerance(q_scale, kwargs):
    with pytest.raises(ValueError):
        conservation_tolerance(q_scale, **kwargs)
