"""Oracle independence and negative controls; no native execution required."""

import importlib
from dataclasses import replace

import numpy as np
import pytest

from pops_witnesses.reference import (
    VARIANTS,
    Physics,
    analytic_cubic_20,
    budget,
    cell_means,
    check_arrays,
    check_pairs,
    cosine_20,
    dense_screened,
    fields_rate,
    laplacian,
    modal_reference,
    reference,
    screened,
)


@pytest.mark.parametrize("variant", tuple(VARIANTS))
def test_fft_against_dense_periodic_matrix(variant):
    p = VARIANTS[variant]
    q, d = cell_means(p).values()
    # Also cover stage-generated harmonics, not only a single input mode.
    q = reference(p)["q2"]
    for sigma, rhs in ((3, q + d), (5, q - 0.75 * d)):
        value = screened(rhs, sigma, p)
        assert np.max(np.abs(value - dense_screened(rhs, sigma, p))) < 5e-14
        assert np.max(np.abs(-laplacian(value, p) + sigma * value - rhs)) < 2e-13


@pytest.mark.parametrize("variant", ("baseline", "linear"))
def test_linear_full_grid_against_affine_modal_recurrence(variant):
    p = VARIANTS[variant]
    full, modal = reference(p), modal_reference(p)
    for key in modal:
        assert np.max(np.abs(full[key] - modal[key])) < 2e-15


def test_first_stage_relations_and_harmonic():
    all_arrays = {key: reference(p) for key, p in VARIANTS.items()}
    result = check_pairs(all_arrays)
    assert abs(result["cosine_20"]["analytic"] + 1.13e-7) < 1e-9
    p = VARIANTS["cubic"]
    delta = all_arrays["cubic"]["q1"] - all_arrays["linear"]["q1"]
    assert abs(cosine_20(delta) - analytic_cubic_20(p)) < 1e-15
    mean_linear = np.mean(all_arrays["linear"]["q1"] - all_arrays["baseline"]["q1"])
    assert abs(mean_linear - 2.783333333333333e-6) < 1e-15


@pytest.mark.parametrize("variant", tuple(VARIANTS))
def test_predefined_budget_accepts_independent_reference(variant):
    p = VARIANTS[variant]
    check_arrays(reference(p), p)
    assert budget(p)["lipschitz"] < 52
    assert budget(p)["bounds"]["qfinal"] < 1e-12


@pytest.mark.parametrize("wrong", ("baseline", "square", "linearized", "frozen-fields"))
def test_wrong_cubic_calculations_are_rejected(wrong):
    p = VARIANTS["cubic"]
    if wrong == "baseline":
        arrays = reference(VARIANTS["baseline"])
    elif wrong == "square":
        arrays = reference(p, cubic_power=2)
    elif wrong == "linearized":
        arrays = reference(p, cubic_power=1)
    else:
        arrays = reference(p, freeze_fields=True)
    with pytest.raises(ValueError, match="oracle mismatch"):
        check_arrays(arrays, p)


def test_q_enclosure_violation_invalidates_budget():
    p = VARIANTS["cubic"]
    arrays = reference(p)
    arrays["q1"].fill(3.001)
    with pytest.raises(ValueError, match="Q=3 premise"):
        check_arrays(arrays, p)


def test_donor_one_bit_mutation_fails_exact_contract():
    p = VARIANTS["linear"]
    arrays = reference(p)
    arrays["d_final"][0, 0, 0] = np.nextafter(arrays["d_final"][0, 0, 0], np.inf)
    with pytest.raises(ValueError, match="donor changed"):
        check_arrays(arrays, p)


def test_cube_of_cell_mean_differs_from_cell_mean_of_continuous_cube():
    p = Physics()
    q = cell_means(p)["q"][0, 0, 0]
    # Independently integrate the continuous cubic in cell (0,0).
    nodes, weights = np.polynomial.legendre.leggauss(12)
    x = (nodes + 1) * np.pi / p.cells[0]
    y = (nodes + 1) * np.pi / p.cells[1]
    continuous = 2 + 0.125 * np.cos(y[:, None]) * np.cos(x[None, :])
    average_cube = float(np.sum(continuous**3 * weights[:, None] * weights[None, :]) / 4)
    assert abs(q**3 - average_cube) > 1e-6
    qgrid, d = cell_means(p).values()
    cubic = fields_rate(qgrid, d, replace(p, beta=0.05))[2]
    linear = fields_rate(qgrid, d, p)[2]
    assert np.max(np.abs(cubic - linear + 0.05 * qgrid**3)) < 3e-16


def test_reader_import_has_no_pops_dependency():
    module = importlib.import_module("pops_witnesses.reference")
    assert "pops" not in module.__dict__
