"""Independent checks of the exact finite M09 witness, without old kernels."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import importlib.util

import numpy as np


SOURCE = Path(__file__).resolve().parents[4] / "examples/migration/scientific/api040_m09_hoffart_oracle.py"
SPEC = importlib.util.spec_from_file_location("api040_m09_hoffart_oracle", SOURCE)
assert SPEC and SPEC.loader
import sys

oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oracle
SPEC.loader.exec_module(oracle)


def test_original_dense_and_eliminated_equations_and_cn_energy() -> None:
    problem = oracle.witness()
    dense = oracle.monolithic(problem)
    reduced = oracle.schur(problem)
    report = oracle.measurements(problem)
    assert np.linalg.matrix_rank(problem.a) == 8
    assert np.linalg.matrix_rank(problem.k) == 4
    assert report["original_residual_monolithic"] < oracle.MAX_ERROR
    assert report["original_residual_schur"] < oracle.MAX_ERROR
    assert report["mono_schur_error"] < oracle.MAX_ERROR
    assert report["cn_energy_defect"] < oracle.MAX_ERROR
    np.testing.assert_allclose(dense[0], reduced[0], rtol=0, atol=oracle.MAX_ERROR)
    np.testing.assert_allclose(dense[1], reduced[1], rtol=0, atol=oracle.MAX_ERROR)
    assert np.isfinite(report["velocity_phase_before"])
    assert np.isfinite(report["velocity_phase_after"])


def test_first_equation_and_gauss_law_detect_independent_perturbations() -> None:
    problem = oracle.witness()
    velocity, potential = oracle.schur(problem)
    assert oracle.original_residual(problem, velocity + 1e-5, potential) > 1e-8
    assert oracle.original_residual(problem, velocity, potential + 1e-5) > 1e-8


def test_wrong_coupling_does_not_pass_original_equations() -> None:
    problem = oracle.witness()
    wrong = replace(problem, c=-problem.c)
    velocity, potential = oracle.schur(wrong)
    assert oracle.original_residual(problem, velocity, potential) > 1e-5


def test_rng_seed_changes_both_block_maps_and_phase() -> None:
    first, other = oracle.witness(), oracle.witness(20260929)
    assert not np.array_equal(first.b, other.b)
    assert not np.array_equal(first.f, other.f)
    assert oracle.measurements(first)["velocity_phase_after"] != oracle.measurements(other)["velocity_phase_after"]
