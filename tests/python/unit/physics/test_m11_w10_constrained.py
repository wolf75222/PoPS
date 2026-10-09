"""Exact W10 matrix, compatibility, independent oracle and public composition."""

import numpy as np
import pops
import pytest

from tests.python.support.m11_w10_case import case_module, oracle


def test_rank_two_friction_requires_the_augmented_product():
    matrix = oracle.LAPLACIAN
    np.testing.assert_array_equal(matrix, case_module.FRICTION_L)
    np.testing.assert_array_equal(matrix @ np.ones(3), np.zeros(3))
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    assert singular_values[-1] < 1.e-14 < singular_values[-2]
    assert np.linalg.matrix_rank(matrix) == 2
    assert np.linalg.matrix_rank(oracle.augmented_matrix()) == 4
    assert oracle.augmented_matrix()[3, 3] == 0.  # multiplier block is singular too


@pytest.mark.parametrize("force", oracle.force_samples(), ids=("baseline", "wave1", "wave2"))
def test_original_equations_constraint_and_reference_multiplier(force):
    flux, multiplier = oracle.reference(force)
    metrics = oracle.residuals(force, flux, multiplier)
    assert all(value < oracle.THRESHOLD for value in metrics.values())
    independent = force / np.diag(oracle.LAPLACIAN)[:, None, None]
    assert np.max(np.abs(independent.sum(axis=0))) > 1.e-4
    repaired = independent - independent.mean(axis=0)
    assert oracle.residuals(force, repaired, np.zeros_like(multiplier))["original_residual"] > 1.e-3


def test_incompatible_force_can_solve_augmented_equation_but_not_the_original():
    force = oracle.force_samples()[1].copy()
    force[0, 1, 2] += .3
    flux, multiplier = oracle.reference(force)
    metrics = oracle.residuals(force, flux, multiplier)
    assert metrics["augmented_residual"] < oracle.THRESHOLD
    assert metrics["constraint"] < oracle.THRESHOLD
    assert metrics["original_residual"] > .09
    np.testing.assert_allclose(multiplier[0], force.sum(axis=0)/3, rtol=0., atol=2.e-16)


@pytest.mark.parametrize("permuted", (False, True))
def test_public_product_emits_both_original_guards_before_commit(permuted):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph

    case, layout, _, _ = case_module.build_case(permuted=permuted)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    emitted = emit_cpp_program(resolved.time, model_graph=graph)
    assert "prepare_local_nonlinear_problem<4>" in emitted
    assert "original_friction" in emitted and "zero_sum_flux" in emitted
    guards = [value for value in resolved.time._values if value.op == "acceptance_guard"]
    assert len(guards) == 4
    assert len(resolved.time._commits) == 2
    for committed in resolved.time._commits.values():
        assert committed.op == "acceptance_guard"
        assert committed.inputs[0].op == "acceptance_guard"
