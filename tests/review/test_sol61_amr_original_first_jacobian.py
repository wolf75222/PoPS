"""Actual frozen AMR request matrix counterchecks; no native/PoPS import."""

import math

import numpy as np
import pytest

from tests.review import sol61_amr_original_first_jacobian_oracle as oracle


@pytest.mark.parametrize("cells,dofs", [(16, 72), (32, 144)])
@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1)])
def test_real_active_faces_metric_and_request_controls(cells, dofs, order):
    problem = oracle.source_matrix(cells, order)
    assert problem["matrix"].shape == (dofs, dofs)
    assert problem["controls"] == dict(restart=80, linear_max_iterations=240, linear_tolerance=1e-5)
    assert np.all(problem["metric"] > 0)
    assert np.array_equal(problem["scalar"] @ np.ones(dofs // 3), np.zeros(dofs // 3))
    assert np.array_equal(problem["metric"][::3] @ problem["scalar"], np.zeros(dofs // 3))
    weighted = problem["metric"][:, None] * problem["matrix"]
    assert np.linalg.norm(weighted - weighted.T) > 1  # Never invent an SPD model.
    assert len({row["path"] for row in problem["pins"]}) == 7


def test_source_initial_forcing_agrees_with_root_native_context_without_receiving_a_runtime():
    problem = oracle.source_matrix(32)
    stop = 1e-5 * np.linalg.norm(np.sqrt(problem["metric"]) * problem["rhs"])
    # External Root typed diagnostic for N32/012, not a local native execution.
    root_stop = float.fromhex("0x1.6f18ebf6991e7p-14")
    assert math.isclose(stop, root_stop, rel_tol=1e-12, abs_tol=0)
    assert oracle.receive()["native_reproduction_claimed"] is False


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1)])
def test_two_passes_alone_do_not_close_n32_iteration_limit(order):
    problem = oracle.source_matrix(32, order)
    for passes in (1, 2):
        result = oracle.linear_algebra(problem, passes)
        assert result["columns"] == 240
        assert not result["converged"]
        assert result["actual"] > 2 * result["stop"]


@pytest.mark.parametrize("cells", [16, 32])
@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1)])
def test_spatial_right_diagonal_preserves_equation_metric_and_budgets(cells, order):
    problem = oracle.source_matrix(cells, order)
    before = {name: problem[name].copy() for name in ("matrix", "rhs", "metric", "spatial")}
    result = oracle.linear_algebra(problem, 2, True)
    assert result["converged"]
    assert result["columns"] <= 240
    assert result["actual"] <= result["stop"]
    assert result["stop"] == 1e-5 * np.linalg.norm(np.sqrt(problem["metric"]) * problem["rhs"])
    for name, value in before.items():
        assert np.array_equal(value, problem[name])


def test_all_four_actual_source_cases_are_preserved():
    receipt = oracle.receive()
    assert receipt["source_commit"] == oracle.FROZEN
    assert receipt["native_execution_here"] is False and receipt["native_states_consumed"] is False
    assert [(case["cells"], case["permutation"]) for case in receipt["cases"]] == [
        (16, [0, 1, 2]),
        (16, [2, 0, 1]),
        (32, [0, 1, 2]),
        (32, [2, 0, 1]),
    ]
