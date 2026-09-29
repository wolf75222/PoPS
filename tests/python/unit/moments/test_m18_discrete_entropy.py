"""Independent cone, damped-dual and public-source checks for M18/W09."""

import numpy as np
import pops
import pytest

from examples.migration.scientific.api040_m18_entropy import (
    BASIS, QUADRATURE, WEIGHTS, make_case, moderate_multipliers, target_moments,
)
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.moments.closures import DiscreteEntropyQuadrature
from tests.python.support.discrete_entropy_oracle import (
    BASIS as ORACLE_BASIS, WEIGHTS as ORACLE_WEIGHTS,
    cone_position, damped_dual_newton, entropy, target_from_multiplier,
)


def test_twenty_moderate_targets_are_interior_and_minimize_entropy():
    multipliers = moderate_multipliers()
    actual = target_moments(multipliers)
    assert multipliers.shape == actual.shape == (3, 4, 5)
    np.testing.assert_array_equal(np.asarray(BASIS), ORACLE_BASIS)
    np.testing.assert_array_equal(np.asarray(WEIGHTS), ORACLE_WEIGHTS)
    _, _, right = np.linalg.svd(ORACLE_BASIS, full_matrices=True)
    null = right[3:].T
    for x, y in np.ndindex(4, 5):
        target = actual[:, x, y]
        position, floor = cone_position(target)
        assert position == "interior" and floor > 1.e-3
        dual, population = damped_dual_newton(target)
        np.testing.assert_allclose(ORACLE_BASIS@population, target, rtol=0, atol=1.e-12)
        assert np.all(population > 0)
        np.testing.assert_allclose(dual, multipliers[:, x, y], rtol=0, atol=2.e-10)
        baseline = entropy(population)
        for coefficients in ((.01, -.005), (-.012, .008), (.003, .016)):
            varied = population + null@np.asarray(coefficients)
            assert np.min(varied) > 0
            np.testing.assert_allclose(ORACLE_BASIS@varied, ORACLE_BASIS@population,
                                       rtol=0, atol=1.e-14)
            assert entropy(varied) > baseline + 1.e-7


def test_outside_and_boundary_are_distinct_from_interior():
    inside = target_from_multiplier((.1, -.2, .15))
    boundary = ORACLE_BASIS[:, -1]  # only the v=1 population can realize this ray
    outside = boundary.copy()
    outside[2] = 1.1  # v²<=1 at all five nodes, with mass one
    assert cone_position(inside)[0] == "interior"
    assert cone_position(boundary)[0] == "boundary"
    assert cone_position(outside)[0] == "outside"


def test_parametric_quadrature_refuses_invalid_coefficients():
    with pytest.raises(ValueError, match="strictly positive"):
        DiscreteEntropyQuadrature((0., 1.), (1., 0.), ((1., 1.),))
    with pytest.raises(ValueError, match="full row rank"):
        DiscreteEntropyQuadrature((0., 1.), (1., 1.), ((1., 1.), (2., 2.)))
    assert QUADRATURE.moment_count == 3 and QUADRATURE.node_count == 5


def test_public_local_residual_compiles_exp_without_python_cell_callback():
    case, layout, _ = make_case()
    plan = pops.resolve(pops.validate(case), layout=layout)
    token = next(value for value in plan.time._values if value.op == "solve_coupled_implicit")
    assert token.attrs["output_count"] == 1 and len(token.inputs) == 2
    assert len(plan.time._commits) == 1
    assert all("dual" in state.qualified_id for state in plan.time._commits)
    graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    source = emit_cpp_program(plan.time, model=graph)
    assert "prepare_local_nonlinear_problem<3>" in source
    assert source.count("std::exp(") >= 5
    assert "pops::solve_prepared_local_nonlinear" in source
