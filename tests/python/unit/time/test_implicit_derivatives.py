"""Finite implicit derivatives preserve the original residual, boundary and gauge equations."""
from dataclasses import replace
import json

import numpy as np
import pytest
import pops

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.graph_functions import GraphFunction, graph
from pops.implicit_derivatives import ImplicitRelation, ImplicitFailure, implicit_jvp, linearize
from pops.time._program.detach import detach_compiled_program
from pops.time.variant_types import Bool, Float64, Product
from examples.api_v02.implicit_relation_checks import (
    D, U, FullResidual, LinearData, build_implicit_check, independent_oracle, make_linearization,
)


@pytest.fixture(scope='module')
def linearization():
    return make_linearization()


@pytest.fixture(scope='module')
def plan_and_cpp():
    spec = build_implicit_check('jvp')
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    return plan, emit_cpp_program(plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))


def test_complete_equations_include_boundary_and_gauge_before_differentiation(linearization):
    assert linearization.full_residual.output == FullResidual
    assert linearization.data_type == LinearData
    assert linearization.unknown_type == U and linearization.direction_type == D
    restored = GraphFunction.from_data(json.loads(json.dumps(linearization.differential.to_data())))
    assert restored.identity == linearization.differential.identity
    assert 'structured_iterate' not in json.dumps(restored.to_data())


def test_missing_gauge_does_not_silently_become_a_differentiable_square_system(linearization):
    empty = graph(inputs=linearization.relation.gauge_equations.inputs, output=Product())(
        lambda p, args: p.product())
    relation = replace(linearization.relation, gauge_equations=empty)
    with pytest.raises(NotImplementedError, match='underdetermined finite relation'):
        linearize(relation, request_test=linearization.problem.request_test)


def test_validity_and_regularity_are_mandatory_closed_named_obligations(linearization):
    with pytest.raises(TypeError, match='closed GraphFunction'):
        replace(linearization.relation, validity=lambda *args: True)
    wrong = graph(inputs=linearization.relation.validity.inputs, output=Bool)(lambda p, a: p.boolean(True))
    with pytest.raises(TypeError, match='Products'):
        replace(linearization.relation, validity=wrong)
    incomplete = graph(inputs=linearization.relation.validity.inputs,
        output=Product(differentiable=Bool, observations=Product()))(
        lambda p, args: p.product(differentiable=p.boolean(True), observations=p.product()))
    with pytest.raises(TypeError, match='regular'):
        replace(linearization.relation, validity=incomplete)


def test_request_cannot_drop_full_tangent_residual_or_data(linearization):
    original = linearization.relation.problem.request_test
    with pytest.raises(TypeError, match='exact full tangent residual'):
        linearize(linearization.relation, request_test=original)


def test_generated_linearization_cannot_be_replaced_by_an_unrelated_residual(linearization):
    with pytest.raises(TypeError, match='original equations'):
        replace(linearization, problem=linearization.relation.problem)


def test_identity_retains_base_selection_and_constraints_not_just_jacobian(linearization):
    relation = linearization.relation
    false = graph(inputs=relation.problem.selection.inputs, output=Bool)(lambda p, a: p.boolean(False))
    changed = linearize(replace(relation, problem=replace(relation.problem, selection=false)),
                        request_test=linearization.problem.request_test)
    assert changed.differential.identity == linearization.differential.identity
    assert changed.identity != linearization.identity


def test_source_contains_original_revalidation_and_regularity_guards(plan_and_cpp):
    plan, cpp = plan_and_cpp
    assert str(int(ImplicitFailure.NOT_REGULAR)) in cpp
    assert str(int(ImplicitFailure.NOT_DIFFERENTIABLE)) in cpp
    assert '2107' in cpp  # independent original-residual guards on base and tangent
    assert cpp.count('2107') >= 2
    assert 'iteration_stop_' not in cpp  # no numerical Newton trace differentiated
    assert 'ctx.axpy' in cpp or 'Pointwise' in cpp or 'pointwise' in cpp
    detached = detach_compiled_program(plan.time)
    assert detached.to_graph().to_data() == plan.time.to_graph().to_data()


def test_explicit_accuracy_is_required_and_failed_authoring_is_atomic(linearization):
    p = pops.Program('bad_implicit_accuracy')
    base = p.product(values=p.product(), data_identity=p.product(),
                     evidence=p.product(), termination=p.product())
    before = p._next_id
    with pytest.raises(TypeError, match='base_accuracy'):
        implicit_jvp(p, linearization, base=base, direction=p.product(), algorithm=None,
                     guess=p.product(), base_accuracy=None, accuracy=None, budget=p.product())
    # Argument construction is outside the transaction, but no staged solve/region survives.
    assert not any(value.op == 'capture_result' for value in p._values)
    assert p._next_id == before+3


def _independent_solve(data, initial):
    result = initial.copy()
    result[2:] = data[2:]
    for _ in range(12):
        x, y, b, g = result
        residual = np.array([x*x+y+b-data[0], x+y*y+g-data[1]])
        step = np.linalg.solve(np.array([[2*x, 1.], [1., 2*y]]), -residual)
        result[:2] += step
        if np.max(np.abs(step)) < 2e-15:
            break
    return result


def test_independent_nonlinear_solution_map_confirms_tangent_and_weighted_duality():
    base = np.array([2., 3., .5, -.25])
    data = np.array([7.5, 10.75, .5, -.25])
    direction, dual = np.array([.3, -.2, .4, .1]), np.array([.2, -.3, .5, -.7])
    tangent, adjoint, duality = independent_oracle(base, direction, dual)
    np.testing.assert_allclose(tangent[2:], direction[2:], rtol=0, atol=1e-15)
    assert abs(duality) < 3e-16 and np.isfinite(adjoint).all()
    errors = []
    for h in (2e-2, 1e-2, 5e-3):
        plus = _independent_solve(data+h*direction, base)
        minus = _independent_solve(data-h*direction, base)
        errors.append(np.max(np.abs((plus-minus)/(2*h)-tangent)))
    assert errors[0]/errors[1] > 3.8 and errors[1]/errors[2] > 3.8, errors
    assert errors[-1] < 1e-8


def test_omitted_boundary_and_gauge_directions_are_falsified_by_original_equations():
    base = np.array([2., 3., .5, -.25])
    direction, dual = np.array([.3, -.2, .4, .1]), np.ones(4)
    tangent, _, _ = independent_oracle(base, direction, dual)
    broken = np.r_[np.linalg.solve(np.array([[4., 1.], [1., 6.]]), direction[:2]), 0., 0.]
    jacobian = np.array([[4., 1., 1., 0.], [1., 6., 0., 1.],
                         [0., 0., 1., 0.], [0., 0., 0., 1.]])
    defect = jacobian@broken-direction
    np.testing.assert_allclose(defect[:2], 0., atol=1e-16)
    np.testing.assert_array_equal(defect[2:], -direction[2:])
    assert np.linalg.norm(defect, ord=np.inf) >= .4
