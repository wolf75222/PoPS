"""Authentication tests for the prepared Krylov nullspace/gauge IR contract."""
from __future__ import annotations

import pytest

import pops
from pops.codegen.krylov_contract import validated_prepared_problem_contract
from pops.codegen.program_codegen import emit_cpp_program
from pops.fields import ConstantNullspace, MeanValueGauge
from pops.linalg import LinearOperatorProperties, LinearProblem
from pops.solvers import CG, GMRES
from pops.time import FailRun, Program


def _solve_node(*, constant: bool = False):
    program = Program("nullspace-ir")
    operator = program.matrix_free_operator("identity")
    program.set_apply(operator, lambda _program, _out, value: value)
    rhs = program.scalar_field("rhs")
    properties = (
        LinearOperatorProperties
        .symmetric_positive_definite_on_nullspace_complement()
        if constant
        else LinearOperatorProperties.symmetric_positive_definite()
    )
    problem = LinearProblem(
        operator,
        rhs,
        properties=properties,
        nullspace=ConstantNullspace() if constant else None,
        gauge=MeanValueGauge(0) if constant else None,
    )
    program.solve(problem, solver=CG(max_iter=3)).consume(action=FailRun())
    node = next(value for value in program._values if value.op == "solve_linear")
    return node.inputs[0], node


def _mutable_attrs(node):
    attrs = dict(node.attrs)
    for key in (
        "nullspace_contract", "gauge_contract", "operator_properties", "krylov_footprint"
    ):
        attrs[key] = dict(attrs[key])
    return attrs


@pytest.mark.parametrize("constant", [False, True])
def test_prepared_problem_contract_round_trips_exact_canonical_metadata(constant):
    operator, node = _solve_node(constant=constant)
    contract = validated_prepared_problem_contract(node.attrs, operator=node.inputs[0])

    assert contract["operator_properties"] == node.attrs["operator_properties"]
    assert contract["nullspace_contract"] == node.attrs["nullspace_contract"]
    assert contract["gauge_contract"] == node.attrs["gauge_contract"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda attrs: attrs.__setitem__("nullspace_contract", {"kind": "none"}),
        lambda attrs: attrs.__setitem__(
            "nullspace_contract", {"schema_version": True, "kind": "none"}),
        lambda attrs: attrs.__setitem__(
            "gauge_contract", {"schema_version": 1, "kind": "mean_value"}),
        lambda attrs: attrs["operator_properties"].__setitem__(
            "positive_definite_on_nullspace_complement", 0),
        lambda attrs: attrs["operator_properties"].__setitem__("unexpected", False),
    ],
)
def test_prepared_problem_contract_rejects_missing_versions_wrong_types_and_extra_keys(mutate):
    operator, node = _solve_node()
    attrs = _mutable_attrs(node)
    mutate(attrs)

    with pytest.raises(ValueError, match="nullspace|gauge|operator-property"):
        validated_prepared_problem_contract(attrs, operator=operator)


def test_prepared_problem_contract_rejects_global_spd_for_a_constant_nullspace():
    operator, node = _solve_node(constant=True)
    attrs = _mutable_attrs(node)
    attrs["operator_properties"] = (
        LinearOperatorProperties.symmetric_positive_definite().canonical_data())

    with pytest.raises(ValueError, match="globally positive definite"):
        validated_prepared_problem_contract(attrs, operator=operator)


def test_prepared_problem_contract_rejects_a_nonsymmetric_constant_kernel_assertion():
    operator, node = _solve_node(constant=True)
    attrs = _mutable_attrs(node)
    attrs["operator_properties"] = LinearOperatorProperties.general().canonical_data()

    with pytest.raises(ValueError, match="symmetric operator certificate"):
        validated_prepared_problem_contract(attrs, operator=operator)


def test_prepared_problem_contract_rejects_complement_spd_without_a_nullspace():
    operator, node = _solve_node()
    attrs = _mutable_attrs(node)
    attrs["operator_properties"] = (
        LinearOperatorProperties
        .symmetric_positive_definite_on_nullspace_complement()
        .canonical_data()
    )

    with pytest.raises(ValueError, match="requires a declared nullspace"):
        validated_prepared_problem_contract(attrs, operator=operator)


def test_general_method_accepts_explicit_symmetry_without_spd_inference():
    program = Program("general-nullspace-ir")
    operator = program.matrix_free_operator("identity")
    program.set_apply(operator, lambda _program, _out, value: value)
    problem = LinearProblem(
        operator,
        program.scalar_field("rhs"),
        properties=LinearOperatorProperties.symmetric_operator(),
        nullspace=ConstantNullspace(),
        gauge=MeanValueGauge(3),
    )
    program.solve(problem, solver=GMRES(max_iter=3, restart=2)).consume(action=FailRun())
    node = next(value for value in program._values if value.op == "solve_linear")

    contract = validated_prepared_problem_contract(node.attrs, operator=node.inputs[0])
    assert contract["nullspace_contract"] == {
        "schema_version": 1,
        "kind": "constant",
    }
    assert contract["operator_properties"] == (
        LinearOperatorProperties.symmetric_operator().canonical_data())


def test_constant_nullspace_codegen_emits_the_prepared_policy_and_gauge_snapshot():
    model = pops.Model("constant-nullspace-model")
    state = model.state("U", components=("u",))
    block = pops.Case("constant-nullspace-case").block("fluid", model)
    program = Program("constant-nullspace-codegen")
    temporal = program.state(block[state])
    operator = program.matrix_free_operator(
        "identity", domain="state", range_="state", ncomp=1)
    program.set_apply(operator, lambda _program, _out, value: value)
    problem = LinearProblem(
        operator,
        temporal.n,
        at=temporal.next.point,
        properties=(
            LinearOperatorProperties
            .symmetric_positive_definite_on_nullspace_complement()
        ),
        nullspace=ConstantNullspace(),
        gauge=MeanValueGauge(3),
    )
    solution = program.solve(problem, solver=CG(max_iter=3)).consume(action=FailRun())
    accepted = program.value("accepted", solution, at=temporal.next.point)
    program.commit(temporal.next, accepted)

    source = emit_cpp_program(program)
    assert "PreparedNullspacePolicy::preserving" in source
    assert "constant_mean_zero_nullspace" in source
    assert "gauges.front().value" in source
    assert "symmetric_positive_definite_on_nullspace_complement" in source
