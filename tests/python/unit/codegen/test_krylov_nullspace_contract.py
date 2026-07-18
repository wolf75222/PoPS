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
        lambda attrs: attrs.__setitem__("nullspace_contract", {"contract": {}}),
        lambda attrs: attrs.__setitem__(
            "nullspace_contract", {
                "schema_version": True,
                "provider": attrs["nullspace_provider"],
                "contract": {"declaration": "nonsingular"},
            }),
        lambda attrs: attrs.__setitem__(
            "gauge_contract", {"constraint": "wrong"}),
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
    assert contract["nullspace_contract"]["provider"]["singular"] is True
    assert contract["nullspace_contract"]["contract"] == {
        "basis": "constant-function", "basis_count": 1}
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

    source = emit_cpp_program(program, target="system")
    amr_source = emit_cpp_program(program, target="amr_system")
    for emitted in (source, amr_source):
        assert "PreparedNullspacePolicy::preserving" in emitted
        assert "constant_mean_zero_nullspace" in emitted
        assert "gauges.front().value" in emitted
        assert "symmetric_positive_definite_on_nullspace_complement" in emitted
        assert "ctx.program_resource_vector_distribution()" in emitted
        assert "ctx.configure_program_resource_field_nullspace(" in emitted
        assert "ctx.program_resource_field_level())" in emitted
        assert ".level_distribution.assign(static_cast<std::size_t>(ctx.nlev())" not in emitted
        assert "krylov_nullspace_grid" not in emitted


def test_registered_header_provider_owns_contract_validation_and_native_plan_emission(
    tmp_path,
):
    from pops._ir.literals import scalar_cpp, scalar_literal
    from pops.fields import nullspace

    include_root = tmp_path / "include"
    header = include_root / "vendor" / "nullspace.hpp"
    header.parent.mkdir(parents=True)
    header.write_text(
        "#pragma once\n"
        "#include <pops/numerics/elliptic/interface/field_nullspace.hpp>\n"
        "namespace vendor { inline pops::FieldNullspacePlan make_plan() {\n"
        "  return pops::constant_mean_zero_nullspace(\"vendor\", \"vendor provider\");\n"
        "} }\n",
        encoding="utf-8",
    )

    def author(options, gauge, _properties, where):
        assert options == {"revision": 7}
        if type(gauge) is not MeanValueGauge:
            raise TypeError("%s requires MeanValueGauge" % where)
        return nullspace.Contracts(
            {"basis_factory": "vendor", "revision": 7},
            {"constraint": "mean-value", "value": scalar_literal(gauge.value)},
        )

    def validate(use, where):
        if use.components is not None and use.components != 1:
            raise ValueError("%s is scalar-only" % where)
        if not use.operator_properties["symmetric"]:
            raise ValueError("%s requires symmetry" % where)
        if dict(use.contracts.nullspace) != {
            "basis_factory": "vendor", "revision": 7,
        }:
            raise ValueError("%s contract mismatch" % where)

    def emit(_node, _prelude, contracts, _identity, _provider):
        return nullspace.NativeEmission(
            "[&]() { auto plan = vendor::make_plan(); "
            "plan.gauges.front().value = static_cast<pops::Real>(%s); "
            "return plan; }()" % scalar_cpp(contracts.gauge["value"])
        )

    provider = nullspace.register(nullspace.Provider(
        provider_id="vendor.test-nullspace",
        emitter_id="vendor.test-nullspace@1",
        singular=True,
        use_policy=nullspace.UsePolicy(
            "vendor.test-nullspace.use", 1,
            {"components": 1, "basis": "provider-owned"}, validate,
        ),
        author=author,
        emitter=emit,
        native_component=nullspace.HeaderOnlyComponent(
            "vendor.test-nullspace",
            include_root=include_root,
            entry_headers=("vendor/nullspace.hpp",),
        ),
    ))

    model = pops.Model("external-nullspace-provider-model")
    state = model.state("U", components=("u",))
    block = pops.Case("external-nullspace-provider-case").block("fluid", model)
    program = Program("external-nullspace-provider")
    temporal = program.state(block[state])
    operator = program.matrix_free_operator(
        "identity", domain="state", range_="state", ncomp=1
    )
    program.set_apply(operator, lambda _program, _out, value: value)
    problem = LinearProblem(
        operator,
        temporal.n,
        at=temporal.next.point,
        properties=LinearOperatorProperties.symmetric_operator(),
        nullspace=nullspace.Prepared(provider, revision=7),
        gauge=MeanValueGauge(2),
    )
    solution = program.solve(problem, solver=GMRES(max_iter=3, restart=2)).consume(
        action=FailRun()
    )
    accepted = program.value("accepted", solution, at=temporal.next.point)
    program.commit(temporal.next, accepted)
    node = next(value for value in program._values if value.op == "solve_linear")

    assert node.attrs["nullspace_provider"] == provider.authority()
    assert node.attrs["nullspace_contract"]["contract"] == {
        "basis_factory": "vendor", "revision": 7,
    }
    emitted = emit_cpp_program(program)
    assert "#include <vendor/nullspace.hpp>" in emitted
    assert "vendor::make_plan()" in emitted
    assert "PreparedNullspacePolicy::preserving" in emitted
