"""Scientific producer catalogs and immutable invocation-owned observations."""

import json

from pops.codegen.program_emit_field_gradient import emit_field_gradient
from pops.codegen.program_emit_field_problem import emit_field_problem_value
from pops.codegen.program_field_provider_catalog import (
    emit_field_provider_catalog,
    field_provider_contract,
)
from pops.time import FailRun
from pops.time._program.detach import detach_compiled_program
from tests.python.unit.fields.test_program_field_problem import field_case


def observations(*, joint=False, twice=False, gradient=False):
    _case, field, problem, program, values, point = field_case(joint=joint)
    result = []
    for i in range(2 if twice else 1):
        stage = point if i == 0 else program.stage("second", c=0)
        solved = program.solve(field, values=values, at=stage).consume(action=FailRun())
        observed = field.observe(solved, byte_capacity=123456)
        for unknown in problem.unknowns:
            result.append(observed[field[unknown]])
        if gradient:
            result.append(observed.gradient(field[problem.unknowns[0]], dimension=2))
    return program, result


def test_catalog_preserves_two_native_routes_for_one_scientific_producer():
    program, (first, second) = observations(twice=True)
    assert first.inputs[0] is not second.inputs[0]
    assert field_provider_contract(first) == field_provider_contract(second)
    assert len(emit_field_provider_catalog(program, target="system")) == 1
    amr = emit_field_provider_catalog(program, target="amr_system")
    assert len(amr) == 2 and amr[0] != amr[1]
    assert emit_field_provider_catalog(detach_compiled_program(program), target="amr_system") == amr


def test_unknown_and_projection_contracts_are_not_shape_aliases():
    program, values = observations(joint=True, gradient=True)
    contracts = [field_provider_contract(value) for value in values]
    assert len(set(contracts)) == 3
    assert [json.loads(contract)["components"] for contract in contracts] == [1, 1, 2]
    rows = emit_field_provider_catalog(program, target="system")
    assert len(rows) == 3 and sum(row.endswith(", 2, 1);") for row in rows) == 1


def test_uniform_observation_allocates_at_execution_and_seals_after_finite_check():
    _program, (value,) = observations()
    var = {value.inputs[0].id: "packed_solution"}
    lines, prelude = [], []
    emit_field_problem_value(value, var, lines, prelude, target="system")
    body = "\n".join(lines)
    assert not prelude
    assert "UINT64_C(123456)" in body
    assert body.index("ctx.allocate_continuation_field_observation(") < body.index("pops::for_each_cell(")
    assert body.index("nonfinite field expression") < body.index("ctx.seal_continuation_field_observation(")
    assert field_provider_contract(value) == json.loads(body.split("ctx.seal_continuation_field_observation(", 1)[1].split(", " + str(value.id) + ",", 1)[0])


def test_amr_observation_snapshots_without_a_reduce_or_gradient_consumer():
    program, (value,) = observations()
    program.product(retained=value)
    var = {value.inputs[0].id: "packed_solution"}
    lines, prelude = [], []
    emit_field_problem_value(value, var, lines, prelude, target="amr_system")
    body = "\n".join(lines)
    assert "ctx.capture_hierarchy_field_observation(" in body
    assert "UINT64_C(123456), false," in body
    assert "ctx.hierarchy_field_scratch" not in body
    assert ("field_observation_snapshot", value.id) in var


def test_gradient_owns_output_and_never_fills_the_retained_source_halo():
    _program, values = observations(gradient=True)
    gradient = values[-1]
    source = gradient.inputs[0]
    var = {source.id: "retained_source", ("field_observation", source.id): source.attrs["field_problem_identity"]}
    lines, prelude = [], []
    emit_field_gradient(gradient, var, lines, prelude, target="system")
    body = "\n".join(lines)
    assert "ctx.own_structural_field(retained_source)" in body
    assert "ctx.gradient(*field_gradient_%d, *field_gradient_%d_input," % (gradient.id, gradient.id) in body
    assert "prepare_mesh_boundary_session" not in "\n".join(prelude)
    assert body.index("ctx.gradient(") < body.index("ctx.seal_continuation_field_observation(")
    assert "UINT64_C(123456)" in body


def test_retained_field_reduction_keeps_local_support_and_exact_provider_preflight():
    from pops.codegen.program_emit_field_problem import field_observation_reduction
    program, (observed,) = observations()
    retained = program.product(observed=observed)["observed"]
    reduction = program.sum(retained)
    for target in ("system", "amr_system"):
        body = field_observation_reduction(reduction, {retained.id: "retained_leaf"}, target=target)
        assert "ctx.continuation_preflight_field(retained_leaf," in body
        assert "field_problem" in body and "ctx.sum_component(retained_leaf, 0)" in body
        assert body.index("continuation_preflight_field") < body.index("ctx.sum_component")
        assert "reduce_hierarchy_field_observation" not in body
        if target == "amr_system":
            assert body.index("ctx.require_program_field_access(retained_leaf)") < body.index("ctx.sum_component")
    direct = program.sum(observed)
    body = field_observation_reduction(direct, {
        observed.id: "observed_leaf",
        ("field_observation", observed.id): observed.attrs["field_problem_identity"],
        ("field_observation_snapshot", observed.id): "exact_tower",
    }, target="amr_system")
    assert body == 'ctx.reduce_hierarchy_field_observation(exact_tower, "sum")'


def test_retained_gradient_reduction_selects_a_real_component_without_a_tower():
    import pytest
    from pops.codegen.program_emit_field_problem import field_observation_reduction
    program, values = observations(gradient=True)
    retained = program.product(gradient=values[-1])["gradient"]
    reduction = program.abs_sum_component(retained, 1)
    body = field_observation_reduction(reduction, {retained.id: "gradient_leaf"}, target="amr_system")
    assert "ctx.abs_sum_component(gradient_leaf, 1)" in body
    assert "ctx.continuation_preflight_field(gradient_leaf," in body
    from types import SimpleNamespace
    malformed = SimpleNamespace(inputs=(retained,), attrs={"kind": "abs_sum", "comp": 2})
    with pytest.raises(ValueError, match="absent component"):
        field_observation_reduction(malformed, {retained.id: "gradient_leaf"}, target="amr_system")
