"""Source-only issued-value lowering: real authored graph, no native module imported."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import pops
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_value_authority import (
    prepare_program_value_authority, instrument_value_cpp,
)
from pops.fields import FieldDiscretization, CellCenteredSecondOrder
from pops.solvers.elliptic import GeometricMG
from pops.time import Program
from pops.time.solve_outcome import FailRun
from test_field_rhs_auxiliary_inputs import _field_model


def _authored(*, auxiliary=True, implicit=False, off=None, accepted_name=None):
    model, physical_state, operator = _field_model(auxiliary=auxiliary)
    if implicit:
        body = model.local_linear_operator("stage_linear", on=physical_state, matrix=((3,),))
        implicit_operator = model.operator("stage_linear", returns=body)
    case = pops.Case("issued-field-source")
    accepted = None if accepted_name is None else case.block(accepted_name, model)
    block = case.block("selected", model)
    if accepted is not None:
        from pops.fields import FieldOperator
        operator = FieldOperator(operator.name, unknown=block[operator.unknown],
                                 equation=operator.equation, providers=block[
                                     model.module.operator_handle(operator.name)],
                                 outputs=operator.outputs)
    field = case.field(operator, FieldDiscretization(
        method=CellCenteredSecondOrder(), boundaries=(), solver=GeometricMG()))
    program = Program("issued-stage-value")
    state = program.state(block[physical_state])
    stage = program.value("stage-value", 2 * state.n, at=state.n.point)
    if implicit:
        from pops.time import LocalLinear
        from pops.solvers import DenseLU
        linear = program.value("stage-operator", implicit_operator(program=program), at=state.n.point)
        stage = program.solve(LocalLinear(operator=program.I - program.dt * linear, rhs=stage),
                              solver=DenseLU()).consume(action=FailRun())
    if off is not None:
        from pops.time import Schedule, Every, AcceptedStep
        schedule = Schedule(Every(AcceptedStep(program.clock), 2), off=off)
        stage = program._replace_value(stage, attrs={**stage.attrs, "schedule": schedule})
    field(stage).consume(action=FailRun())
    endpoint = program.value("endpoint", 1 * stage, at=state.next.point)
    program.commit(state.next, endpoint)
    provider = case.resolve(model.module.operator_handle(operator.name), block=block)
    key = model.module.operator_registry().get(operator.name).lowering["field_provider"]["key"]
    providers, route = (provider,), ({"owner_block": "selected", "key": key},)
    if accepted is not None:
        providers += (case.resolve(model.module.operator_handle(operator.name), block=accepted),)
        route += ({"owner_block": accepted_name, "key": key},)
    solve = next(node for node in program._values if node.op == "solve_fields")
    plans = {field.local_id: SimpleNamespace(
        rhs_providers=providers, native_options={
            "provider_slot": "source-only-resolved-route",
            "provider_pack": route,
            "output_route": {"components": solve.field_context.outputs},
            "boundary_kernel_required": False,
        })}
    lowered, _ = lower_and_validate(model, facade=model)
    return program, lowered, plans, state.n, stage, solve


def test_plan_seals_actual_transitive_producers_and_field_source():
    program, model, fields, state, stage, solve = _authored()
    before = program._ir_hash()
    plan = prepare_program_value_authority(program, model, fields)
    assert program._ir_hash() == before == plan.program_identity
    assert [(row.ssa, row.storage, row.storage_id, row.inputs) for row in plan.producers] == [
        (state.id, "State", -1, ()), (stage.id, "StateScratch", stage.id, (state.id,))]
    assert [(row.field_node, row.block, row.source) for row in plan.fields] == [(solve.id, 0, stage.id)]
    with pytest.raises(FrozenInstanceError):
        plan.program_identity = "detached"
    assert "ctx.install_program_value_plan(" in plan.cpp_install()[0]


def test_empty_aux_pack_preserves_v1_without_issued_value_hooks():
    program, model, fields, *_ = _authored(auxiliary=False)
    plan = prepare_program_value_authority(program, model, fields)
    assert plan.fields == plan.producers == ()
    assert plan.cpp_install() == []


def test_ticket_surrounds_actual_write_and_publishes_after_success_guard():
    program, model, fields, state, stage, _ = _authored()
    plan = prepare_program_value_authority(program, model, fields)
    variables = {("program_value_authority",): plan, state.id: "current", stage.id: "candidate"}
    lines = ["auto& candidate = ctx.scratch_state(1, 0, current);",
             "ctx.set_stage_time(1, 3);", "ctx.axpy(candidate, 2, current, dt);",
             "if (failed) throw failure;"]
    instrument_value_cpp(stage, variables, lines, 2)
    assert "begin_program_value_write" in lines[2]
    assert "ctx.program_value(%d, 0, current)" % state.id in lines[2]
    assert lines[3].startswith("ctx.axpy")
    assert lines[-2] == "if (failed) throw failure;"
    assert "complete_program_value_write(std::move(value_write_" in lines[-1]


def test_full_source_emits_plan_and_successful_value_production():
    program, model, fields, state, stage, _ = _authored()
    from pops.codegen.program_codegen import emit_cpp_program
    source = emit_cpp_program(program, model=model, field_plans=fields)
    assert "ctx.install_program_value_plan(" in source
    assert "ctx.capture_program_value(%d, 0, u%d)" % (state.id, state.id) in source
    write = source.index("ctx.begin_program_value_write(%d," % stage.id)
    success = source.index("ctx.complete_program_value_write(std::move(value_write_%d))" % stage.id)
    assert write < source.index("ctx.axpy(", write) < success
    assert "ctx.solve_fields_from_program_values_at(" in source


def test_consumed_local_solve_keeps_success_guard_and_exact_alias_chain():
    program, model, fields, _, stage, _ = _authored(implicit=True)
    plan = prepare_program_value_authority(program, model, fields)
    consumed = plan.producer(stage.id)
    outcome = plan.producer(consumed.inputs[0])
    solved = plan.producer(outcome.inputs[0])
    assert consumed.storage == outcome.storage == "Alias"
    assert solved.storage == "StateScratch"
    from pops.codegen.program_codegen import emit_cpp_program
    source = emit_cpp_program(program, model=model, field_plans=fields)
    completion = source.index("ctx.complete_program_value_write(std::move(value_write_%d))" % solved.ssa)
    assert source.index(".solved_value_available()") < completion
    assert completion < source.index("ctx.alias_program_value(%d," % outcome.ssa)
    assert source.index("ctx.alias_program_value(%d," % consumed.ssa) < source.index(
        "ctx.solve_fields_from_program_values_at(")


@pytest.mark.parametrize("policy_name,write_action", [("Hold", "cache_restore_scratch"),
                                                     ("Zero", "set_val")])
def test_off_cadence_write_issues_only_its_declared_branch(policy_name, write_action):
    from pops import time
    program, model, fields, _, stage, _ = _authored(off=getattr(time, policy_name)())
    plan = prepare_program_value_authority(program, model, fields)
    assert plan.producer(stage.id).input_branches == ((),)
    from pops.codegen.program_codegen import emit_cpp_program
    source = emit_cpp_program(program, model=model, field_plans=fields)
    off = source.index("} else {")
    begin = source.index("ctx.begin_program_value_write(%d," % stage.id, off)
    action = source.index(write_action, begin)
    complete = source.index("ctx.complete_program_value_write(std::move(value_write_%d))" % stage.id, action)
    assert begin < action < complete
    assert ", {});" in source[begin:source.index("\n", begin)]


def test_detached_source_is_rejected_before_an_authority_plan_exists():
    program, model, fields, _, stage, solve = _authored()
    detached = SimpleNamespace(id=stage.id, block=stage.block)
    object.__setattr__(solve, "inputs", (detached,))
    with pytest.raises(ValueError, match="detached SSA source"):
        prepare_program_value_authority(program, model, fields)


@pytest.mark.parametrize("accepted_name", ["unadvanced", "renamed_contributor"])
def test_undeclared_accepted_owner_appends_exact_native_name_after_active_owner(accepted_name):
    program, model, fields, _, stage, solve = _authored(accepted_name=accepted_name)
    assert [block.local_id for block in program._block_indices()] == ["selected"]
    plan = prepare_program_value_authority(program, model, fields)
    assert plan.block_names == ("selected", accepted_name)
    assert sorted((edge.block, edge.source, edge.accepted_only) for edge in plan.fields) == [
        (0, stage.id, False), (1, -1, True)]
    from pops.codegen.program_codegen import emit_cpp_program
    source = emit_cpp_program(program, model=model, field_plans=fields)
    assert 'case 0: return "selected";' in source
    assert 'case 1: return "%s";' % accepted_name in source
    assert "{%d, 1, -1, true}" % solve.id in source
    assert "ProgramValueStorage::State, -1" in source
    assert "capture_program_value(-1" not in source


def test_field_route_cannot_replace_accepted_provider_by_another_owner():
    program, model, fields, *_ = _authored(accepted_name="unadvanced")
    fields["recover"].native_options["provider_pack"][1]["owner_block"] = "selected"
    with pytest.raises(ValueError, match="authenticated provider owner"):
        prepare_program_value_authority(program, model, fields)


def test_field_point_follows_authored_partition_and_actual_runtime_cursor():
    from fractions import Fraction
    from pops.time import StagePoint, TimePoint
    from pops.codegen.program_emit_field_routes import field_point_cpp
    program, _, _, _, _, solve = _authored()
    node = SimpleNamespace(id=solve.id, clock=program.clock,
                          attrs={"evaluation_partition": "late"},
                          point=StagePoint("separate-coordinates", {
                              "early": TimePoint(program.clock, offset=Fraction(1, 4)),
                              "late": TimePoint(program.clock, offset=Fraction(3, 4))}))
    source = "\n".join(field_point_cpp(program, node, "exact-field"))
    assert "partition_slot = 2;" in source
    for member, runtime_member in (("time", "physical_time"), ("dt", "dt"),
                                   ("step", "tick"), ("substep", "substep"), ("level", "level")):
        assert ".%s = field_boundary_point_%d.%s;" % (member, solve.id, runtime_member) in source
    assert "ctx.physical_time()" not in source
    node.attrs["evaluation_partition"] = "foreign"
    with pytest.raises(ValueError, match="not declared"):
        field_point_cpp(program, node, "exact-field")
