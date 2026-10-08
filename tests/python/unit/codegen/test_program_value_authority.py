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


def _authored_diffusive(*, off=None, partition=False, source=False):
    from pops import math
    from pops.numerics import Diffusion, DiscretizationPlan
    model, physical_state, operator = _field_model(name="delegated-diffusion-field")
    flux = model.diffusive_flux("actual-conduction", state=physical_state,
                               value=.1 * math.grad(physical_state[0]))
    expression = math.div(flux)
    if source:
        reaction = model.source("actual-reaction", on=physical_state, value=(physical_state[0],))
        expression += .25 * reaction
    rate = model.rate("actual-balance", equation=math.ddt(physical_state) == expression)
    case = pops.Case("delegated-diffusion-value")
    block = case.block("selected", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, Diffusion(flux=flux))
    case.numerics(numerics, block=block)
    field = case.field(operator, FieldDiscretization(
        method=CellCenteredSecondOrder(), boundaries=(), solver=GeometricMG()))
    program = Program("delegated-diffusion-value")
    state = program.state(block[physical_state])
    diffusion = rate(state.n)
    if partition:
        from pops.time import StagePoint, TimePoint
        from fractions import Fraction
        point = StagePoint("actual-diffusion-stage", {
            "selected": TimePoint(program.clock, offset=Fraction(1, 3))})
        diffusion = program._replace_value(diffusion, attrs={
            **diffusion.attrs, "evaluation_partition": "selected"}, point=point)
    if off is not None:
        from pops.time import Schedule, Every, AcceptedStep
        schedule = Schedule(Every(AcceptedStep(program.clock), 2), off=off)
        diffusion = program._replace_value(diffusion, attrs={**diffusion.attrs, "schedule": schedule})
    stage = program.value("actual-predictor", state.n + program.dt * diffusion, at=state.next.point)
    field(stage).consume(action=FailRun())
    # Partition/cadence cases observe a produced candidate without claiming an
    # unsupported accepted diffusive quadrature. The ordinary witness advances Euler.
    endpoint = (program.value("unchanged-endpoint", 1 * state.n, at=state.next.point)
                if partition or off is not None else stage)
    program.commit(state.next, endpoint)
    provider = case.resolve(model.module.operator_handle(operator.name), block=block)
    key = model.module.operator_registry().get(operator.name).lowering["field_provider"]["key"]
    solve = next(node for node in program._values if node.op == "solve_fields")
    plans = {field.local_id: SimpleNamespace(rhs_providers=(provider,), native_options={
        "provider_slot": "source-only-resolved-route",
        "provider_pack": ({"owner_block": "selected", "key": key},),
        "output_route": {"components": solve.field_context.outputs},
        "boundary_kernel_required": False,
    })}
    emitter, _ = lower_and_validate(model, facade=model)
    return program, emitter, plans, state.n, diffusion, stage, solve


def test_diffusive_field_source_seals_actual_rhs_allocation_identity():
    program, model, fields, state, diffusion, stage, solve = _authored_diffusive()
    plan = prepare_program_value_authority(program, model, fields)
    row = plan.producer(diffusion.id)
    assert (row.storage, row.storage_id, row.subslot, row.inputs) == (
        "RhsScratch", diffusion.id, 0, (state.id,))
    assert plan.producer(stage.id).inputs == (state.id, diffusion.id)
    assert plan.fields[0].field_node == solve.id and plan.fields[0].source == stage.id


@pytest.mark.parametrize("target", ["system", "amr_system"])
def test_diffusive_publication_follows_real_stage_write_and_successful_guard_tail(target):
    from pops.codegen.program_codegen import emit_cpp_program
    program, model, fields, state, diffusion, stage, _ = _authored_diffusive(
        source=True, partition=True)
    source = emit_cpp_program(program, model=model, field_plans=fields, target=target)
    allocation = source.index("& diffusive_rhs_%d = ctx.rhs_scratch(%d, 0," % (
        diffusion.id, diffusion.id))
    point = source.index("ctx.set_stage_time(1, 3);", allocation)
    begin = source.index("ctx.begin_program_value_write(%d," % diffusion.id, point)
    preparation = source.index("ctx.require_cartesian_generated_operator(", begin)
    apply = source.index(".apply(", preparation)
    stability = source.index('"combined_transport_diffusion_stability"', apply)
    last_source = source.index("ctx.axpy(diffusive_rhs_%d,0.25,diffusive_source_" % diffusion.id,
                               stability)
    complete = source.index("ctx.complete_program_value_write(std::move(value_write_%d))" %
                            diffusion.id, last_source)
    consumer = source.index("ctx.solve_fields_from_program_values_at(", complete)
    assert allocation < point < begin < preparation < apply < stability < last_source < complete < consumer
    assert "ctx.program_value(%d, 0, u%d)" % (state.id, state.id) in source[begin:preparation]
    assert 'catch (const pops::runtime::program::DiffusiveEvaluationError& error)' in source[apply:complete]
    assert "diffusive_face_evaluation" in source[apply:complete]


@pytest.mark.parametrize("policy_name, action", [("Hold", "cache_restore_scratch"),
                                               ("Zero", "set_val")])
def test_scheduled_diffusion_shares_allocation_without_hoisting_physical_preparation(policy_name, action):
    from pops import time
    from pops.codegen.program_codegen import emit_cpp_program
    program, model, fields, _, diffusion, _, _ = _authored_diffusive(
        off=getattr(time, policy_name)(), partition=True)
    source = emit_cpp_program(program, model=model, field_plans=fields)
    allocation = source.index("& diffusive_rhs_%d = ctx.rhs_scratch(%d, 0," % (
        diffusion.id, diffusion.id))
    due = source.index("if (ctx.schedule_decision(%d," % diffusion.id, allocation)
    point = source.index("ctx.set_stage_time(1, 3);", due)
    prepare = source.index("ctx.require_cartesian_generated_operator(", point)
    off = source.index("} else {", prepare)
    alternate = source.index("ctx.begin_program_value_write(%d," % diffusion.id, off)
    alternate_action = source.index(action, alternate)
    complete = source.index("ctx.complete_program_value_write(std::move(value_write_%d))" %
                            diffusion.id, alternate_action)
    assert allocation < due < point < prepare < off < alternate < alternate_action < complete
    assert ", {});" in source[alternate:source.index("\n", alternate)]
    assert "ctx.set_stage_time" not in source[allocation:due]
    assert "PreparedDiffusion" not in source[allocation:due]
    assert "ctx.require_cartesian_generated_operator" not in source[allocation:due]


@pytest.mark.parametrize("changes", [
    {"version": 2}, {"ssa": -7}, {"block": 1}, {"output": "foreign_storage"},
    {"storage_id": -7}, {"subslot": 1}, {"setup_end": 0}, {"evaluation_end": 99},
    {"ssa": True},
])
def test_delegated_boundary_cannot_publish_foreign_or_unallocated_storage(changes):
    from dataclasses import replace
    from pops.codegen.program_value_authority import (
        DelegatedOutputPublication, delegated_output_boundaries,
    )
    program, model, fields, state, diffusion, *_ = _authored_diffusive()
    plan = prepare_program_value_authority(program, model, fields)
    var = {("program_value_authority",): plan, state.id: "actual_input", diffusion.id: "actual_rhs"}
    lines = ["auto& actual_rhs = ctx.rhs_scratch(1, 0, actual_input);", "physical_evaluation();"]
    boundary = DelegatedOutputPublication(
        diffusion.id, 0, "RhsScratch", diffusion.id, 0, "actual_rhs", 0, 1, 1, 2)
    with pytest.raises(ValueError):
        delegated_output_boundaries(diffusion, var, lines, 0, 0, replace(boundary, **changes))


def test_implicit_diffusion_delegate_preserves_prepared_result_api_and_cpp_order():
    from pops.codegen.program_emit_kernels import ProgramProviderPlans
    from pops.codegen.program_emit_diffusion import _emit_diffusive_rhs
    program, model, _, state, diffusion, *_ = _authored_diffusive()
    # Residual delegates own their repeated output; they are not a top-level SSA publication.
    lines = []
    variables = {state.id: "actual_input"}
    provider_plans = ProgramProviderPlans()
    returned = _emit_diffusive_rhs(diffusion, variables, lines, model, provider_plans, 0,
                                    "system", prepared_var="retained_implicit_operator")
    assert returned == "retained_implicit_operator"
    assert "begin_program_value_write" not in "\n".join(lines)
    assert "retained_implicit_operator.apply(" in "\n".join(lines)
    assert "ctx.rhs_scratch(%d, 0, actual_input)" % diffusion.id in lines[0]


def test_late_refusal_remains_after_rate_and_field_candidates_before_atomic_commit():
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.time.solve_outcome import RejectAttempt
    program, model, fields, _, diffusion, stage, _ = _authored_diffusive()
    # A deliberately impossible late observation tests ordering without replacing
    # the physics or suppressing any generated stability/publication guard.
    program.guard("late-diffusive-refusal", stage, program.norm2(stage) < 0,
                  action=RejectAttempt())
    source = emit_cpp_program(program, model=model, field_plans=fields)
    complete = source.index("ctx.complete_program_value_write(std::move(value_write_%d))" %
                            diffusion.id)
    field = source.index("ctx.solve_fields_from_program_values_at(", complete)
    refusal = source.index("throw pops::runtime::program::StepAttemptRejected", field)
    commit = source.index("ctx.commit_many(", refusal)
    assert complete < field < refusal < commit
    assert "acceptance guard 'late-diffusive-refusal' failed" in source[refusal:commit]
