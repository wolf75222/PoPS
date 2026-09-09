"""Explicit consumed observations own exact provider inputs and their stage context."""
from __future__ import annotations

import pytest
import pops
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.time import FailRun, FixedDt
from tests.python.unit.fields.test_program_field_problem import field_case


def publication_case():
    case, field, problem, program, values, point = field_case(publication_fields=True)
    model = case._block_registry.spec("first")["model"]
    module = model.module
    target = case.blocks()["first"][module.field_handle(module.field_spaces()["fields"])]
    solution = field.observe(program.solve(field, values=values, at=point).consume(action=FailRun()))
    gradient = solution.gradient(field[problem.unknowns[0]], dimension=2)
    context = solution.publish({
        (target, "observed_phi"): solution[field[problem.unknowns[0]]],
        (target, "observed_gx"): (gradient, 0),
        (target, "observed_gy"): (gradient, 1),
    })
    for handle in values:
        state = program.state(handle)
        program.commit(state.next, program.value("unchanged_" + state.n.name, 1 * state.n, at=state.next.point))
    program.step_strategy(FixedDt(0.125))
    case.program(program)
    frame = model._frame
    layout = Uniform(CartesianGrid(frame=frame, cells=(16, 16), periodic=PeriodicAxes(frame.axes)))
    return case, layout, program, context, target


def test_consumed_field_publication_resolves_exact_existing_provider_and_emits_transaction():
    case, layout, program, context, target = publication_case()
    resolved = pops.resolve(pops.validate(case), layout=layout)
    plan = resolved.blocks[0].resolved_operations
    claims = plan.provider_evidence["program_field_publications"]
    assert len(claims) == 3 and claims[0]["key"]["space_name"] == target.local_id
    assert claims[0]["producer"] == context.attrs["field_problem_identity"]
    assert context.field_context.stage_sources
    code = emit_cpp_program(program, model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert "ctx.publish_field_components(" in code
    assert "provider:pops.field-problem" in code
    publication = code.index("ctx.publish_field_components(")
    assert "ctx.prepare_provider_values(" in code[:publication]
    assert "observed_static" in code
    retained = [
        row
        for row in resolved.continuation_transitions.to_data()["objects"]
        if row["kind"] == "auxiliary" and row["name"] == "first"
        and row["validity"].get("space_kind") == "field"
    ]
    by_component = {row["validity"]["component"]: row for row in retained}
    assert set(by_component) == {
        "observed_phi", "observed_gx", "observed_gy", "observed_static",
    }
    assert all(
        by_component[name]["transitions"]["initialization"]["action"] == "invalidate"
        for name in ("observed_phi", "observed_gx", "observed_gy")
    )
    assert by_component["observed_static"]["transitions"]["initialization"]["action"] == "transfer"

    # The retained destination must remain tied to the exact Case field problem that publishes it.
    from types import SimpleNamespace
    from pops.runtime._continuation_transitions import derive_continuation_transitions

    missing_owner = SimpleNamespace(
        target=resolved.target,
        time=resolved.time,
        blocks=resolved.blocks,
        field_plans=resolved.field_plans,
        program_field_plans={},
        bootstrap_plan=resolved.bootstrap_plan,
    )
    with pytest.raises(ValueError, match="no exact resolved Program owner"):
        derive_continuation_transitions(missing_owner)

    from copy import deepcopy

    block = resolved.blocks[0]
    evidence = block.resolved_operations.to_data()
    evidence["provider_evidence"]["program_field_publications"][0]["producer"] += "/foreign"
    foreign_block = SimpleNamespace(
        name=block.name,
        instance_owner_qid=block.instance_owner_qid,
        state_identities=block.state_identities,
        resolved_operations=SimpleNamespace(to_data=lambda: deepcopy(evidence)),
    )
    foreign_producer = SimpleNamespace(
        target=resolved.target,
        time=resolved.time,
        blocks=(foreign_block, *resolved.blocks[1:]),
        field_plans=resolved.field_plans,
        program_field_plans=resolved.program_field_plans,
        bootstrap_plan=resolved.bootstrap_plan,
    )
    with pytest.raises(ValueError, match="no exact resolved Program owner"):
        derive_continuation_transitions(foreign_producer)


def test_publication_rejects_unconsumed_or_unqualified_input():
    case, field, problem, program, values, point = field_case()
    outcome = program.solve(field, values=values, at=point)
    with pytest.raises(TypeError, match="consumed"):
        field.observe(outcome)
    solution = field.observe(outcome.consume(action=FailRun()))
    with pytest.raises(TypeError, match="block-qualified"):
        solution.publish({field[problem.unknowns[0]]: solution[field[problem.unknowns[0]]]})


@pytest.mark.parametrize("transport", (False, True))
def test_actual_source_and_flux_consumers_resolve_field_signature(transport):
    from tests.python.integration.runtime.test_public_field_consumers import consumer_case
    case, layout = consumer_case(16, transport=transport)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    from pops.time._program.detach import detach_compiled_program
    from pops.codegen.program_graph_lowering import emit_program_graph
    detached = detach_compiled_program(case._time)
    code = emit_program_graph(detached.to_graph(), lowering_program=detached,
                              model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert "ctx.publish_field_components(" in code
    assert "potential_grad_y" in code


@pytest.mark.parametrize("corruption", ("component", "space", "source_component"))
def test_publication_node_authenticates_destination_and_selected_width(corruption):
    from pops.fields._program_publication import validate_field_publication
    case, layout, program, context, target = publication_case()
    # IR replacement is a deliberate counterexample to authoring checks: admission reauthenticates.
    rows = tuple(dict(row) for row in context.attrs["bindings"])
    if corruption == "component":
        rows[0]["component"] = "not_declared"
    elif corruption == "source_component":
        rows[0]["source_component"] = 1
    if corruption == "space":
        program._replace_value(context, space=None)
    else:
        program._replace_value(context, attrs={**context.attrs, "bindings": rows})
    replaced = next(value for value in program._values if value.id == context.id)
    with pytest.raises(ValueError, match="component|field space"):
        validate_field_publication(replaced)


def test_independent_driver_publication_retains_actual_consumer_ssa_without_solve_dependency():
    from tests.python.integration.runtime.test_public_field_consumers import consumer_case
    case, layout = consumer_case(16)
    program = case._time
    context = next(node for node in program._values if node.op == "field_publication")
    load = next(node for node in program._values if node.op == "field_problem_load")
    assert {value.block.local_id for value in load.inputs} == {"driver"}
    assert context.inputs[-1].vtype == "state" and context.inputs[-1].block.local_id == "fluid"
    assert {block.local_id for block, _ in context.field_context.stage_sources} == {"driver", "fluid"}
    pops.resolve(pops.validate(case), layout=layout)


@pytest.mark.parametrize("corruption", ("future_point", "fresh_equal_time_stage", "foreign_state", "duplicate"))
def test_supplemental_consumer_state_refuses_stale_or_conflicting_authority(corruption):
    from pops.fields._program_publication import validate_field_publication
    from tests.python.integration.runtime.test_public_field_consumers import consumer_case
    case, _layout = consumer_case(16)
    program = case._time
    context = next(node for node in program._values if node.op == "field_publication")
    state = context.inputs[-1]
    if corruption in ("future_point", "fresh_equal_time_stage"):
        offset = 1 if corruption == "future_point" else 0
        stale = program.value("future consumer", 1 * state, at=program.stage("later", c=offset))
        inputs, attrs = (*context.inputs[:-1], stale), context.attrs
    elif corruption == "foreign_state":
        load = next(node for node in program._values if node.op == "field_problem_load")
        inputs, attrs = (*context.inputs[:-1], load.inputs[0]), context.attrs
    else:
        inputs = (*context.inputs, state)
        attrs = {**context.attrs, "consumer_states": 2 * context.attrs["consumer_states"]}
    replaced = program._new("fields", "field_publication", inputs, attrs, "corrupted context", None,
                            field_context=context.field_context, space=context.space,
                            point=context.point, inherit_state_ref=False)
    with pytest.raises(ValueError, match="stage point|state mapping|repeated block"):
        validate_field_publication(replaced)


def test_same_model_instances_cannot_alias_distinct_consumed_publications():
    case, field, problem, program, values, point = field_case(publication_fields=True)
    first = case.blocks()["first"]
    model = case._block_registry.spec("first")["model"]
    module = model.module
    declaration = module.field_handle(module.field_spaces()["fields"])
    solution = field.observe(program.solve(field, values=values, at=point).consume(action=FailRun()))
    gradient = solution.gradient(field[problem.unknowns[0]], dimension=2)
    second_instance = case.block("repeated_definition", model)
    with pytest.raises(ValueError, match="provider key across block instances"):
        solution.publish({(first[declaration], "observed_gx"): (gradient, 0),
                          (second_instance[declaration], "observed_gy"): (gradient, 1)})


@pytest.mark.parametrize("transport", (False, True))
def test_provider_storage_halo_follows_actual_face_reads_without_inventing_boundaries(transport):
    from tests.python.integration.runtime.test_public_field_consumers import consumer_case
    from pops.codegen._compile_emit import _emit_auxiliary_route_registration
    from pops.codegen.program_emit_kernels import _model_impl
    case, layout = consumer_case(16, transport=transport)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    source = _emit_auxiliary_route_registration(_model_impl(graph.model_for_block(case.blocks()["fluid"])))
    output_rows = [line for line in source.splitlines() if "std::vector<Output>" in line]
    x = next(line for line in output_rows if '"potential_grad_x"' in line)
    y = next(line for line in output_rows if '"potential_grad_y"' in line)
    assert "halo[axis] = 0" in x
    assert ("halo[axis] = 1" if transport else "halo[axis] = 0") in y
    assert "BoundaryKind::inherit_topology" in x and "BoundaryKind::inherit_topology" in y
